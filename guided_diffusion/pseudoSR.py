from scipy.signal import convolve2d as conv2
import numpy as np
import torch
import torch.nn as nn
import copy
import collections
from torchvision.transforms import functional as VF
from guided_diffusion.imresize_pseudoSR import (
    imresize,
    imresize_efficient,
    calc_strides,
)


class Filter_Layer(nn.Module):
    def __init__(self, filter, pre_filter_func, post_filter_func=None, num_channels=3):
        super(Filter_Layer, self).__init__()
        self.Filter_OP = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=filter.shape,
            bias=False,
            groups=num_channels,
        )
        self.Filter_OP.weight = nn.Parameter(
            data=torch.from_numpy(
                np.tile(
                    np.expand_dims(np.expand_dims(filter, 0), 0),
                    reps=[num_channels, 1, 1, 1],
                )
            ).type(torch.FloatTensor),
            requires_grad=False,
        )
        self.Filter_OP.filter_layer = True
        self.pre_filter_func = pre_filter_func
        self.post_filter_func = (
            (lambda x: x) if post_filter_func is None else post_filter_func
        )

    def forward(self, x):
        x = self.pre_filter_func(x)
        x = self.Filter_OP(x)
        x = self.post_filter_func(x)
        return x


class pseudoSR:
    NFFT_add = 36

    def __init__(self, conf, upscale_kernel=None, kernel_indx=0):
        self.conf = conf
        self.ds_factor = np.array(conf.scale_factor, dtype=np.int32)
        assert (
            np.round(self.ds_factor) == self.ds_factor
        ), "Currently only supporting integer scale factors"
        assert (
            upscale_kernel is None
            or isinstance(upscale_kernel, str)
            or isinstance(upscale_kernel, np.ndarray)
        ), "To support given kernels, change the Return_Invalid_Margin_Size_in_LR function and make sure everything else works"
        self.ds_kernel, self.pre_stride, self.post_stride = Return_kernel(
            self.ds_factor, upscale_kernel=upscale_kernel, kernel_indx=kernel_indx
        )
        self.ds_kernel_invalidity_half_size_LR = self.Return_Invalid_Margin_Size_in_LR(
            "ds_kernel", self.conf.filter_pertubation_limit
        )
        self.compute_inv_hTh()
        self.invalidity_margins_LR = (
            2 * self.ds_kernel_invalidity_half_size_LR
            + self.inv_hTh_invalidity_half_size
        )
        self.invalidity_margins_HR = self.ds_factor * self.invalidity_margins_LR

    def Return_Invalid_Margin_Size_in_LR(self, filter, max_allowed_perturbation):
        TEST_IM_SIZE = 100
        assert filter in ["ds_kernel", "inv_hTh"]
        if filter == "ds_kernel":
            output_im = imresize(
                np.ones(
                    [self.ds_factor * TEST_IM_SIZE, self.ds_factor * TEST_IM_SIZE, 3],
                    dtype=np.float32,
                ),
                [1 / self.ds_factor],
                use_zero_padding=True,
            )
        elif filter == "inv_hTh":
            output_im = conv2(
                np.ones([TEST_IM_SIZE, TEST_IM_SIZE, 3], dtype=np.float32),
                self.inv_hTh,
                mode="same",
            )
        output_im = output_im[:, :, 0]
        output_im /= output_im[int(TEST_IM_SIZE / 2), int(TEST_IM_SIZE / 2)]
        output_im[output_im <= 0] = (
            max_allowed_perturbation / 2
        )  # Negative output_im are hella invalid... (and would not be identified as such without this line since I'm taking their log).
        invalidity_mask = np.exp(-np.abs(np.log(output_im))) < max_allowed_perturbation
        # Finding invalid shoulder size, by searching for the index of the deepest invalid pixel, to accomodate cases of non-continuous invalidity:
        margin_sizes = [
            np.argwhere(
                invalidity_mask[: int(TEST_IM_SIZE / 2), int(TEST_IM_SIZE / 2)]
            )[-1][0]
            + 1,
            np.argwhere(
                invalidity_mask[int(TEST_IM_SIZE / 2), : int(TEST_IM_SIZE / 2)]
            )[-1][0]
            + 1,
        ]
        margin_sizes = np.max(margin_sizes) * np.ones([2]).astype(margin_sizes[0].dtype)
        return np.max(margin_sizes)

    def WrapArchitecture_PyTorch(self, grayscale=False):
        # assert (
        #     pytorch_loaded
        # ), "Failed to load PyTorch - Necessary for this function of pseudoSR"
        self.loss_mask = None
        returnable = pseudoSR_PyTorch(self, grayscale=grayscale)
        self.OP_names = [
            m[0] for m in returnable.named_modules() if "Filter_OP" in m[0]
        ]
        return returnable

    def compute_inv_hTh(self):
        hTh = conv2(self.ds_kernel, np.rot90(self.ds_kernel, 2)) * self.ds_factor**2
        hTh = Aliased_Down_Sampling(hTh, self.ds_factor)
        pad_pre = pad_post = np.array(self.NFFT_add / 2, dtype=np.int32)
        hTh_fft = np.fft.fft2(
            np.pad(
                hTh,
                ((pad_pre, pad_post), (pad_pre, pad_post)),
                mode="constant",
                constant_values=0,
            )
        )
        # When ds_kernel is wide, some frequencies get completely wiped out, which causes instability when hTh is inverted. I therfore bound this filter's magnitude from below in the Fourier domain:
        magnitude_increasing_map = np.maximum(
            1, self.conf.lower_magnitude_bound / np.abs(hTh_fft)
        )
        hTh_fft = hTh_fft * magnitude_increasing_map
        # Now inverting the filter:
        self.inv_hTh = np.real(np.fft.ifft2(1 / hTh_fft))
        # Making sure the filter's maximal value sits in its middle:
        max_row = np.argmax(self.inv_hTh) // self.inv_hTh.shape[0]
        max_col = np.mod(np.argmax(self.inv_hTh), self.inv_hTh.shape[0])
        if not np.all(
            np.equal(
                np.ceil(np.array(self.inv_hTh.shape) / 2),
                np.array([max_row, max_col]) - 1,
            )
        ):
            half_filter_size = np.min(
                [
                    self.inv_hTh.shape[0] - max_row - 1,
                    self.inv_hTh.shape[0] - max_col - 1,
                    max_row,
                    max_col,
                ]
            )
            self.inv_hTh = self.inv_hTh[
                max_row - half_filter_size : max_row + half_filter_size + 1,
                max_col - half_filter_size : max_col + half_filter_size + 1,
            ]

        self.inv_hTh_invalidity_half_size = 26  # self.Return_Invalid_Margin_Size_in_LR('inv_hTh',self.conf.filter_pertubation_limit)
        margins_2_drop = (
            self.inv_hTh.shape[0] // 2 - 26
        )  # self.Return_Invalid_Margin_Size_in_LR('inv_hTh',self.conf.desired_inv_hTh_energy_portion)
        if margins_2_drop > 0:
            self.inv_hTh = self.inv_hTh[
                margins_2_drop:-margins_2_drop, margins_2_drop:-margins_2_drop
            ]


class pseudoSR_PyTorch(nn.Module):
    def __init__(self, pseudoSR, grayscale=False):
        super(pseudoSR_PyTorch, self).__init__()
        num_channels = 1 if grayscale else 3
        self.ds_factor = pseudoSR.ds_factor
        self.conf = pseudoSR.conf
        inv_hTh_padding = np.floor(np.array(pseudoSR.inv_hTh.shape) / 2).astype(
            np.int32
        )
        Replication_Padder = nn.ReplicationPad2d(
            (
                inv_hTh_padding[1],
                inv_hTh_padding[1],
                inv_hTh_padding[0],
                inv_hTh_padding[0],
            )
        )
        self.Conv_LR_with_Inv_hTh_OP = Filter_Layer(
            pseudoSR.inv_hTh,
            pre_filter_func=Replication_Padder,
            num_channels=num_channels,
        )
        downscale_antialiasing = np.rot90(pseudoSR.ds_kernel, 2)
        upscale_antialiasing = pseudoSR.ds_kernel * pseudoSR.ds_factor**2
        pre_stride, post_stride = calc_strides(None, pseudoSR.ds_factor)
        Upscale_Padder = lambda x: nn.functional.pad(
            x, (pre_stride[1], post_stride[1], 0, 0, pre_stride[0], post_stride[0])
        )
        Aliased_Upscale_OP = lambda x: Upscale_Padder(x.unsqueeze(4).unsqueeze(3)).view(
            [
                x.size()[0],
                x.size()[1],
                pseudoSR.ds_factor * x.size()[2],
                pseudoSR.ds_factor * x.size()[3],
            ]
        )
        antialiasing_padding = np.floor(np.array(pseudoSR.ds_kernel.shape) / 2).astype(
            np.int32
        )
        antialiasing_Padder = nn.ReplicationPad2d(
            (
                antialiasing_padding[1],
                antialiasing_padding[1],
                antialiasing_padding[0],
                antialiasing_padding[0],
            )
        )
        self.Upscale_OP = Filter_Layer(
            upscale_antialiasing,
            pre_filter_func=lambda x: antialiasing_Padder(Aliased_Upscale_OP(x)),
            num_channels=num_channels,
        )
        Reshaped_input = lambda x: x.view(
            [
                x.size()[0],
                x.size()[1],
                int(x.size()[2] / self.ds_factor),
                self.ds_factor,
                int(x.size()[3] / self.ds_factor),
                self.ds_factor,
            ]
        )
        Aliased_Downscale_OP = lambda x: Reshaped_input(x)[
            :, :, :, pre_stride[0], :, pre_stride[1]
        ]
        self.DownscaleOP = Filter_Layer(
            downscale_antialiasing,
            pre_filter_func=antialiasing_Padder,
            post_filter_func=lambda x: Aliased_Downscale_OP(x),
            num_channels=num_channels,
        )
        self.ds_kernel = pseudoSR.ds_kernel
        self.pre_stride, self.post_stride = pre_stride, post_stride

    def A_pinv(self, LR, generated_image=None, jpeg_decode=None, jpeg_encode=None):
        LR = LR[
            :, -3:, :, :
        ]  # Handling the case of adding noise channel(s) - Using only last 3 image channels

        if jpeg_decode is None:
            jpeg_decode = lambda x: x

        if jpeg_encode is None:
            jpeg_encode = lambda x: x

        # generated_image torch tensor torch.Size([1, 3, 512, 512])
        if generated_image is not None:
            # generated_image=torch.ones([1,3,512,512],dtype=torch.float32).to(LR.device)
            assert np.all(np.mod(generated_image.size()[2:], self.ds_factor) == 0)
            ortho_2_NS_HR_component = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(LR))
            ortho_2_NS_generated = self.Upscale_OP(
                self.Conv_LR_with_Inv_hTh_OP(
                    jpeg_decode(jpeg_encode(self.DownscaleOP(generated_image)))
                )
            )
            NS_HR_component = generated_image - ortho_2_NS_generated
            if self.conf.sigmoid_range_limit:
                NS_HR_component = torch.tanh(NS_HR_component) * (
                    self.conf.input_range[1] - self.conf.input_range[0]
                )
            output = (
                ortho_2_NS_generated - ortho_2_NS_HR_component
            )  # NS_HR_component + ortho_2_NS_HR_component
            return output
        else:
            ortho_2_NS_HR_component = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(LR))
            output = ortho_2_NS_HR_component
            return output

    def A(self, HR, scale_factor=1.0, use_zero_padding=False, kk=None):
        # res = kk - self.ds_kernel
        # assert np.abs(res).sum() < np.finfo(np.float32).eps
        LR = imresize_efficient(
            HR,
            self.ds_kernel,
            scale_factor,
            None,
            self.pre_stride,
            self.post_stride,
            use_zero_padding=use_zero_padding,
        )
        return LR

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        # print(sigma_t.shape, a.shape)
        # print(144, sigma_t[...,:2], a[...,:2])

        if sigma_t.mean() < (a * sigma_y).mean():
            factor = sigma_t * (1 - eta**2) ** 0.5 / a / sigma_y
            return vec * factor
        else:
            return vec

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon=None):
        if (sigma_t).mean() >= (a * sigma_y).mean():
            factor = torch.sqrt(sigma_t**2 - a**2 * sigma_y**2)
            return vec * factor
        else:
            return vec * sigma_t * eta


def Aliased_Down_Sampling(array, factor):
    pre_stride, post_stride = calc_strides(array, 1 / factor, align_center=True)
    if array.ndim > 2:
        array = array[pre_stride[0] :: factor, pre_stride[1] :: factor, :]
    else:
        array = array[pre_stride[0] :: factor, pre_stride[1] :: factor]
    return array


def Aliased_Down_Up_Sampling(array, factor):
    half_stride_size = np.floor(factor / 2).astype(np.int32)
    input_shape = list(array.shape)
    if array.ndim > 2:
        array = array[
            half_stride_size:-half_stride_size:factor,
            half_stride_size:-half_stride_size:factor,
            :,
        ]
    else:
        array = array[
            half_stride_size:-half_stride_size:factor,
            half_stride_size:-half_stride_size:factor,
        ]
    array = np.expand_dims(np.expand_dims(array, 2), 1)
    array = np.pad(
        array,
        (
            (0, 0),
            (half_stride_size, half_stride_size),
            (0, 0),
            (half_stride_size, half_stride_size),
        ),
        mode="constant",
    )
    return np.reshape(array, newshape=input_shape)


def Return_kernel(ds_factor, upscale_kernel=None, kernel_indx=0):
    antialiasing_kernel, pre_stride, post_stride = imresize(
        None,
        [ds_factor, ds_factor],
        return_upscale_kernel=True,
        kernel=upscale_kernel,
        kernel_indx=kernel_indx,
    )
    return (
        np.rot90(antialiasing_kernel, 2).astype(np.float32) / (ds_factor**2),
        pre_stride,
        post_stride,
    )


def Pad_Image(image, margin_size):
    try:
        padding = ((margin_size, margin_size), (margin_size, margin_size))
        if image.ndim == 2:
            padding = ((margin_size, margin_size), (margin_size, margin_size))
        else:  # 3 channels
            padding = ((margin_size, margin_size), (margin_size, margin_size), (0, 0))
        return np.pad(image, pad_width=padding, mode="edge")
    except:
        print("Reproduced the BUG I" "m looking for")


def Unpad_Image(image, margin_size):
    return image[margin_size:-margin_size, margin_size:-margin_size, :]


def Get_pseudoSR_Conf(sf):
    class conf:
        scale_factor = sf
        avoid_skip_connections = False
        generate_HR_image = False
        pseudo_pseudoSR_supplement = False
        desired_inv_hTh_energy_portion = 1 - 1e-6  # 1-1e-10
        filter_pertubation_limit = 1.1
        sigmoid_range_limit = False
        lower_magnitude_bound = (
            0.01  # Lower bound on hTh filter magnitude in Fourier domain
        )

    return conf


def Adjust_State_Dict_Keys(loaded_state_dict, current_state_dict):
    if all(
        [
            ("generated_image_model" in key or "Filter" in key)
            for key in current_state_dict.keys()
        ]
    ) and not any(
        ["generated_image_model" in key for key in loaded_state_dict.keys()]
    ):  # Using pseudoSR_arch
        modified_names_dict = collections.OrderedDict()
        for key in loaded_state_dict:
            modified_names_dict["generated_image_model." + key] = loaded_state_dict[key]
        for key in [k for k in current_state_dict.keys() if "Filter" in k]:
            modified_names_dict[key] = current_state_dict[key]
        return modified_names_dict
    else:
        return loaded_state_dict
