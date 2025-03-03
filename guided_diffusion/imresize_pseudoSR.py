import torch 
import cv2
import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d
from scipy.signal import gaussian
from scipy.stats import norm
import torch.nn.functional as torchF

def imresize(im, scale_factor=None, output_shape=None, kernel=None,align_center=False, return_upscale_kernel=False,use_zero_padding=False,antialiasing=True, kernel_shift_flag=False, kernel_indx=10):
    assert kernel is None or any([word in kernel for word in ['cubic','blurry_cubic','reset_2_default']]) or isinstance(kernel,np.ndarray)
    imresize.kernels = getattr(imresize,'kernels',{})
    if scale_factor is None:
        scale_factor = [output_shape[0]/im.shape[0]]
    elif not isinstance(scale_factor,list):
        scale_factor = [scale_factor]
    assert np.round(scale_factor[0])==scale_factor[0] or np.round(1/scale_factor[0])==1/scale_factor[0],'Only supporting integer downsampling or upsampling rates'
    sf_4_kernel = np.maximum(scale_factor[0], 1 / scale_factor[0]).astype(np.int32)
    assert len(scale_factor)==1 or scale_factor[0]==scale_factor[1]
    scale_factor = scale_factor[0]
    pre_stride,post_stride = calc_strides(im,scale_factor,align_center)
    # Padding the kernel to compenstae for imbalanced padding, in the case of an even scale factor. This increases kernel size by 1 for even scale factors or 0 for odd:
    kernel_post_padding = np.maximum(0, pre_stride - post_stride)
    kernel_pre_padding = np.maximum(0, post_stride - pre_stride)
    if isinstance(kernel,np.ndarray):
        if kernel_indx >=8:
            if str(sf_4_kernel) in imresize.kernels.keys():
                print('Overriding previous kernel with given kernel...')
            kernel_2_use = Cubic_Kernel(sf_4_kernel)
            assert np.abs(1-np.sum(kernel))<np.finfo(np.float32).eps,'Supplied non-default kernel does not sum to 1'
            # I assume the supplied kernel is a downscaling kernel, while the kernel saved here should be an upscaling one:
            # kernel = np.rot90(kernel,2)
            kernel = Center_Mass(kernel,ds_factor=sf_4_kernel)*sf_4_kernel**2
            assert kernel.shape[0]==kernel.shape[1],'Only square kernels supported for now'
            assert np.all(np.mod(kernel.shape+kernel_post_padding+kernel_pre_padding-1,sf_4_kernel)==0),'Convolution-invalidated size should be an integer multiplication of sf_4_kernel'
            # kernel_2_use = convolve2d(kernel_2_use, kernel)
            imresize.kernels[str(sf_4_kernel)] = kernel#kernel_2_use#Gaussian_2D(sigma=1., size=23)[1:-2,1:-2] #kernel
        elif kernel_indx<8:
            if str(sf_4_kernel) in imresize.kernels.keys(): #Called by 'reset_2_default'
                print('Overriding previous kernel with default kernel...')
            kernel_2_use = Cubic_Kernel(sf_4_kernel)
            blur_kernel = kernel
            imresize.kernels['blur_'+str(sf_4_kernel)] = blur_kernel
            kernel_2_use = convolve2d(kernel_2_use,blur_kernel)
            imresize.kernels[str(sf_4_kernel)] = kernel_2_use
    elif str(sf_4_kernel) not in imresize.kernels.keys() or kernel=='reset_2_default':
        if str(sf_4_kernel) in imresize.kernels.keys(): #Called by 'reset_2_default'
            print('Overriding previous kernel with default kernel...')
        kernel_2_use = Cubic_Kernel(sf_4_kernel)
        if kernel is not None and 'blurry_cubic' in kernel:
            sigma = float(kernel[len('blurry_cubic_'):])
            blur_kernel = Gaussian_2D(sigma=sigma)
            imresize.kernels['blur_'+str(sf_4_kernel)] = blur_kernel
            kernel_2_use = convolve2d(kernel_2_use,blur_kernel)
        imresize.kernels[str(sf_4_kernel)] = kernel_2_use
    antialiasing_kernel = np.pad(imresize.kernels[str(sf_4_kernel)],((kernel_pre_padding[0],kernel_post_padding[0]),(kernel_pre_padding[1],kernel_post_padding[1])),mode='constant')
    if scale_factor < 1:
        antialiasing_kernel = np.rot90(antialiasing_kernel * scale_factor ** 2, 2)
        sio.savemat("rot59.mat",{"rot59":antialiasing_kernel})
    if return_upscale_kernel:
        return antialiasing_kernel, pre_stride, post_stride
    assert output_shape is None or np.all(scale_factor*np.array(im.shape[:2])==output_shape[:2])

    padding_size = np.floor(np.array(antialiasing_kernel.shape)/2).astype(np.int32)
    
    mask = np.rot90(antialiasing_kernel.astype(np.float32), 2)
    antialiasing_kernel = torch.from_numpy(np.expand_dims(np.stack([mask,mask,mask], axis=0),axis=0).transpose(1,0,2,3))
    output = []
    def filter2d(input,special_padding_size=None):
        input = torch.from_numpy(input).unsqueeze(0)
        if special_padding_size is not None:
            input = torchF.pad(input, (special_padding_size[0], special_padding_size[0], special_padding_size[1], special_padding_size[1]), "reflect")
            return torch.nn.functional.conv2d(input,     
                antialiasing_kernel,
                padding=(antialiasing_kernel.shape[0] // 2,antialiasing_kernel.shape[1] // 2) if special_padding_size is None else 0,
                groups=3).squeeze(0).squeeze(0).cpu().numpy()
    output = filter2d(im.transpose(2,0,1), special_padding_size=padding_size)
    output = output.transpose(1,2,0)
    return output[pre_stride[0]::int(1 / scale_factor),pre_stride[1]::int(1 / scale_factor),:]

def calc_strides(array,factor,align_center = False):
    integer_factor = np.maximum(factor,1/factor).astype(np.int32)
    # Overall I should pad with (integer_factor-1) zeros:
    if align_center:
        half_image_size = np.ceil(np.array(array.shape[:2])/2*(factor if factor>1 else 1))
        pre_stride = np.mod(half_image_size,integer_factor)
        pre_stride[np.equal(pre_stride,0)] = integer_factor
        pre_stride = (pre_stride-1).astype(np.int32)
        post_stride = integer_factor-pre_stride-1
    else:
        # This is an arbitrary convention for dividing the padding before and after each value (for the case of even factor). The padding of the DS kernel should comply to avoid translation.
        post_stride = (np.floor(integer_factor/2)*np.ones([2])).astype(np.int32)
        pre_stride = (integer_factor-post_stride-1).astype(np.int32)
    return pre_stride,post_stride

def Cubic_Kernel(sf):
    DELTA_SIZE = 11
    delta_im = Delta_Im(DELTA_SIZE)
    upscale_kernel = cv2.resize(delta_im, dsize=(sf * DELTA_SIZE, sf * DELTA_SIZE),interpolation=cv2.INTER_CUBIC)
    kernel_support = np.nonzero(upscale_kernel[sf * np.ceil(DELTA_SIZE / 2).astype(np.int32) - 1, :])[0]
    kernel_support = np.array([kernel_support[0], kernel_support[-1]])
    return upscale_kernel[kernel_support[0]:kernel_support[1] + 1,kernel_support[0]:kernel_support[1] + 1]

def Delta_Im(size):
    delta_im = np.zeros([size, size])
    delta_im[np.ceil(size / 2).astype(np.int32) - 1, np.ceil(size / 2).astype(np.int32) - 1] = 1
    return delta_im

def Gaussian_2D(sigma,size=None):
    if size is None:
        # I want the kernel to contain 99% of the filter's energy (in 1D), so I'm leaving 0.5% on each side:
        size = int(1+2*np.ceil(-1*norm.ppf(0.005,scale=sigma)))
    else:
        assert (size+1)/2==np.round((size+1)/2),'Size must be odd integer'
    gaussian_2D =  gaussian(size,sigma).reshape([1,size])*gaussian(size,sigma).reshape([size,1])
    return gaussian_2D/np.sum(gaussian_2D)

def Round_2_Int(num):
    return int(np.round(num))

def Center_Mass(kernel,ds_factor):
    assert kernel.shape[0]==kernel.shape[1],'Currently supporting only square kernels'
    kernel_size = kernel.shape[0]
    x_grid,y_grid = np.meshgrid(np.arange(kernel_size),np.arange(kernel_size))
    x_grid,y_grid = convolve2d(x_grid,kernel,mode='valid')+1,convolve2d(y_grid,kernel,mode='valid')+1
    x_pad,y_pad = 2*(kernel_size/2-x_grid),2*(kernel_size/2-y_grid)
    padding_diff = np.round(np.abs(y_pad))-np.round(np.abs(x_pad))
    pre_x_pad,post_x_pad = np.maximum(0,-x_pad),np.maximum(0,x_pad)
    pre_y_pad,post_y_pad = np.maximum(0,-y_pad),np.maximum(0,y_pad)

    def Wisely_Add_Padding_2_Axis(pre_pad,post_pad,padding_diff):
        # Decide how to split the extra padding needed (in case it's odd), considering the padding quantization error:
        centering_offset_to_right = np.round(post_pad)-post_pad-(np.round(pre_pad)-pre_pad)
        pre_pad,post_pad = Round_2_Int(pre_pad), Round_2_Int(post_pad)
        if centering_offset_to_right>0:
            post_pad += int(np.ceil(padding_diff/2))
            pre_pad += int(np.floor(padding_diff/2))
        else:
            pre_pad += int(np.ceil(padding_diff/2))
            post_pad += int(np.floor(padding_diff/2))
        return pre_pad,post_pad
    if padding_diff>0:#Pad horizontal axis (x):
        pre_y_pad,post_y_pad = Round_2_Int(pre_y_pad), Round_2_Int(post_y_pad)
        pre_x_pad,post_x_pad = Wisely_Add_Padding_2_Axis(pre_x_pad,post_x_pad,padding_diff)
    elif padding_diff<0:#Pad vetical axis (y):
        pre_x_pad,post_x_pad = Round_2_Int(pre_x_pad), Round_2_Int(post_x_pad)
        pre_y_pad,post_y_pad = Wisely_Add_Padding_2_Axis(pre_y_pad,post_y_pad,-padding_diff)
        # pre_y_pad,post_y_pad = pre_y_pad-padding_diff/2,post_y_pad-padding_diff/2
    kernel = np.pad(kernel,((Round_2_Int(pre_y_pad),Round_2_Int(post_y_pad)),(Round_2_Int(pre_x_pad),Round_2_Int(post_x_pad))),mode='constant')
    assert kernel.shape[0]==kernel.shape[1],'I caused the kernel to stop being a square...'
    margins_2_remove = np.argwhere(Return_Filter_Energy_Distribution(kernel)<0.99)[0][0]*np.ones([2]).astype(np.int32)
    pre_post_index = 0
    while np.mod(kernel.shape[0]-np.sum(margins_2_remove)-1+np.mod(ds_factor+1,2),ds_factor)!=0:
        margins_2_remove[pre_post_index] -= 1
        pre_post_index = np.mod(pre_post_index+1,2)
    kernel = kernel[margins_2_remove[0]:-margins_2_remove[1],margins_2_remove[0]:-margins_2_remove[1]]
    return kernel/np.sum(kernel)

def Return_Filter_Energy_Distribution(filter):
    sqrt_energy =  [np.sqrt(np.sum(filter**2))]+[np.sqrt(np.sum(filter[frame_num:-frame_num,frame_num:-frame_num]**2)) for frame_num in range(1,int(np.ceil(filter.shape[0]/2)))]
    return sqrt_energy/sqrt_energy[0]

def imresize_efficient(im, antialiasing_kernel, scale_factor=0.25, output_shape=None, pre_stride=[0,0], post_stride=[0,0], use_zero_padding=False):
    padding_size = np.floor(np.array(antialiasing_kernel.shape)/2).astype(np.int32)
    mask = np.rot90(antialiasing_kernel.astype(np.float32), 2)
    antialiasing_kernel = torch.from_numpy(np.expand_dims(np.stack([mask,mask,mask], axis=0),axis=0).transpose(1,0,2,3))
    output = []
    def filter2d(input,special_padding_size=None):
        if special_padding_size is not None:
            # input = 1*np.pad(input, pad_width=((special_padding_size[0], special_padding_size[0]), (special_padding_size[1], special_padding_size[1])), mode='edge')
            input = torchF.pad(input, (special_padding_size[0], special_padding_size[0], special_padding_size[1], special_padding_size[1]), "reflect")
        return torch.nn.functional.conv2d(input,
            antialiasing_kernel,
            padding=(antialiasing_kernel.shape[0] // 2,antialiasing_kernel.shape[1] // 2) if special_padding_size is None else 0,
            groups=3)
    output = filter2d(im, special_padding_size=padding_size)
    # print(im.shape, output.shape,1111111111111111)
    return output[:,:,pre_stride[0]::int(1 / scale_factor),pre_stride[1]::int(1 / scale_factor)]