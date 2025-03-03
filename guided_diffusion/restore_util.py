from typing import Any, Callable
import torch
import numpy as np
from einops import rearrange, repeat
import math
from guided_diffusion.superslomo import SuperSloMo
import more_itertools as mit
from torch.nn import functional as F


class A_functions:
    """
    A class replacing the SVD of a matrix A, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()

        factors = 1.0 / singulars
        factors[singulars == 0] = 0.0

        #         temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        temp[:, : singulars.shape[0]] = temp[:, : singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))

    def A_pinv_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of A with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars * singulars + eta)
        #         print(temp.size(), factors.size(), singulars.size())
        temp[:, : singulars.shape[0]] = temp[:, : singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        raise NotImplementedError()

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        raise NotImplementedError()


class SRConv(A_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim, dim)).reshape(
            v.shape[0], self.channels, M.shape[0], dim
        )

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim, dim), M).reshape(
            v.shape[0], self.channels, dim, M.shape[1]
        )

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.y_dim = small_dim
        # build 1D conv matrix
        A_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                # reflective padding
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= img_dim:
                    j_effective = (img_dim - 1) - (j_effective - img_dim)
                # matrix building
                A_small[i // stride, j_effective] += kernel[
                    j - i + kernel.shape[0] // 2
                ]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(
            A_small, some=False
        )
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small.reshape(small_dim, 1),
            self.singulars_small.reshape(1, small_dim),
        ).reshape(small_dim**2)
        # permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = (
            torch.Tensor(
                [
                    self.img_dim * i + j
                    for i in range(self.y_dim)
                    for j in range(self.y_dim)
                ]
                + [
                    self.img_dim * i + j
                    for i in range(self.y_dim)
                    for j in range(self.y_dim, self.img_dim)
                ]
            )
            .to(device)
            .long()
        )

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )[:, : self._perm.shape[0], :]
        temp[:, self._perm.shape[0] :, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )[:, self._perm.shape[0] :, :]
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(
            vec.shape[0], -1
        )
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries
        temp[:, :, : self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.y_dim**2, self.channels, device=vec.device
        )
        temp[:, : self.y_dim**2, :] = vec.clone().reshape(
            vec.shape[0], self.y_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.y_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.y_dim).reshape(
            vec.shape[0], -1
        )
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.y_dim)
        temp = self.img_by_mat(temp, self.U_small, self.y_dim).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class SuperResolution(A_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim[0] % ratio == 0 and img_dim[1] % ratio == 0
        self.img_dim = img_dim
        self.img_num_pixels = img_dim[0] * img_dim[1]
        self.channels = channels
        self.y_dim = tuple([d // ratio for d in img_dim])
        self.y_num_pixels = self.y_dim[0] * self.y_dim[1]
        self.ratio = ratio
        A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(
            vec.shape[0],
            self.channels,
            self.y_num_pixels,
            self.ratio**2,
            device=vec.device,
        )
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_num_pixels].view(
            vec.shape[0], self.channels, -1
        )
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[
                :, (self.channels * self.y_num_pixels + idx) :: self.ratio**2 - 1
            ].view(vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(
            self.V_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # repatch the patches into an image
        patches_orig = patches.reshape(
            vec.shape[0],
            self.channels,
            self.y_dim[0],
            self.y_dim[1],
            self.ratio,
            self.ratio,
        )
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_num_pixels)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim[0], self.img_dim[1]
        )
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches = patches.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        # multiply each by the small V transposed
        patches = torch.matmul(
            self.Vt_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(
            vec.shape[0], self.channels * self.img_num_pixels, device=vec.device
        )
        recon[:, : self.channels * self.y_num_pixels] = patches[:, :, :, 0].view(
            vec.shape[0], self.channels * self.y_num_pixels
        )
        for idx in range(self.ratio**2 - 1):
            recon[
                :, (self.channels * self.y_num_pixels + idx) :: self.ratio**2 - 1
            ] = patches[:, :, :, idx + 1].view(
                vec.shape[0], self.channels * self.y_num_pixels
            )
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_num_pixels)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        singulars = self.singulars_small

        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim[0], self.img_dim[1]
        )
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches = patches.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )

        patches = torch.matmul(
            self.Vt_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)

        lambda_t = torch.ones(self.ratio**2, device=vec.device)

        temp = torch.zeros(self.ratio**2, device=vec.device)
        temp[: singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                singulars * sigma_t * (1 - eta**2) ** 0.5 / a / sigma_y
            )

        lambda_t = lambda_t.reshape(1, 1, 1, -1)
        #         print("lambda_t:", lambda_t)
        #         print("V:", self.V_small)
        #         print(lambda_t.size(), self.V_small.size())
        #         print("Sigma_t:", torch.matmul(torch.matmul(self.V_small, torch.diag(lambda_t.reshape(-1))), self.Vt_small))
        patches = patches * lambda_t

        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1))

        patches = patches.reshape(
            vec.shape[0],
            self.channels,
            self.y_dim[0],
            self.y_dim[1],
            self.ratio,
            self.ratio,
        )
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(vec.shape[0], self.channels * self.img_num_pixels)

        return patches

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        singulars = self.singulars_small

        patches_vec = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim[0], self.img_dim[1]
        )
        patches_vec = patches_vec.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches_vec = patches_vec.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )

        patches_eps = epsilon.clone().reshape(
            vec.shape[0], self.channels, self.img_dim[0], self.img_dim[1]
        )
        patches_eps = patches_eps.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches_eps = patches_eps.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )

        d1_t = torch.ones(self.ratio**2, device=vec.device) * sigma_t * eta
        d2_t = (
            torch.ones(self.ratio**2, device=vec.device)
            * sigma_t
            * (1 - eta**2) ** 0.5
        )

        temp = torch.zeros(self.ratio**2, device=vec.device)
        temp[: singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index
                * (sigma_t**2 - a**2 * sigma_y**2 * inverse_singulars**2)
            )
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = (
                d2_t * (-change_index + 1.0)
                + change_index * sigma_t * (1 - eta**2) ** 0.5
            )

        d1_t = d1_t.reshape(1, 1, 1, -1)
        d2_t = d2_t.reshape(1, 1, 1, -1)
        patches_vec = patches_vec * d1_t
        patches_eps = patches_eps * d2_t

        patches_vec = torch.matmul(
            self.V_small, patches_vec.reshape(-1, self.ratio**2, 1)
        )

        patches_vec = patches_vec.reshape(
            vec.shape[0],
            self.channels,
            self.y_dim[0],
            self.y_dim[1],
            self.ratio,
            self.ratio,
        )
        patches_vec = patches_vec.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_vec = patches_vec.reshape(
            vec.shape[0], self.channels * self.img_num_pixels
        )

        patches_eps = torch.matmul(
            self.V_small, patches_eps.reshape(-1, self.ratio**2, 1)
        )

        patches_eps = patches_eps.reshape(
            vec.shape[0],
            self.channels,
            self.y_dim[0],
            self.y_dim[1],
            self.ratio,
            self.ratio,
        )
        patches_eps = patches_eps.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_eps = patches_eps.reshape(
            vec.shape[0], self.channels * self.img_num_pixels
        )

        return patches_vec + patches_eps
