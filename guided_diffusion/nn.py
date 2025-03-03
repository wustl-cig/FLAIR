"""
Various utilities for neural networks.
"""

from ast import mod
import math
from typing import Tuple
from einops import rearrange, repeat
from math import ceil, floor
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import more_itertools as mit
from flash_attn.flash_attn_interface import flash_attn_func


def get_block_from_index(blocks, d, h, w, D, H, W):
    # blocks: [(B, C, N_d, N_h, N_w), ...] list of block tensors
    # d, h, w: index of the block
    # D, H, W: number of blocks in each dimension
    return blocks[d * H * W + h * W + w]


## patchity a 3d volume
def patchify(voxels, block_size, stride, padding_mode="constant"):
    """
    Args:
        voxels: (B, C, T, H, W)
        block_size: [N_d, N_h, N_w]
        stride: [S_d, S_h, S_w]
        padding_mode: 'constant' or 'edge' or 'reflect'
    """
    B, C, D, H, W = voxels.shape
    pad_d = (ceil((D - block_size[0]) / stride[0]) * stride[0] + block_size[0] - D) / 2
    pad_h = (ceil((H - block_size[1]) / stride[1]) * stride[1] + block_size[1] - H) / 2
    pad_w = (ceil((W - block_size[2]) / stride[2]) * stride[2] + block_size[2] - W) / 2
    padded_voxels = th.from_numpy(
        np.pad(
            voxels.detach().cpu().numpy(),
            (
                (0, 0),
                (0, 0),
                (ceil(pad_d), floor(pad_d)),
                (ceil(pad_h), floor(pad_h)),
                (ceil(pad_w), floor(pad_w)),
            ),
            mode=padding_mode,
        )
    ).to(voxels.device, voxels.dtype)
    padded_D, padded_H, padded_W = padded_voxels.shape[2:]
    for d in range(0, padded_D - block_size[0] + 1, stride[0]):
        for h in range(0, padded_H - block_size[1] + 1, stride[1]):
            for w in range(0, padded_W - block_size[2] + 1, stride[2]):
                block = padded_voxels[
                    :,
                    :,
                    d : d + block_size[0],
                    h : h + block_size[1],
                    w : w + block_size[2],
                ]
                yield block


def mean_merge(block, overlap, d, h, w, D, H, W):
    if d != 0:
        block[:, :, : overlap[0], ...] = block[:, :, : overlap[0], ...] / 2
    if d != D - 1:
        block[:, :, -overlap[0] :, ...] = block[:, :, -overlap[0] :, ...] / 2
    if h != 0:
        block[:, :, :, : overlap[1], ...] = block[:, :, :, : overlap[1], ...] / 2
    if h != H - 1:
        block[:, :, :, -overlap[1] :, ...] = block[:, :, :, -overlap[1] :, ...] / 2
    if w != 0:
        block[:, :, :, :, : overlap[2]] = block[:, :, :, :, : overlap[2]] / 2
    if w != W - 1:
        block[:, :, :, :, -overlap[2] :] = block[:, :, :, :, -overlap[2] :] / 2
    return block


def max_merge(blocks, overlap, D, H, W):
    for d in range(D):
        for h in range(H):
            for w in range(W):
                if d != 0:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, : overlap[0], ...
                    ] = th.maximum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, : overlap[0], ...
                        ],
                        blocks[d - 1, h, w][:, :, -overlap[0] :, ...],
                    )
                if d != D - 1:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, -overlap[0] :, ...
                    ] = th.maximum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, -overlap[0] :, ...
                        ],
                        blocks[d + 1, h, w][:, :, : overlap[0], ...],
                    )
                if h != 0:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, : overlap[1], ...
                    ] = th.maximum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, : overlap[1], ...
                        ],
                        blocks[d, h - 1, w][:, :, -overlap[1] :, ...],
                    )
                if h != H - 1:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, :, -overlap[1] :, ...
                    ] = th.maximum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, :, -overlap[1] :, ...
                        ],
                        blocks[d, h + 1, w][:, :, :, : overlap[1], ...],
                    )
                if w != 0:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, :, :, : overlap[2]
                    ] = th.maximum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, :, :, : overlap[2]
                        ],
                        blocks[d, h, w - 1][:, :, :, :, -overlap[2] :],
                    )
                if w != W - 1:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, :, :, -overlap[2] :
                    ] = th.maximum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, :, :, -overlap[2] :
                        ],
                        blocks[d, h, w + 1][:, :, :, :, : overlap[2]],
                    )
    return blocks


def min_merge(blocks, overlap, D, H, W):
    for d in range(D):
        for h in range(H):
            for w in range(W):
                if d != 0:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, : overlap[0], ...
                    ] = th.minimum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, : overlap[0], ...
                        ],
                        blocks[d - 1, h, w][:, :, -overlap[0] :, ...],
                    )
                if d != D - 1:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, -overlap[0] :, ...
                    ] = th.minimum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, -overlap[0] :, ...
                        ],
                        blocks[d + 1, h, w][:, :, : overlap[0], ...],
                    )
                if h != 0:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, : overlap[1], ...
                    ] = th.minimum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, : overlap[1], ...
                        ],
                        blocks[d, h - 1, w][:, :, -overlap[1] :, ...],
                    )
                if h != H - 1:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, :, -overlap[1] :, ...
                    ] = th.minimum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, :, -overlap[1] :, ...
                        ],
                        blocks[d, h + 1, w][:, :, :, : overlap[1], ...],
                    )
                if w != 0:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, :, :, : overlap[2]
                    ] = th.minimum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, :, :, : overlap[2]
                        ],
                        blocks[d, h, w - 1][:, :, :, :, -overlap[2] :],
                    )
                if w != W - 1:
                    get_block_from_index(blocks, d, h, w, D, H, W)[
                        :, :, :, :, -overlap[2] :
                    ] = th.minimum(
                        get_block_from_index(blocks, d, h, w, D, H, W)[
                            :, :, :, :, -overlap[2] :
                        ],
                        blocks[d, h, w + 1][:, :, :, :, : overlap[2]],
                    )
    return blocks


def linear_merge(block, overlap, d, h, w, D, H, W):
    if d != 0:
        lin_space = (
            th.linspace(0, 1, overlap[0]).view(1, 1, overlap[0], 1, 1).to(block.device)
        )
        block[:, :, : overlap[0], ...] = lin_space * block[:, :, : overlap[0], ...]
    if d != D - 1:
        lin_space = (
            th.linspace(1, 0, overlap[0]).view(1, 1, overlap[0], 1, 1).to(block.device)
        )
        block[:, :, -overlap[0] :, ...] = lin_space * block[:, :, -overlap[0] :, ...]
    if h != 0:
        lin_space = (
            th.linspace(0, 1, overlap[1]).view(1, 1, 1, overlap[1], 1).to(block.device)
        )
        block[:, :, :, : overlap[1], ...] = (
            lin_space * block[:, :, :, : overlap[1], ...]
        )
    if h != H - 1:
        lin_space = (
            th.linspace(1, 0, overlap[1]).view(1, 1, 1, overlap[1], 1).to(block.device)
        )
        block[:, :, :, -overlap[1] :, ...] = (
            lin_space * block[:, :, :, -overlap[1] :, ...]
        )
    if w != 0:
        lin_space = (
            th.linspace(0, 1, overlap[2]).view(1, 1, 1, 1, overlap[2]).to(block.device)
        )
        block[:, :, :, :, : overlap[2]] = lin_space * block[:, :, :, :, : overlap[2]]
    if w != W - 1:
        lin_space = (
            th.linspace(1, 0, overlap[2]).view(1, 1, 1, 1, overlap[2]).to(block.device)
        )
        block[:, :, :, :, -overlap[2] :] = lin_space * block[:, :, :, :, -overlap[2] :]
    return block


def mid_merge(block, overlap, d, h, w, D, H, W):
    if d != 0:
        block[:, :, : ceil(overlap[0] / 2), ...] = 0
    if d != D - 1:
        block[:, :, -floor(overlap[0] / 2) :, ...] = 0
    if h != 0:
        block[:, :, :, : ceil(overlap[1] / 2), ...] = 0
    if h != H - 1:
        block[:, :, :, -floor(overlap[1] / 2) :, ...] = 0
    if w != 0:
        block[:, :, :, :, : ceil(overlap[2] / 2)] = 0
    if w != W - 1:
        block[:, :, :, :, -floor(overlap[2] / 2) :] = 0
    return block


## unpatchity a 3d volume, assume the blocks are in the same order as patchify
def unpatchify(blocks, dhw, stride, merge_mode="mean"):
    """
    Args:
        blocks: [(B, C, N_d, N_h, N_w), ...]
        dhw: (D, H, W)
        stride: [S_d, S_h, S_w]
        merge_mode: 'mean' or '
    """
    D, H, W = dhw
    B, C, *block_size = blocks[0].shape
    padded_D = ceil((D - block_size[0]) / stride[0]) * stride[0] + block_size[0]
    step_D = ceil((D - block_size[0]) / stride[0]) + 1
    pad_d = (padded_D - D) / 2
    padded_H = ceil((H - block_size[1]) / stride[1]) * stride[1] + block_size[1]
    step_H = ceil((H - block_size[1]) / stride[1]) + 1
    pad_h = (padded_H - H) / 2
    padded_W = ceil((W - block_size[2]) / stride[2]) * stride[2] + block_size[2]
    step_W = ceil((W - block_size[2]) / stride[2]) + 1
    pad_w = (padded_W - W) / 2
    padded_voxels = th.zeros(
        (B, C, padded_D, padded_H, padded_W),
        device=blocks[0].device,
        dtype=blocks[0].dtype,
    )
    # block_array = np.array(blocks, dtype=object).reshape(
    #     step_D, step_H, step_W
    # )  # 3d array of block tensors
    if merge_mode == "max":
        blocks = max_merge(blocks, [b - s for b, s in zip(block_size, stride)])
    elif merge_mode == "min":
        blocks = min_merge(blocks, [b - s for b, s in zip(block_size, stride)])
    for d in range(step_D):
        for h in range(step_H):
            for w in range(step_W):
                if merge_mode == "mean" or merge_mode == "max" or merge_mode == "min":
                    block = mean_merge(
                        get_block_from_index(blocks, d, h, w, step_D, step_H, step_W),
                        [b - s for b, s in zip(block_size, stride)],
                        d,
                        h,
                        w,
                        step_D,
                        step_H,
                        step_W,
                    )
                elif merge_mode == "linear":
                    block = linear_merge(
                        get_block_from_index(blocks, d, h, w, step_D, step_H, step_W),
                        [b - s for b, s in zip(block_size, stride)],
                        d,
                        h,
                        w,
                        step_D,
                        step_H,
                        step_W,
                    )
                elif merge_mode == "mid":
                    block = mid_merge(
                        get_block_from_index(blocks, d, h, w, step_D, step_H, step_W),
                        [b - s for b, s in zip(block_size, stride)],
                        d,
                        h,
                        w,
                        step_D,
                        step_H,
                        step_W,
                    )
                padded_voxels[
                    :,
                    :,
                    d * stride[0] : d * stride[0] + block_size[0],
                    h * stride[1] : h * stride[1] + block_size[1],
                    w * stride[2] : w * stride[2] + block_size[2],
                ] += block
    return padded_voxels[
        :,
        :,
        ceil(pad_d) : padded_D - floor(pad_d),
        ceil(pad_h) : padded_H - floor(pad_h),
        ceil(pad_w) : padded_W - floor(pad_w),
    ]


class PlaceHolder(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.wrapped_module = module

    def forward(self, x, *args, **kwargs):
        return self.wrapped_module(x, *args, **kwargs)


class LazyReshaper2D(PlaceHolder):
    def forward(self, x, *args, **kwargs):
        B, N, C, H, W = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        x = self.wrapped_module(x, *args, **kwargs)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=B)
        return x


class LazyReshaper3D(PlaceHolder):
    def forward(self, x, *args, **kwargs):
        output = rearrange(
            self.wrapped_module(
                rearrange(x, "b t c h w -> b c t h w"), *args, **kwargs
            ),
            "b c t h w -> b t c h w",
        )
        return output


def flash_attn_wrapper(q, k, v, dropout):
    dtype = q.dtype
    attn = []
    for q_sliced, k_sliced, v_sliced in zip(
        q.contiguous().split(65535, dim=0),
        k.contiguous().split(65535, dim=0),
        v.contiguous().split(65535, dim=0),
    ):
        attn_sliced = flash_attn_func(
            q_sliced.to(th.float16),
            k_sliced.to(th.float16),
            v_sliced.to(th.float16),
            dropout_p=dropout,
        )  # (B*N*W, num_heads, C)
        attn.append(attn_sliced)
    attn = th.cat(attn, dim=0)
    return attn.to(dtype)


class FalshAttn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return flash_attn_wrapper(q, k, v, 0.0)


class SliceProcessor(nn.Module):
    def __init__(self, module, offload_to_cpu=False):
        super().__init__()
        self.module = module
        self.offload_to_cpu = offload_to_cpu
        self.orig_device = self.module.parameters().__next__().device
        if offload_to_cpu:
            self.module = self.module.to("cpu")


class SliceProcessor1D(SliceProcessor):
    def __init__(self, conv, win_size, offload_to_cpu=False):
        super().__init__(conv, offload_to_cpu)
        self.win_size = win_size

    def forward(self, x):
        if self.offload_to_cpu and x.device != "cpu":
            x = x.to("cpu")
        B, C, T = x.shape
        if self.offload_to_cpu:
            self.module = self.module.to(self.orig_device)
        ret = []
        for x_sliced in x.split(self.win_size, dim=2):
            x_sliced = x_sliced.to(self.orig_device)
            x_sliced = self.module(x_sliced)
            if self.offload_to_cpu:
                x_sliced = x_sliced.to("cpu")
            ret.append(x_sliced)
        if self.offload_to_cpu:
            self.module = self.module.to("cpu")
        ret = th.cat(ret, dim=2).to(self.orig_device)
        return ret


class SliceProcessorLinear(SliceProcessor):
    def __init__(self, module, win_size, offload_to_cpu=False):
        super().__init__(module, offload_to_cpu)
        self.win_size = win_size

    def forward(self, x):
        if self.offload_to_cpu and x.device != "cpu":
            x = x.to("cpu")
        *Bs, C = x.shape
        x = x.reshape(-1, C)
        if self.offload_to_cpu:
            self.module = self.module.to(self.orig_device)
        ret = []
        for x_sliced in x.split(self.win_size, dim=1):
            x_sliced = x_sliced.to(self.orig_device)
            x_sliced = self.module(x_sliced)
            if self.offload_to_cpu:
                x_sliced = x_sliced.to("cpu")
            ret.append(x_sliced)
        if self.offload_to_cpu:
            self.module = self.module.to("cpu")
        ret = th.cat(ret, dim=1)
        ret = ret.reshape(*Bs, -1).to(self.orig_device)
        return ret


class SliceProcessorFlashAttn(SliceProcessor):
    def __init__(self, module, win_size, offload_to_cpu=False):
        super().__init__(module, offload_to_cpu)
        self.win_size = win_size

    def forward(self, q, k, v):
        if self.offload_to_cpu and q.device != "cpu":
            q, k, v = q.to("cpu"), k.to("cpu"), v.to("cpu")
        if self.offload_to_cpu:
            self.module = self.module.to(self.orig_device)
        ret = []
        for q_sliced, k_sliced, v_sliced in zip(
            q.split(self.win_size, dim=0),
            k.split(self.win_size, dim=0),
            v.split(self.win_size, dim=0),
        ):
            q_sliced, k_sliced, v_sliced = (
                q_sliced.to(self.orig_device),
                k_sliced.to(self.orig_device),
                v_sliced.to(self.orig_device),
            )
            attn_sliced = self.module(q_sliced, k_sliced, v_sliced)
            if self.offload_to_cpu:
                attn_sliced = attn_sliced.to("cpu")
            ret.append(attn_sliced)
        if self.offload_to_cpu:
            self.module = self.module.to("cpu")
        ret = th.cat(ret, dim=0).to(self.orig_device)
        return ret


class SliceProcessor2D(SliceProcessor):
    def __init__(self, conv, win_size, offload_to_cpu=False):
        super().__init__(conv, offload_to_cpu)
        self.win_size = win_size

    def forward(self, x):
        if self.offload_to_cpu and x.device != "cpu":
            x = x.to("cpu")
        B, C, H, W = x.shape
        if self.offload_to_cpu:
            self.module = self.module.to(self.orig_device)
        ret = []
        for x_sliced in x.split(self.win_size, dim=0):
            x_sliced = x_sliced.to(self.orig_device)
            x_sliced = self.module(x_sliced)
            if self.offload_to_cpu:
                x_sliced = x_sliced.to("cpu")
            ret.append(x_sliced)
        if self.offload_to_cpu:
            self.module = self.module.to("cpu")
        ret = th.cat(ret, dim=0).to(self.orig_device)
        return ret


class SliceProcessor3D(SliceProcessor):
    def __init__(
        self,
        conv: th.nn.Conv3d,
        win_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        offload_to_cpu=False,
    ):
        self.win_size = win_size
        self.stride = stride
        conv_new = nn.Conv3d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=0,
            bias=not conv.bias == None,
            dilation=conv.dilation,
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        conv_new.weight.data = conv.weight.data
        conv_new.bias.data = conv.bias.data
        self.padding = [s // 2 for s in conv_new.kernel_size]
        self.padded_win_size = [w + 2 * p for w, p in zip(self.win_size, self.padding)]
        super().__init__(conv_new, offload_to_cpu)

    def forward(self, x):
        if self.offload_to_cpu and x.device != "cpu":
            x = x.to("cpu")
        B, C, T, H, W = x.shape

        if self.offload_to_cpu:
            self.module = self.module.to(self.orig_device)
        blocks = []
        for block in patchify(x, self.padded_win_size, self.stride):
            block = block.to(self.orig_device)
            block = self.module(block)
            if self.offload_to_cpu:
                block = block.to("cpu")
            blocks.append(block)
        ret = unpatchify(blocks, (T, H, W), self.stride)
        if self.offload_to_cpu:
            self.module = self.module.to("cpu")
        return ret.to(self.orig_device)


class SliceProcessorGroupNorm(SliceProcessor):
    def __init__(self, norm, win_size, offload_to_cpu=False):
        super().__init__(norm, offload_to_cpu)
        self.win_size = win_size

    def forward(self, x):
        if self.offload_to_cpu and x.device != "cpu":
            x = x.to("cpu")
        if self.offload_to_cpu:
            self.module = self.module.to(self.orig_device)
        ret = []
        for x_sliced in x.split(self.win_size, dim=2):
            x_sliced = x_sliced.to(self.orig_device)
            x_sliced = self.module(x_sliced)
            if self.offload_to_cpu:
                x_sliced = x_sliced.to("cpu")
            ret.append(x_sliced)
        if self.offload_to_cpu:
            self.module = self.module.to("cpu")
        ret = th.cat(ret, dim=2)
        return ret.to(self.orig_device)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, dims, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, dims, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# Pyth 1.7 has SiLU, but we support Pyth 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ShiftWindowGroupNorm32(nn.Module):
    def __init__(
        self,
        win_size,
        num_groups,
        num_channels,
        eps=1e-05,
        padding_mode="replicate",  # "zeros", "none"
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert win_size % 2 == 1, "win_size must be odd"
        self.win_size = win_size
        self.padding_mode = padding_mode
        self.padding = (win_size - 1) // 2
        assert (
            num_channels % num_groups == 0
        ), "num_channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(th.empty(num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(th.empty(num_channels, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def replicate_pad(self, x):
        return th.cat(
            [
                x[:, :1].repeat(1, self.padding, 1, 1, 1),
                x,
                x[:, -1:].repeat(1, self.padding, 1, 1, 1),
            ],
            dim=1,
        )

    def zero_pad(self, x):
        return th.cat(
            [
                th.zeros_like(x[:, : self.padding]),
                x,
                th.zeros_like(x[:, : self.padding]),
            ],
            dim=1,
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        dtype = x.dtype
        x = x.to(th.float32)
        T = x.shape[1]
        if T == 1:
            x_sliced = rearrange(
                x, "B T (G C) H W -> B T G C (H W) ()", G=self.num_groups
            )
        else:
            x_sliced = rearrange(
                (
                    self.replicate_pad(x)
                    if self.padding_mode == "replicate"
                    else (self.zero_pad(x) if self.padding_mode == "zeros" else x)
                ).unfold(1, self.win_size, 1),
                "B T (G C) H W WIN -> B T G C (H W) WIN",
                G=self.num_groups,
            )
        if self.padding_mode == "none":
            x = x[:, self.padding : -self.padding]
        x_sliced_mean = repeat(
            x_sliced.mean([3, 4, 5]),
            "B T G -> B T (G C) () ()",
            C=self.num_channels // self.num_groups,
        )
        x_sliced_var = repeat(
            x_sliced.var(dim=[3, 4, 5], unbiased=False),
            "B T G -> B T (G C) () ()",
            C=self.num_channels // self.num_groups,
        )
        x = (x - x_sliced_mean) / th.sqrt(x_sliced_var + self.eps) * rearrange(
            self.weight, "C -> () () C () ()"
        ) + rearrange(self.bias, "C -> () () C () ()")
        return x.to(dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return LazyReshaper2D(nn.Conv2d(*args, **kwargs))
    elif dims == 3:
        return LazyReshaper3D(nn.Conv3d(*args, **kwargs))
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def upsample(dims, *args, **kwargs):
    """
    Create a 2D or 3D upsampling module.
    """
    if dims == 2:
        return LazyReshaper2D(Upsample(dims, *args, **kwargs))
    else:
        return LazyReshaper3D(Upsample(dims, *args, **kwargs))


def downsample(dims, *args, **kwargs):
    """
    Create a 2D or 3D downsampling module.
    """
    if dims == 2:
        return LazyReshaper2D(Downsample(dims, *args, **kwargs))
    else:
        return LazyReshaper3D(Downsample(dims, *args, **kwargs))


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, *args, **kwargs):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return LazyReshaper3D(GroupNorm32(32, channels))


def shift_window_normalization(channels, win_size):
    """
    Make a shift window normalization layer.

    :param channels: number of input channels.
    :param win_size: window size.
    :return: an nn.Module for normalization.
    """
    return PlaceHolder(ShiftWindowGroupNorm32(win_size, 32, channels))


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple([p for p in params if p.requires_grad])
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


if __name__ == "__main__":
    with th.no_grad():
        conv = nn.Conv3d(1, 1, 3, padding=1)
        wrapper = SliceProcessor3D(conv, (16, 16, 16), (13, 13, 13))
        x = th.randn(1, 1, 64, 64, 64)
        y = wrapper(x)
