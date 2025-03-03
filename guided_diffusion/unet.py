from abc import abstractmethod
from argparse import Namespace
from functools import partial

import math
import random
from re import L

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    GroupNorm32,
    ShiftWindowGroupNorm32,
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    shift_window_normalization,
    upsample,
    downsample,
    FalshAttn,
)
from .gaussian_diffusion import _extract_into_tensor
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv,
    SPyNet,
)
from mmedit.models.common import PixelShufflePack, flow_warp
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d
import more_itertools as mit


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, SecondOrderDeformableAlignment)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, SecondOrderDeformableAlignment)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


class TemporalWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.wrapped_module = module
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, enable_cross_frames=True, **kwargs):
        if not enable_cross_frames:
            return x
        output = self.wrapped_module(x, *args, **kwargs)
        output = (1 - torch.sigmoid(self.weight.to(x.dtype))) * x + torch.sigmoid(
            self.weight.to(x.dtype)
        ) * output
        return output


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, *args, **kwargs):
        for layer in self:
            if isinstance(layer, TimestepBlock) or (
                isinstance(layer, TemporalWrapper)
                and isinstance(layer.wrapped_module, TimestepBlock)
            ):
                x = layer(x, emb, *args, **kwargs)
            else:
                if isinstance(layer, TemporalWrapper):
                    x = layer(x, *args, **kwargs)
                else:
                    x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        kernel_size=3,
        padding=1,
        stride=1,
        padding_mode="zeros",
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        norm_type="group_norm",
        win_size=5,
    ):
        super().__init__()
        self.dims = dims
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        if norm_type == "group_norm":
            norm_fn = normalization
        elif norm_type == "shift_window_norm":
            norm_fn = shift_window_normalization
        elif norm_type == "none":
            norm_fn = lambda *args, **kwargs: nn.Identity()
        self.in_layers = nn.Sequential(
            norm_fn(channels, win_size=win_size),
            nn.SiLU(),
            conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                padding_mode=padding_mode,
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = upsample(dims, channels, False)
            self.x_upd = upsample(dims, channels, False)
        elif down:
            self.h_upd = downsample(dims, channels, False)
            self.x_upd = downsample(dims, channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            norm_fn(self.out_channels, win_size=win_size),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    padding_mode=padding_mode,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, *args, **kwargs):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = rearrange(
            self.emb_layers(emb).type(h.dtype), "(b n) c -> b n c () ()", n=h.shape[1]
        )
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=2)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        norm_type="group_norm",
        win_size=5,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        if norm_type == "group_norm":
            norm_fn = normalization
        elif norm_type == "shift_window_norm":
            norm_fn = shift_window_normalization
        elif norm_type == "none":
            norm_fn = lambda *args, **kwargs: nn.Identity()
        self.norm = norm_fn(channels, win_size=win_size)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attn = FalshAttn()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, *args, **kwargs):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        x = self.norm(x)
        n = x.shape[1]
        x = x.view(-1, *x.shape[2:])
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        q, k, v = rearrange(
            self.qkv(x), "b (qkv h c) n -> qkv b n h c", qkv=3, h=self.num_heads
        ).unbind(0)
        h = self.attn(q, k, v)
        h = rearrange(h, "b n h c -> b (h c) n")
        h = self.proj_out(h)
        return (x + h).reshape(-1, n, c, *spatial)


class BasicVSRPP(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(
        self,
        mid_channels=64,
        max_residue_magnitude=10,
        use_checkpoint=False,
        shared_spynet=None,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.use_checkpoint = use_checkpoint
        # optical flow
        self.spynet = shared_spynet

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        # self.norm = nn.ModuleDict()
        modules = ["backward_1", "forward_1"]
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=max_residue_magnitude,
                )
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, 1
            )
            # self.norm[module] = GroupNorm32(32, mid_channels)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            3 * mid_channels, mid_channels, 1
        )
        self.conv_last = zero_module(nn.Conv2d(mid_channels, mid_channels, 1, 1))

    @th.no_grad()
    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        lqs = ((lqs + 1) / 2).clamp(0, 1)
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def propagate(self, feats, flows: th.Tensor, module_name, weight):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(
            n, self.mid_channels, h, w, dtype=feats["spatial"][0].dtype
        )
        for i, idx in enumerate(frame_idx):
            feat_current: th.Tensor = feats["spatial"][mapping_idx[idx]]
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :].to(feat_current.dtype)

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :].to(feat_current.dtype)

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = checkpoint(
                    self.deform_align[module_name],
                    (feat_prop, cond, flow_n1, flow_n2),
                    self.deform_align[module_name].parameters(),
                    self.use_checkpoint,
                )

            # concatenate and residual blocks
            feat = (
                [feat_current]
                + [feats[k][idx] for k in feats if k not in ["spatial", module_name]]
                + [feat_prop]
            )

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + checkpoint(
                self.backbone[module_name],
                (feat,),
                self.backbone[module_name].parameters(),
                self.use_checkpoint,
            )
            # feat_prop = self.norm[module_name](feat_prop)
            feats[module_name].append(feat_prop)
            feat_prop *= weight[:, mapping_idx[idx], ...]

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, hidden, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """
        num_outputs = len(feats["spatial"])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]
        recons = []
        for i in range(0, hidden.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            hr = checkpoint(
                self.reconstruction,
                (hr,),
                self.reconstruction.parameters(),
                self.use_checkpoint,
            )
            recons.append(hr)
        recons = torch.stack(recons, dim=1)
        recons = rearrange(
            self.conv_last(rearrange(recons, "n t c h w -> (n t) c h w")),
            "(n t) c h w -> n t c h w",
            n=hidden.shape[0],
        )
        outputs = recons + hidden
        return outputs

    def forward(self, hidden, lqs, *args, weight=None, **kwargs):
        # def forward(self, lqs):         #TODO
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        if lqs.shape[-2] != hidden.shape[-2] or lqs.shape[-1] != hidden.shape[-1]:
            lqs = rearrange(
                F.interpolate(
                    rearrange(lqs, "n t c h w -> (n t) c h w"),
                    size=(hidden.shape[-2], hidden.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ),
                "(n t) c h w -> n t c h w",
                n=lqs.shape[0],
            )
        n, t, c, h, w = lqs.size()
        feats = {}
        # compute spatial features
        feats["spatial"] = [hidden[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs.size(3) >= 64 and lqs.size(4) >= 64, (
            "The height and width of low-res inputs must be at least 64, "
            f"but got {h} and {w}."
        )
        flows_forward, flows_backward = self.compute_flow(lqs)
        if weight is None:
            weight = torch.ones(n, t, 1, 1, 1, device=hidden.device)
        elif isinstance(weight, float):
            weight = torch.ones(n, t, 1, 1, 1, device=hidden.device) * weight
        else:
            if weight.shape[-2] != h or weight.shape[-1] != w:
                weight = rearrange(
                    F.interpolate(
                        rearrange(weight, "b t c h w -> (b t) c h w"),
                        size=(h, w),
                        mode="nearest",
                    ),
                    "(b t) c h w -> b t c h w",
                    t=t,
                )
        # feature propgation
        for direction in ["backward_1", "forward_1"]:
            module = direction

            feats[module] = []

            if "backward" in direction:
                flows = flows_backward
            elif flows_forward is not None:
                flows = flows_forward
            else:
                flows = flows_backward.flip(1)

            feats = self.propagate(feats, flows, module, weight)

        return self.upsample(hidden, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


class TemporalAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_frames,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        norm_type="group_norm",
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        assert num_frames % 2 == 1, "num_frames must be odd"
        self.num_frames = num_frames
        self.num_head_channels = num_head_channels
        self.use_checkpoint = use_checkpoint
        self.qk_scale = (channels // num_heads) ** -0.5
        self.q_linear = linear(channels, channels)
        self.k_linear = linear(channels, channels)
        self.v_linear = linear(channels, channels)
        self.attn = FalshAttn()
        self.proj = zero_module(conv_nd(2, channels, channels, 1))

        self.norm = (
            shift_window_normalization(channels, num_frames)
            if norm_type == "shift_window_norm"
            else (
                normalization(channels) if norm_type == "group_norm" else nn.Identity()
            )
        )

        t = timestep_embedding(
            (torch.arange(self.num_frames, dtype=torch.long) - (self.num_frames // 2)),
            channels,
        )
        self.t_mid = t[self.num_frames // 2 : self.num_frames // 2 + 1, ...]
        self.t_rest = t[torch.arange(self.num_frames) != self.num_frames // 2, ...]

    def forward(self, x, *args, **kwargs):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, h):
        B, T, C, H, W = h.shape
        x = self.norm(h)
        x_sliced = rearrange(
            th.cat(
                [
                    x[:, :1].repeat(1, self.num_frames // 2, 1, 1, 1),
                    x,
                    x[:, -1:].repeat(1, self.num_frames // 2, 1, 1, 1),
                ],
                dim=1,
            ).unfold(1, self.num_frames, 1),
            "b t c h w f -> (b t h w) f c",
        )
        t_q = rearrange(self.t_mid, "n d -> () n d").to(dtype=x.dtype, device=x.device)
        t_kv = rearrange(self.t_rest, "n d -> () n d").to(
            dtype=x.dtype, device=x.device
        )
        q = rearrange(
            self.q_linear(x_sliced[:, [self.num_frames // 2], ...] + t_q),
            # "b n (h c) -> (b h) n c",
            "b n (h c) -> b n h c",
            h=self.num_heads,
        )  # (B*T*H*W, 1, Head, C)
        kv_input = x_sliced[
            :,
            torch.arange(self.num_frames, device=x.device) != self.num_frames // 2,
            ...,
        ]
        k = rearrange(
            self.k_linear(kv_input + t_kv),
            # "b n (h c) -> (b h) c n",
            "b n (h c) -> b n h c",
            h=self.num_heads,
        )  # (B*T*H*W, num_frames, Head, C)
        v = rearrange(
            self.v_linear(kv_input),
            # "b n (h c) -> (b h) n c",
            "b n (h c) -> b n h c",
            h=self.num_heads,
        )  # (B*T*H*W, num_frames, Head, C)
        attn = self.attn(q, k, v)  # (B*T*H*W, 1, Hea
        attn = rearrange(
            attn, "(b t h w) () head c -> b t (head c) h w", b=B, t=T, h=H, w=W
        )
        x = self.proj(attn)
        return x + h


class CrossFrameUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        cross_frame_module=False,
        res3d_kernel_size=(3, 1, 1),
        temp_attn_num_frames=5,
        norm_type="group_norm",
        temporal_norm_type=None,
        spatial_attn=True,
        **kwargs,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.cross_frame_module = cross_frame_module
        self.use_scale_shift_norm = use_scale_shift_norm
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_type=norm_type,
                        win_size=temp_attn_num_frames,
                    )
                ]
                ch = int(mult * model_channels)
                if cross_frame_module:
                    layers.append(
                        TemporalWrapper(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                dims=3,
                                use_checkpoint=self.use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                kernel_size=res3d_kernel_size,
                                padding=(
                                    res3d_kernel_size[0] // 2,
                                    res3d_kernel_size[1] // 2,
                                    res3d_kernel_size[2] // 2,
                                ),
                                win_size=temp_attn_num_frames,
                                norm_type=temporal_norm_type
                                if temporal_norm_type is not None
                                else norm_type,
                            )
                        )
                    )
                else:
                    layers.append(nn.Identity())
                if ds in attention_resolutions:
                    if spatial_attn:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=self.use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                norm_type=norm_type,
                                win_size=temp_attn_num_frames,
                            )
                        )
                    else:
                        layers.append(nn.Identity())
                    if cross_frame_module:
                        layers.append(
                            TemporalWrapper(
                                TemporalAttention(
                                    ch,
                                    num_frames=temp_attn_num_frames,
                                    use_checkpoint=self.use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                                    norm_type=temporal_norm_type
                                    if temporal_norm_type is not None
                                    else norm_type,
                                )
                            )
                        )
                    else:
                        layers.append(nn.Identity())
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm_type=norm_type,
                            win_size=temp_attn_num_frames,
                        )
                        if resblock_updown
                        else downsample(dims, ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_type=norm_type,
                win_size=temp_attn_num_frames,
            ),
            TemporalWrapper(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=3,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    kernel_size=res3d_kernel_size,
                    padding=(
                        res3d_kernel_size[0] // 2,
                        res3d_kernel_size[1] // 2,
                        res3d_kernel_size[2] // 2,
                    ),
                    win_size=temp_attn_num_frames,
                    norm_type=temporal_norm_type
                    if temporal_norm_type is not None
                    else norm_type,
                )
            )
            if cross_frame_module
            else nn.Identity(),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                norm_type=norm_type,
                win_size=temp_attn_num_frames,
            )
            if spatial_attn
            else nn.Identity(),
            TemporalWrapper(
                TemporalAttention(
                    ch,
                    num_frames=temp_attn_num_frames,
                    use_checkpoint=self.use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    norm_type=temporal_norm_type
                    if temporal_norm_type is not None
                    else norm_type,
                )
            )
            if cross_frame_module
            else nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_type=norm_type,
                win_size=temp_attn_num_frames,
            ),
            TemporalWrapper(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=3,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    kernel_size=res3d_kernel_size,
                    padding=(
                        res3d_kernel_size[0] // 2,
                        res3d_kernel_size[1] // 2,
                        res3d_kernel_size[2] // 2,
                    ),
                    win_size=temp_attn_num_frames,
                    norm_type=temporal_norm_type
                    if temporal_norm_type is not None
                    else norm_type,
                )
            )
            if cross_frame_module
            else nn.Identity(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_type=norm_type,
                        win_size=temp_attn_num_frames,
                    )
                ]
                ch = int(model_channels * mult)
                if cross_frame_module:
                    layers.append(
                        TemporalWrapper(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                dims=3,
                                use_checkpoint=self.use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                kernel_size=res3d_kernel_size,
                                padding=(
                                    res3d_kernel_size[0] // 2,
                                    res3d_kernel_size[1] // 2,
                                    res3d_kernel_size[2] // 2,
                                ),
                                win_size=temp_attn_num_frames,
                                norm_type=temporal_norm_type
                                if temporal_norm_type is not None
                                else norm_type,
                            )
                        )
                    )
                else:
                    layers.append(nn.Identity())
                if ds in attention_resolutions:
                    if spatial_attn:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=self.use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                norm_type=norm_type,
                                win_size=temp_attn_num_frames,
                            )
                        )
                    else:
                        layers.append(nn.Identity())
                    if cross_frame_module:
                        layers.append(
                            TemporalWrapper(
                                TemporalAttention(
                                    ch,
                                    num_frames=temp_attn_num_frames,
                                    use_checkpoint=self.use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=num_head_channels,
                                    norm_type=temporal_norm_type
                                    if temporal_norm_type is not None
                                    else norm_type,
                                )
                            )
                        )
                    else:
                        layers.append(nn.Identity())
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm_type=norm_type,
                            win_size=temp_attn_num_frames,
                        )
                        if resblock_updown
                        else upsample(dims, ch, conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch)
            if norm_type == "group_norm"
            else (
                shift_window_normalization(ch, temp_attn_num_frames)
                if norm_type == "shift_window_norm"
                else nn.Identity()
            ),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        if cross_frame_module:
            self.dav = BasicVSRPP(
                3,
                64,
                7,
                10,
                time_embed_dim,
                None,
                win_size=temp_attn_num_frames,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_type=temporal_norm_type
                if temporal_norm_type is not None
                else norm_type,
            )

    def freeze_spatial(self):
        """
        Freeze the model.
        """
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, (TemporalWrapper, BasicVSRPP)):
                for p in m.parameters():
                    p.requires_grad = True
        if hasattr(self, "dav"):
            for p in self.dav.spynet.parameters():
                p.requires_grad = False

    def freeze_temporal(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, (AttentionBlock, BasicVSRPP)) or (
                isinstance(m, ResBlock) and m.dims == 2
            ):
                for p in m.parameters():
                    p.requires_grad = True
        if hasattr(self, "dav"):
            for p in self.dav.spynet.parameters():
                p.requires_grad = False
        for p in self.input_blocks[0].parameters():
            p.requires_grad = True
        for p in self.out.parameters():
            p.requires_grad = True

    def freeze_for_image_training(self):
        for p in self.parameters():
            p.requires_grad = True

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        for m in self.modules():
            if isinstance(m, TemporalAttention):
                for p in m.q_linear.parameters():
                    p.data = p.data.to(th.float16)
                for p in m.k_linear.parameters():
                    p.data = p.data.to(th.float16)
                for p in m.v_linear.parameters():
                    p.data = p.data.to(th.float16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        for m in self.modules():
            if isinstance(m, TemporalAttention):
                for p in m.q_linear.parameters():
                    p.data = p.data.to(th.float32)
                for p in m.k_linear.parameters():
                    p.data = p.data.to(th.float32)
                for p in m.v_linear.parameters():
                    p.data = p.data.to(th.float32)

    def forward(
        self,
        x,
        timesteps,
        low_res_input=None,
        num_frames=None,
        sqrt_recip_alphas_cumprod=None,
        sqrt_recipm1_alphas_cumprod=None,
        old_ts=None,
        enable_cross_frames=True,
    ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x_t = x
        x = rearrange(x, "(b n) c h w -> b n c h w", n=num_frames)
        B, N, C, H, W = x.shape
        if low_res_input.shape[-2] != H or low_res_input.shape[-1] != W:
            low_res_input_scaled = rearrange(
                F.interpolate(
                    rearrange(low_res_input, "b n c h w -> (b n) c h w", n=num_frames),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ),
                "(b n) c h w -> b n c h w",
                n=num_frames,
            )
        else:
            low_res_input_scaled = low_res_input
        x_input = th.cat([x, low_res_input_scaled], dim=2)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []
        h = x_input.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, enable_cross_frames=enable_cross_frames)
            hs.append(h)
        h = self.middle_block(h, emb, enable_cross_frames=enable_cross_frames)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=2)
            h = module(h, emb, enable_cross_frames=enable_cross_frames)
        h = h.type(x.dtype)
        h = rearrange(self.out(h), "b n c h w -> (b n) c h w")
        if not enable_cross_frames or not self.cross_frame_module:
            return h
        x0 = self._predict_xstart_from_eps(
            x_t, old_ts, h, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod
        )
        x0_hat = rearrange(
            self.dav(
                rearrange(x0, "(b n) c h w -> b n c h w", n=num_frames),
                low_res_input,
                emb,
            ),
            "b n c h w -> (b n) c h w",
        )
        h = self._predict_eps_from_xstart(
            x_t, old_ts, x0_hat, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod
        )
        return h

    @staticmethod
    def _predict_xstart_from_eps(
        x_t, t, eps, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod
    ):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    @staticmethod
    def _predict_eps_from_xstart(
        x_t, t, pred_xstart, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod
    ):
        return (
            _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
