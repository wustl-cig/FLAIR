from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .nn_new import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .nn import LazyReshaper2D, LazyReshaper3D, FalshAttn
from einops import rearrange, repeat
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv,
    SPyNet,
)
from mmedit.models.common import PixelShufflePack, flow_warp
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d


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

    def forward(self, x, *args, enable_cross_frames=True, **kwargs):
        if not enable_cross_frames:
            return x
        output = self.wrapped_module(x, *args, **kwargs)
        return output


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


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

    def forward(self, x, emb, flows, vsrpp_weights, enable_cross_frames):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TemporalWrapper) and isinstance(
                layer.wrapped_module, BasicVSRPP
            ):
                flow = flows[x.shape[-1]]
                x = layer(
                    x, *flow, vsrpp_weights, enable_cross_frames=enable_cross_frames
                )
            elif isinstance(layer, TemporalWrapper) and isinstance(
                layer.wrapped_module, ResBlock
            ):
                x = layer(x, emb, enable_cross_frames=enable_cross_frames)
            elif isinstance(layer, TemporalWrapper) and isinstance(
                layer.wrapped_module, TemporalAttention
            ):
                x = layer(x, enable_cross_frames=enable_cross_frames)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
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

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
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
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            LazyReshaper3D(normalization(channels)),
            nn.SiLU(),
            LazyReshaper2D(conv_nd(dims, channels, self.out_channels, 3, padding=1))
            if dims == 2
            else LazyReshaper3D(
                conv_nd(dims, channels, self.out_channels, 3, padding=1)
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = LazyReshaper2D(Upsample(channels, False, dims))
            self.x_upd = LazyReshaper2D(Upsample(channels, False, dims))
        elif down:
            self.h_upd = LazyReshaper2D(Downsample(channels, False, dims))
            self.x_upd = LazyReshaper2D(Downsample(channels, False, dims))
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
            LazyReshaper3D(normalization(self.out_channels)),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                LazyReshaper2D(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                )
                if dims == 2
                else LazyReshaper3D(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = (
                LazyReshaper2D(conv_nd(dims, channels, self.out_channels, 3, padding=1))
                if dims == 2
                else LazyReshaper3D(
                    conv_nd(dims, channels, self.out_channels, 3, padding=1)
                )
            )
        else:
            self.skip_connection = (
                LazyReshaper2D(conv_nd(dims, channels, self.out_channels, 1))
                if dims == 2
                else LazyReshaper3D(conv_nd(dims, channels, self.out_channels, 1))
            )

    def forward(self, x, emb):
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
            self.emb_layers(emb).type(h.dtype), "(b n) c -> b n c () ()", n=x.shape[1]
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
        use_new_attention_order=False,
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
        self.norm = LazyReshaper3D(normalization(channels))
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, n, c, *spatial = x.shape
        qkv = self.qkv(rearrange(self.norm(x), "b n c h w -> (b n) c (h w)"))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, n, c, *spatial)


class AttentionbottleBlock(TimestepBlock):
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
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels

        self.emb_layers = nn.Sequential(nn.SiLU(), linear(512, 512))
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = LazyReshaper3D(normalization(channels))
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), True)

    def _forward(self, x, emb=None):
        b, n, c, *spatial = x.shape
        qkv = self.qkv(rearrange(self.norm(x), "b n c h w -> (b n) c (h w)"))
        h = self.attention(qkv)
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out.unsqueeze(-1)
        h = self.proj_out(h + emb_out)
        return x + h.reshape(b, n, c, *spatial)


class TemporalAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_frames,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
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
        self.proj = zero_module(LazyReshaper2D(conv_nd(2, channels, channels, 1)))

        self.norm = LazyReshaper3D(normalization(channels))

        t = timestep_embedding(
            (th.arange(self.num_frames, dtype=th.long) - (self.num_frames // 2)),
            channels,
        )
        self.t_mid = t[self.num_frames // 2 : self.num_frames // 2 + 1, ...]
        self.t_rest = t[th.arange(self.num_frames) != self.num_frames // 2, ...]

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
            :, th.arange(self.num_frames, device=x.device) != self.num_frames // 2, ...
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
        attn = self.attn(q, k, v)  # (B*T*H*W, 1, Head, C)
        attn = rearrange(
            attn, "(b t h w) () head c -> b t (head c) h w", b=B, t=T, h=H, w=W
        )
        x = self.proj(attn)
        return x + h


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


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

    def __init__(self, mid_channels=64, max_residue_magnitude=10, use_checkpoint=False):
        super().__init__()
        self.mid_channels = mid_channels
        self.use_checkpoint = use_checkpoint
        # optical flow

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        # self.norm = nn.ModuleDict()
        modules = ["backward_1", "forward_1"]
        for i, module in enumerate(modules):
            if th.cuda.is_available():
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
                feat_n2 = th.zeros_like(feat_prop)
                flow_n2 = th.zeros_like(flow_n1)
                cond_n2 = th.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :].to(feat_current.dtype)

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = th.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = th.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](
                    feat_prop, cond, flow_n1, flow_n2
                )

            # concatenate and residual blocks
            feat = (
                [feat_current]
                + [feats[k][idx] for k in feats if k not in ["spatial", module_name]]
                + [feat_prop]
            )

            feat = th.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
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
            hr = th.cat(hr, dim=1)
            hr = self.reconstruction(hr)
            recons.append(hr)
        recons = th.stack(recons, dim=1)
        recons = rearrange(
            self.conv_last(rearrange(recons, "n t c h w -> (n t) c h w")),
            "(n t) c h w -> n t c h w",
            n=hidden.shape[0],
        )
        outputs = recons + hidden
        return outputs

    def forward(self, hidden, flows_forward, flows_backward, weight):
        return checkpoint(
            lambda *x: self._forward(*x, flows_forward, flows_backward, weight),
            (hidden,),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, hidden, flows_forward, flows_backward, weight, **kwargs):
        # def forward(self, lqs):         #TODO
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = hidden.size()
        feats = {}
        # compute spatial features
        feats["spatial"] = [hidden[:, i, :, :, :] for i in range(0, t)]

        if weight is None:
            weight = th.ones(n, t, 1, 1, 1, device=hidden.device)
        elif isinstance(weight, float):
            weight = th.ones(n, t, 1, 1, 1, device=hidden.device) * weight
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
        extra_feat = th.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = th.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * th.tanh(th.cat((o1, o2), dim=1))
        offset_1, offset_2 = th.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = th.cat([offset_1, offset_2], dim=1)

        # mask
        mask = th.sigmoid(mask)

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


class UNetModel(nn.Module):
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
        rnn_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        temporal_block=False,
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
        self.rnn_resolutions = rnn_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.need_flows_res = [image_size // s for s in rnn_resolutions]

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.spynet = SPyNet(pretrained=None)
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    LazyReshaper2D(conv_nd(dims, in_channels, ch, 3, padding=1))
                )
            ]
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
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if temporal_block:
                    layers.append(
                        TemporalWrapper(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                use_scale_shift_norm=use_scale_shift_norm,
                                dims=3,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                    )
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    if temporal_block:
                        layers.append(
                            TemporalWrapper(
                                TemporalAttention(
                                    ch, 5, num_heads, num_head_channels, use_checkpoint
                                )
                            )
                        )
                if ds in rnn_resolutions and temporal_block:
                    layers.append(
                        TemporalWrapper(
                            BasicVSRPP(mid_channels=ch, use_checkpoint=use_checkpoint)
                        )
                    )
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
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else LazyReshaper2D(
                            Downsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
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
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            TemporalWrapper(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=3,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            if temporal_block
            else nn.Identity(),
            AttentionbottleBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            TemporalWrapper(
                TemporalAttention(ch, 5, num_heads, num_head_channels, use_checkpoint)
            )
            if temporal_block
            else nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            TemporalWrapper(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=3,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            if temporal_block
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
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if temporal_block:
                    layers.append(
                        TemporalWrapper(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                use_scale_shift_norm=use_scale_shift_norm,
                                dims=3,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                    )
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    if temporal_block:
                        layers.append(
                            TemporalWrapper(
                                TemporalAttention(
                                    ch,
                                    5,
                                    num_heads_upsample,
                                    num_head_channels,
                                    use_checkpoint,
                                )
                            )
                        )
                if ds in rnn_resolutions and temporal_block:
                    layers.append(
                        TemporalWrapper(
                            BasicVSRPP(mid_channels=ch, use_checkpoint=use_checkpoint)
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else LazyReshaper2D(
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            LazyReshaper3D(normalization(ch)),
            nn.SiLU(),
            zero_module(
                LazyReshaper2D(conv_nd(dims, input_ch, out_channels, 3, padding=1))
            ),
        )

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

    def freeze_spatial(self):
        """
        Freeze the model.
        """
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, (TemporalWrapper)):
                for p in m.parameters():
                    p.requires_grad = True
        for p in self.spynet.parameters():
            p.requires_grad = False

    def freeze_temporal(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, (AttentionBlock, ResBlock, AttentionbottleBlock)):
                for p in m.parameters():
                    p.requires_grad = True
        for p in self.spynet.parameters():
            p.requires_grad = False

    def freeze_for_image_training(self):
        for p in self.parameters():
            p.requires_grad = True

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

    def forward(
        self,
        x,
        timesteps,
        low_res_input=None,
        num_frames=None,
        rnn_input=None,
        enable_cross_frames=True,
        vsrpp_weights=None,
        **kwargs,
    ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = rearrange(x, "(b n) c h w -> b n c h w", n=num_frames)
        x = th.cat([x, low_res_input], dim=2)
        if rnn_input is None:
            rnn_input = low_res_input
        flows = {}
        for res in self.need_flows_res:
            if rnn_input.shape[-1] != res:
                flow_input = rearrange(
                    F.interpolate(
                        rearrange(rnn_input, "b n c h w -> (b n) c h w"),
                        (res, res),
                        mode="bicubic",
                    ),
                    "(b n) c h w -> b n c h w",
                    n=num_frames,
                )
            else:
                flow_input = rnn_input
            flows[res] = self.compute_flow(flow_input)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, flows, vsrpp_weights, enable_cross_frames)
            hs.append(h)
        h = self.middle_block(h, emb, flows, vsrpp_weights, enable_cross_frames)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=2)
            h = module(h, emb, flows, vsrpp_weights, enable_cross_frames)
        h = h.type(x.dtype)
        return rearrange(self.out(h), "b n c h w -> (b n) c h w")


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
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
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
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
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
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
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
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
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
