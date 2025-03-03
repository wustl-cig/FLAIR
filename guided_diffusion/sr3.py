import math
import time
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from .unet import (
    convert_module_to_f16,
    convert_module_to_f32,
    TemporalAttention,
    TemporalWrapper,
    ResBlock,
    BasicVSRPP,
    CrossFrameUNetModel,
)
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    SPyNet,
)
from .nn import (
    GroupNorm32,
    conv_nd,
    LazyReshaper2D,
    LazyReshaper3D,
    normalization,
    shift_window_normalization,
    checkpoint,
    zero_module,
    linear,
)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch, t = x.shape[:2]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embed)
                .to(x.dtype)
                .view(batch, 1, -1, 1, 1)
                .chunk(2, dim=2)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).to(x.dtype).view(batch, t, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            LazyReshaper3D(GroupNorm32(groups, dim)),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            LazyReshaper2D(nn.Conv2d(dim, dim_out, 3, padding=1)),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
        use_checkpoint=False,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = (
            LazyReshaper2D(nn.Conv2d(dim, dim_out, 1))
            if dim != dim_out
            else nn.Identity()
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x, time_emb):
        return checkpoint(
            self._forward, (x, time_emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32, use_checkpoint=False):
        super().__init__()

        self.n_head = n_head

        self.norm = LazyReshaper3D(GroupNorm32(norm_groups, in_channel))
        self.qkv = LazyReshaper2D(nn.Conv2d(in_channel, in_channel * 3, 1, bias=False))
        self.out = LazyReshaper2D(nn.Conv2d(in_channel, in_channel, 1))

        self.use_checkpoint = use_checkpoint

    def forward(self, input):
        return checkpoint(
            self._forward, (input,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class TemporalWrapper2(nn.Module):
    def __init__(self, module, dim, time_emb_dim=512):
        super().__init__()
        self.wrapped_module = module
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(
                linear(
                    time_emb_dim,
                    dim,
                )
            ),
        )

    def forward(self, x, emb, *args, enable_cross_frames=True, **kwargs):
        if not enable_cross_frames:
            return x
        B, N, C, H, W = x.shape
        output = self.wrapped_module(x, *args, **kwargs)
        weight = self.emb_layers(emb).view(B, N, C, 1, 1)
        output = (1 - torch.sigmoid(weight.to(x.dtype))) * x + torch.sigmoid(
            weight.to(x.dtype)
        ) * output
        return output


class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        conv_3d=False,
        spatial_attn=False,
        temporal_attn=False,
        conv_3d_kernel_size=(3, 1, 1),
        num_frames=5,
        head_dim=32,
        vsrpp=False,
        shared_spynet=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.spatial_attn = spatial_attn
        self.res_block = ResnetBlock(
            dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        if conv_3d:
            self.conv_3d = TemporalWrapper2(
                ResBlock(
                    dim_out,
                    noise_level_emb_dim,
                    0.0,
                    dims=3,
                    kernel_size=conv_3d_kernel_size,
                    padding=(
                        conv_3d_kernel_size[0] // 2,
                        conv_3d_kernel_size[1] // 2,
                        conv_3d_kernel_size[2] // 2,
                    ),
                    use_checkpoint=use_checkpoint,
                ),
                dim_out,
                time_emb_dim=noise_level_emb_dim,
            )
        if spatial_attn:
            self.attn = SelfAttention(
                dim_out, norm_groups=norm_groups, use_checkpoint=use_checkpoint
            )
        if temporal_attn:
            self.temp_attn = TemporalWrapper2(
                TemporalAttention(
                    dim_out,
                    num_frames=num_frames,
                    num_heads=8,
                    num_head_channels=head_dim,
                    use_checkpoint=use_checkpoint,
                ),
                dim_out,
                time_emb_dim=noise_level_emb_dim,
            )
        if vsrpp:
            self.vsrpp = TemporalWrapper2(
                BasicVSRPP(
                    dim_out,
                    max_residue_magnitude=5,
                    shared_spynet=shared_spynet,
                    use_checkpoint=use_checkpoint,
                ),
                dim_out,
                time_emb_dim=noise_level_emb_dim,
            )

    def forward(self, x, lqs, time_emb, cross_frame_enabled=True, vsrpp_weights=None):
        x = self.res_block(x, time_emb)
        if hasattr(self, "conv_3d") and cross_frame_enabled:
            x = self.conv_3d(x, time_emb, time_emb)
        if self.spatial_attn:
            x = self.attn(x)
        if hasattr(self, "temp_attn") and cross_frame_enabled:
            x = self.temp_attn(x, time_emb, time_emb)
        if hasattr(self, "vsrpp") and cross_frame_enabled:
            x = self.vsrpp(x, time_emb, lqs, weight=vsrpp_weights)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        vsrpp_res=(64,),
        spatial_attn=False,
        temporal_attn=False,
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        dtype=torch.float32,
        cross_frame_module=False,
        use_checkpoint=False,
        num_frames=5,
        head_dim=32,
    ):
        super().__init__()
        if len(vsrpp_res) > 0:
            shared_spynet = SPyNet(pretrained=None)
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        self.dtype = dtype
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [
            LazyReshaper2D(
                nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)
            )
        ]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            spatial_use_attn = now_res in attn_res and spatial_attn
            temporal_use_attn = (
                now_res in attn_res and temporal_attn and cross_frame_module
            )
            vsrpp_use = now_res in vsrpp_res and cross_frame_module
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        conv_3d=cross_frame_module,
                        spatial_attn=spatial_use_attn,
                        temporal_attn=temporal_use_attn,
                        num_frames=num_frames,
                        head_dim=head_dim,
                        use_checkpoint=use_checkpoint,
                        vsrpp=vsrpp_use,
                        shared_spynet=shared_spynet if vsrpp_use else None,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(LazyReshaper2D(Downsample(pre_channel)))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    conv_3d=cross_frame_module,
                    spatial_attn=spatial_attn,
                    temporal_attn=temporal_attn and cross_frame_module,
                    num_frames=num_frames,
                    head_dim=head_dim,
                    use_checkpoint=use_checkpoint,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    conv_3d=cross_frame_module,
                    spatial_attn=spatial_attn,
                    temporal_attn=temporal_attn and cross_frame_module,
                    num_frames=num_frames,
                    head_dim=head_dim,
                    use_checkpoint=use_checkpoint,
                ),
            ]
        )

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            spatial_use_attn = now_res in attn_res and spatial_attn
            temporal_use_attn = (
                now_res in attn_res and temporal_attn and cross_frame_module
            )
            vsrpp_use = now_res in vsrpp_res and cross_frame_module
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        conv_3d=cross_frame_module,
                        spatial_attn=spatial_use_attn,
                        temporal_attn=temporal_use_attn,
                        num_frames=num_frames,
                        head_dim=head_dim,
                        use_checkpoint=use_checkpoint,
                        vsrpp=vsrpp_use,
                        shared_spynet=shared_spynet if vsrpp_use else None,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(LazyReshaper2D(Upsample(pre_channel)))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(
            pre_channel, default(out_channel, in_channel), groups=norm_groups
        )

    def forward(
        self,
        x,
        timesteps,
        low_res_input=None,
        rnn_input=None,
        num_frames=None,
        enable_cross_frames=True,
        vsrpp_weights=None,
        **kwargs,
    ):
        if rnn_input is None:
            rnn_input = low_res_input
        x = rearrange(x, "(b n) c h w -> b n c h w", n=num_frames)
        x = torch.cat((low_res_input, x), dim=2) if exists(low_res_input) else x
        t = self.noise_level_mlp(timesteps) if exists(self.noise_level_mlp) else None

        feats = []
        dtype = x.dtype
        x = x.to(self.dtype)
        rnn_input = rnn_input.to(self.dtype)
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    x,
                    rnn_input,
                    t,
                    cross_frame_enabled=enable_cross_frames,
                    vsrpp_weights=vsrpp_weights,
                )
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    x,
                    rnn_input,
                    t,
                    cross_frame_enabled=enable_cross_frames,
                    vsrpp_weights=vsrpp_weights,
                )
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(
                    torch.cat((x, feats.pop()), dim=2),
                    rnn_input,
                    t,
                    cross_frame_enabled=enable_cross_frames,
                    vsrpp_weights=vsrpp_weights,
                )
            else:
                x = layer(x)
        x = x.to(dtype)
        eps = rearrange(self.final_conv(x), "b n c h w -> (b n) c h w")
        return eps

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.downs.apply(convert_module_to_f16)
        self.mid.apply(convert_module_to_f16)
        self.ups.apply(convert_module_to_f16)
        for m in self.modules():
            if isinstance(m, TemporalAttention):
                for p in m.q_linear.parameters():
                    p.data = p.data.to(torch.float16)
                for p in m.k_linear.parameters():
                    p.data = p.data.to(torch.float16)
                for p in m.v_linear.parameters():
                    p.data = p.data.to(torch.float16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.downs.apply(convert_module_to_f32)
        self.mid.apply(convert_module_to_f32)
        self.ups.apply(convert_module_to_f32)
        for m in self.modules():
            if isinstance(m, TemporalAttention):
                for p in m.q_linear.parameters():
                    p.data = p.data.to(torch.float32)
                for p in m.k_linear.parameters():
                    p.data = p.data.to(torch.float32)
                for p in m.v_linear.parameters():
                    p.data = p.data.to(torch.float32)

    def freeze_spatial(self):
        """
        Freeze the model.
        """
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(
                m,
                (TemporalWrapper, BasicVSRPP, TemporalWrapper2),
            ):
                for p in m.parameters():
                    p.requires_grad = True
            if isinstance(m, (SelfAttention, ResnetBlock)):
                m.use_checkpoint = False
            if isinstance(m, BasicVSRPP):
                for p in m.spynet.parameters():
                    p.requires_grad = False

    def freeze_temporal(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, (SelfAttention, BasicVSRPP, ResnetBlock)):
                for p in m.parameters():
                    p.requires_grad = True
            if isinstance(
                m,
                (TemporalWrapper, TemporalWrapper2)
                and isinstance(
                    m.wrapped_module, (TemporalAttention, BasicVSRPP, ResBlock)
                ),
            ):
                m.wrapped_module.use_checkpoint = False
            if isinstance(m, BasicVSRPP):
                for p in m.spynet.parameters():
                    p.requires_grad = False

    def freeze_for_image_training(self):
        for p in self.parameters():
            p.requires_grad = True

    def train_vsrpp(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, TemporalWrapper2) and isinstance(
                m.wrapped_module, BasicVSRPP
            ):
                for p in m.parameters():
                    p.requires_grad = True
                for p in m.wrapped_module.spynet.parameters():
                    p.requires_grad = False


if __name__ == "__main__":
    model = UNet(
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        norm_groups=16,
        channel_mults=(1, 2, 4, 8, 16),
        attn_res=(32,),
        image_size=512,
        spatial_attn=False,
        temporal_attn=True,
        res_blocks=1,
        dropout=0,
        dtype=torch.float32,
        cross_frame_module=True,
    ).to("cuda:0")

    x = torch.randn(1, 3, 512, 512).to("cuda:0")
    low_res_input = torch.randn_like(x).to("cuda:0").unsqueeze(1)
    timesteps = torch.randn(1).to("cuda:0")
    old_ts = torch.randint_like(timesteps, 0, 100, dtype=torch.long).to("cuda:0")
    sqrt_recip_alphas_cumprod = np.random.randn(2000)
    sqrt_recipm1_alphas_cumprod = np.random.randn(2000)
    y = model(
        x,
        timesteps,
        num_frames=1,
        low_res_input=low_res_input,
        old_ts=old_ts,
        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
    )
