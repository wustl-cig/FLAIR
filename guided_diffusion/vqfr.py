import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from distutils.version import LooseVersion
from einops import rearrange
from timm.models.layers import trunc_normal_
from guided_diffusion.dcn import ModulatedDeformConvPack, modulated_deform_conv


class L2VectorQuantizer(nn.Module):
    def __init__(self, num_code, code_dim, spatial_size):
        super().__init__()
        self.num_code = num_code
        self.code_dim = code_dim
        self.spatial_size = spatial_size

        self.beta = 0.25

        self.embedding = nn.Embedding(self.num_code, self.code_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_code, 1.0 / self.num_code)

        self.register_buffer(
            "usage", torch.zeros(self.num_code, dtype=torch.int), persistent=False
        )

    def get_feature(self, indices):
        quant = self.embedding(indices)
        quant = rearrange(quant, "b (h w) c -> b c h w", h=self.spatial_size[0])
        return quant

    def get_distance(self, z, embedding):
        # (b h w) c
        z_flattened = z.view(-1, self.code_dim)
        distance = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("b c, d c -> b d", z_flattened, embedding)
        )
        return distance

    def compute_codebook_loss(self, z_quant, z):
        loss = torch.mean((z_quant.detach() - z) ** 2) + self.beta * torch.mean(
            (z_quant - z.detach()) ** 2
        )
        return loss

    def reset_usage(self):
        self.usage = self.usage * 0

    def get_usage(self):
        codebook_usage = 1.0 * (self.num_code - (self.usage == 0).sum()) / self.num_code
        return codebook_usage

    def forward(self, z, iters=-1):
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        embedding = self.embedding.weight

        distance = self.get_distance(z, embedding)

        min_encoding_indices = torch.argmin(distance, dim=1)

        if not self.training:
            for idx in range(self.num_code):
                self.usage[idx] += (min_encoding_indices == idx).sum()

        z_quant = self.embedding(min_encoding_indices).view(z.shape)

        loss = self.compute_codebook_loss(z_quant, z)
        # preserve gradients
        z_quant = z + (z_quant - z).detach()
        # reshape back to match original input shape
        z_quant = rearrange(z_quant, "b h w c -> b c h w").contiguous()
        min_encoding_indices = rearrange(
            min_encoding_indices, "(b n)->b n", b=z_quant.shape[0]
        )
        return z_quant, loss, (min_encoding_indices)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=channels_in, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            channels_in, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1
        )

        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=channels_out, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            channels_out, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1
        )

        self.act = nn.SiLU(inplace=True)
        if channels_in != channels_out:
            self.residual_func = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.residual_func = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + self.residual_func(residual)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class VQGANEncoder(nn.Module):
    def __init__(
        self,
        base_channels,
        channel_multipliers,
        num_blocks,
        use_enc_attention,
        code_dim,
    ):
        super(VQGANEncoder, self).__init__()

        self.num_levels = len(channel_multipliers)
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(
            3,
            base_channels * channel_multipliers[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )
        self.blocks = nn.ModuleList()

        for i in range(self.num_levels):
            blocks = []
            if i == 0:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i - 1]

            if i != 0:
                blocks.append(Downsample(channels_prev))

            channels = base_channels * channel_multipliers[i]
            blocks.append(ResnetBlock(channels_prev, channels))
            if i == self.num_levels - 1 and use_enc_attention:
                blocks.append(AttnBlock(channels))

            for j in range(self.num_blocks - 1):
                blocks.append(ResnetBlock(channels, channels))
                if i == self.num_levels - 1 and use_enc_attention:
                    blocks.append(AttnBlock(channels))

            self.blocks.append(nn.Sequential(*blocks))

        channels = base_channels * channel_multipliers[-1]
        if use_enc_attention:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels),
                AttnBlock(channels),
                ResnetBlock(channels, channels),
            )
        else:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels), ResnetBlock(channels, channels)
            )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, code_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.num_levels):
            x = self.blocks[i](x)
        x = self.mid_blocks(x)
        x = self.conv_out(x)
        return x


class VQGANDecoder(nn.Module):
    def __init__(
        self,
        base_channels,
        channel_multipliers,
        num_blocks,
        use_dec_attention,
        code_dim,
    ):
        super(VQGANDecoder, self).__init__()

        self.num_levels = len(channel_multipliers)
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(
            code_dim,
            base_channels * channel_multipliers[-1],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )
        self.blocks = nn.ModuleList()

        channels = base_channels * channel_multipliers[-1]

        if use_dec_attention:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels),
                AttnBlock(channels),
                ResnetBlock(channels, channels),
            )
        else:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels), ResnetBlock(channels, channels)
            )

        for i in reversed(range(self.num_levels)):
            blocks = []

            if i == self.num_levels - 1:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i + 1]

            if i != self.num_levels - 1:
                blocks.append(Upsample(channels_prev))

            channels = base_channels * channel_multipliers[i]
            blocks.append(ResnetBlock(channels_prev, channels))
            if i == self.num_levels - 1 and use_dec_attention:
                blocks.append(AttnBlock(channels))

            for j in range(self.num_blocks - 1):
                blocks.append(ResnetBlock(channels, channels))
                if i == self.num_levels - 1 and use_dec_attention:
                    blocks.append(AttnBlock(channels))
            self.blocks.append(nn.Sequential(*blocks))

        channels = base_channels * channel_multipliers[0]
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x, return_feat=False):
        dec_res = {}
        x = self.conv_in(x)
        x = self.mid_blocks(x)
        for i, level in enumerate(reversed(range(self.num_levels))):
            x = self.blocks[i](x)
            dec_res["Level_%d" % 2**level] = x
        x = self.conv_out(x)
        if return_feat:
            return x, dec_res
        else:
            return x


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        if LooseVersion(torchvision.__version__) >= LooseVersion("0.9.0"):
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
        else:
            return modulated_deform_conv(
                x,
                offset,
                mask,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.deformable_groups,
            )


class TextureWarpingModule(nn.Module):
    def __init__(
        self,
        channel,
        cond_channels,
        cond_downscale_rate,
        deformable_groups,
        previous_offset_channel=0,
    ):
        super(TextureWarpingModule, self).__init__()
        self.cond_downscale_rate = cond_downscale_rate
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(channel + cond_channels, channel, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, groups=channel, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1),
        )

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(channel + previous_offset_channel, channel, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
        )
        self.dcn = DCNv2Pack(
            channel, channel, 3, padding=1, deformable_groups=deformable_groups
        )

    def forward(self, x_main, inpfeat, previous_offset=None):
        _, _, h, w = inpfeat.shape
        inpfeat = F.interpolate(
            inpfeat,
            size=(h // self.cond_downscale_rate, w // self.cond_downscale_rate),
            mode="bilinear",
            align_corners=False,
        )
        offset = self.offset_conv1(torch.cat([inpfeat, x_main], dim=1))
        if previous_offset is None:
            offset = self.offset_conv2(offset)
        else:
            offset = self.offset_conv2(torch.cat([offset, previous_offset], dim=1))
        warp_feat = self.dcn(x_main, offset)
        return warp_feat, offset


class MainDecoder(nn.Module):
    def __init__(self, base_channels, channel_multipliers, align_opt):
        super(MainDecoder, self).__init__()
        self.num_levels = len(channel_multipliers)

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()
        self.align_func_dict = nn.ModuleDict()

        for i in reversed(range(self.num_levels)):
            if i == self.num_levels - 1:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i + 1]
            channels = base_channels * channel_multipliers[i]

            if i != self.num_levels - 1:
                self.pre_upsample_dict["Level_%d" % 2**i] = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(channels_prev, channels, kernel_size=3, padding=1),
                )

            previous_offset_channel = 0 if i == self.num_levels - 1 else channels_prev

            self.align_func_dict["Level_%d" % (2**i)] = TextureWarpingModule(
                channel=channels,
                cond_channels=align_opt["cond_channels"],
                cond_downscale_rate=2**i,
                deformable_groups=align_opt["deformable_groups"],
                previous_offset_channel=previous_offset_channel,
            )

            if i != self.num_levels - 1:
                self.decoder_dict["Level_%d" % 2**i] = ResnetBlock(
                    2 * channels, channels
                )

    def forward(self, dec_res_dict, inpfeat, fidelity_ratio=1.0):
        x, offset = self.align_func_dict["Level_%d" % 2 ** (self.num_levels - 1)](
            dec_res_dict["Level_%d" % 2 ** (self.num_levels - 1)], inpfeat
        )

        for scale in reversed(range(self.num_levels - 1)):
            x = self.pre_upsample_dict["Level_%d" % 2**scale](x)
            upsample_offset = (
                F.interpolate(
                    offset, scale_factor=2, align_corners=False, mode="bilinear"
                )
                * 2
            )
            warp_feat, offset = self.align_func_dict["Level_%d" % 2**scale](
                dec_res_dict["Level_%d" % 2**scale],
                inpfeat,
                previous_offset=upsample_offset,
            )
            x = self.decoder_dict["Level_%d" % 2**scale](
                torch.cat([x, warp_feat], dim=1)
            )
        return dec_res_dict["Level_1"] + fidelity_ratio * x


class VQFRv2(nn.Module):
    def __init__(
        self,
        base_channels,
        channel_multipliers,
        num_enc_blocks,
        use_enc_attention,
        num_dec_blocks,
        use_dec_attention,
        code_dim,
        inpfeat_dim,
        code_selection_mode,
        align_opt,
    ):
        super().__init__()

        self.encoder = VQGANEncoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_enc_blocks,
            use_enc_attention=use_enc_attention,
            code_dim=code_dim,
        )

        if code_selection_mode == "Nearest":
            self.feat2index = None
        elif code_selection_mode == "Predict":
            self.feat2index = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 1024),
            )

        self.decoder = VQGANDecoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_dec_blocks,
            use_dec_attention=use_dec_attention,
            code_dim=code_dim,
        )

        self.main_branch = MainDecoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            align_opt=align_opt,
        )
        self.inpfeat_extraction = nn.Conv2d(3, inpfeat_dim, 3, padding=1)

        self.quantizer = L2VectorQuantizer(
            num_code=1024, code_dim=256, spatial_size=(16, 16)
        )

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_last_layer(self):
        return self.decoder.conv_out[-1].weight

    def forward(self, x_lq, fidelity_ratio=1.0):
        inp_feat = self.inpfeat_extraction(x_lq)
        res = {}
        enc_feat = self.encoder(x_lq)
        res["enc_feat"] = enc_feat

        if self.feat2index is not None:
            # cross-entropy predict token
            enc_feat = rearrange(enc_feat, "b c h w -> b (h w) c")
            quant_logit = self.feat2index(enc_feat)
            res["quant_logit"] = quant_logit
            quant_index = quant_logit.argmax(-1)
            quant_feat = self.quantizer.get_feature(quant_index)
        else:
            # nearest predict token
            quant_feat, _, _ = self.quantizer(enc_feat)

        with torch.no_grad():
            texture_dec, texture_feat_dict = self.decoder(quant_feat, return_feat=True)
            res["texture_dec"] = texture_dec

        main_feature = self.main_branch(
            texture_feat_dict, inp_feat, fidelity_ratio=fidelity_ratio
        )
        main_dec = self.decoder.conv_out(main_feature)
        res["main_dec"] = main_dec
        return res
