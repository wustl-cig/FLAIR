import argparse
import inspect

import torch

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .sr3 import UNet as BicubicUNet
from .unet_new import UNetModel as BlurUNet

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        test_start_timesteps=None,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        task="street",  # "face"
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="",
        vsrpp_resolutions="",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        cross_frame_module=True,
        res3d_kernel_size=(3, 1, 1),
        temp_attn_num_frames=5,
        norm_type="group_norm",
        spatial_attn=True,
        temporal_norm_type=None,
        rebuttal="none",
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    task,
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    vsrpp_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    cross_frame_module,
    res3d_kernel_size,
    temp_attn_num_frames,
    norm_type,
    spatial_attn,
    temporal_norm_type,
    test_start_timesteps,
    rebuttal,
):
    if task == "face_bicubic":
        noise_schedule = "face_bicubic"
        diffusion_steps = 2000
        t_schedule = "uniform"
    elif task == "face_blur":
        noise_schedule = "face_blur"
        diffusion_steps = 1000
        learn_sigma = True
        t_schedule = "uniform"
    model = create_model(
        task,
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        vsrpp_resolutions=vsrpp_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        cross_frame_module=cross_frame_module,
        res3d_kernel_size=res3d_kernel_size,
        temp_attn_num_frames=temp_attn_num_frames,
        norm_type=norm_type,
        spatial_attn=spatial_attn,
        temporal_norm_type=temporal_norm_type,
        rebuttal=rebuttal,
    )
    diffusion = create_gaussian_diffusion(
        diffusion_steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        test_start_timesteps=test_start_timesteps,
        t_schedule=t_schedule,
    )
    return model, diffusion


def create_model(
    task,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    vsrpp_resolutions="512",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    cross_frame_module=False,
    res3d_kernel_size=(3, 1, 1),
    temp_attn_num_frames=5,
    norm_type="group_norm",
    spatial_attn=True,
    temporal_norm_type=None,
    rebuttal="none",
):
    if task == "face_blur":
        return BlurUNet(
            image_size=512,
            in_channels=6,
            model_channels=128,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=(
                512 // 32,
                512 // 16,
                512 // 8,
            ),
            rnn_resolutions=(1, 2),
            channel_mult=(0.5, 1, 1, 2, 2, 4, 4),
            use_fp16=use_fp16,
            num_head_channels=64,
            resblock_updown=True,
            use_scale_shift_norm=True,
            temporal_block=cross_frame_module,
            use_checkpoint=use_checkpoint,
        )
    elif task == "face_bicubic":
        if rebuttal=="none":
            attn_res = (64, 32)
            vsrpp_res = (512, 256)
        elif rebuttal=="res":
            attn_res = ()
            vsrpp_res = ()
        elif rebuttal=="attn":
            attn_res = (64, 32)
            vsrpp_res = ()
        elif rebuttal=="rnn":
            attn_res = ()
            vsrpp_res = (512, 256)
        return BicubicUNet(
            image_size=512,
            in_channel=6,
            out_channel=3,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8, 16),
            attn_res=attn_res,
            vsrpp_res=vsrpp_res,
            spatial_attn=False,
            temporal_attn=cross_frame_module,
            res_blocks=1,
            dropout=0.0,
            dtype=torch.float16 if use_fp16 else torch.float32,
            cross_frame_module=cross_frame_module,
            use_checkpoint=use_checkpoint,
            num_frames=7,
            head_dim=64,
        )


def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=True,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    test_start_timesteps=None,
    t_schedule="uniform",
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if test_start_timesteps is not None:
        test_start_timesteps = int(test_start_timesteps)
    if not timestep_respacing:
        timestep_respacing = [
            diffusion_steps if test_start_timesteps is None else test_start_timesteps
        ]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(
            diffusion_steps if test_start_timesteps is None else test_start_timesteps,
            timestep_respacing,
            t_schedule,
        ),
        noise_schedule=noise_schedule,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
