from functools import partial
from typing import Any, Dict, List, Tuple, Union, Optional, Sequence, Iterable, Literal
from typing_extensions import Annotated
from pathlib import Path
from copy import deepcopy
import sys
sys.path.append(f"{Path(__file__).parent.parent.resolve()}")
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from guided_diffusion.gaussian_diffusion import ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from guided_diffusion.respace import space_timesteps, SpacedDiffusion
from guided_diffusion.sr3 import UNet as BicubicUNet
from guided_diffusion.unet_new import UNetModel as BlurUNet
import guided_diffusion.pseudoSR as pseudo_sr
from guided_diffusion.restore_util import SRConv
from guided_diffusion.codeformer import CodeFormer
import cyclopts
from cyclopts import validators
from natsort import natsorted
import cv2
import hdf5storage
import numpy as np
import os
import more_itertools as mit
from einops import rearrange
from guided_diffusion.jpeg import jpeg_decode, jpeg_encode
from guided_diffusion.facelib.utils.face_restoration_helper import FaceRestoreHelper


torch.set_grad_enabled(False)

app = cyclopts.App()

DIFFUSION_CONFIG = {
    "x8_bicubic": {
        "diffusion_steps": 2000,
        "noise_schedule": "face_bicubic",
        "model_mean_type": ModelMeanType.EPSILON,
        "model_var_type": ModelVarType.FIXED_SMALL,
        "loss_type": LossType.MSE,
        "rescale_timesteps": False,
    },
    "x16_bicubic": {
        "diffusion_steps": 2000,
        "noise_schedule": "face_bicubic",
        "model_mean_type": ModelMeanType.EPSILON,
        "model_var_type": ModelVarType.FIXED_SMALL,
        "loss_type": LossType.MSE,
        "rescale_timesteps": False,
    },
    "gaussian": {
        "diffusion_steps": 1000,
        "noise_schedule": "face_blur",
        "model_mean_type": ModelMeanType.EPSILON,
        "model_var_type": ModelVarType.LEARNED_RANGE,
        "loss_type": LossType.RESCALED_MSE,
        "rescale_timesteps": False,
    },
    "jpeg": {
        "diffusion_steps": 1000,
        "noise_schedule": "face_blur",
        "model_mean_type": ModelMeanType.EPSILON,
        "model_var_type": ModelVarType.LEARNED_RANGE,
        "loss_type": LossType.RESCALED_MSE,
        "rescale_timesteps": False,
    },
}

MODEL_TYPE = {
    "x8_bicubic": BicubicUNet,
    "x16_bicubic": BicubicUNet,
    "gaussian": BlurUNet,
    "jpeg": BlurUNet,
}

MODEL_CONFIG = {
    "x8_bicubic": {
        "image_size": 512,
        "in_channel": 6,
        "out_channel": 3,
        "inner_channel": 64,
        "norm_groups": 16,
        "channel_mults": (1, 2, 4, 8, 16),
        "attn_res": (64, 32),
        "vsrpp_res": (512, 256),
        "spatial_attn": False,
        "temporal_attn": True,
        "res_blocks": 1,
        "dropout": 0.0,
        "dtype": torch.float16,
        "cross_frame_module": True,
        "use_checkpoint": True,
        "num_frames": 7,
        "head_dim": 64,
    },
    "x16_bicubic": {
        "image_size": 512,
        "in_channel": 6,
        "out_channel": 3,
        "inner_channel": 64,
        "norm_groups": 16,
        "channel_mults": (1, 2, 4, 8, 16),
        "attn_res": (64, 32),
        "vsrpp_res": (512, 256),
        "spatial_attn": False,
        "temporal_attn": True,
        "res_blocks": 1,
        "dropout": 0.0,
        "dtype": torch.float16,
        "cross_frame_module": True,
        "use_checkpoint": True,
        "num_frames": 7,
        "head_dim": 64,
    },
    "gaussian": {
        "image_size": 512,
        "in_channels": 6,
        "model_channels": 128,
        "out_channels": 6,
        "num_res_blocks": 2,
        "attention_resolutions": (
            512 // 32,
            512 // 16,
            512 // 8,
        ),
        "rnn_resolutions": (1, 2),
        "channel_mult": (0.5, 1, 1, 2, 2, 4, 4),
        "use_fp16": True,
        "num_head_channels": 64,
        "resblock_updown": True,
        "use_scale_shift_norm": True,
        "temporal_block": True,
        "use_checkpoint": True,
    },
    "jpeg": {
        "image_size": 512,
        "in_channels": 6,
        "model_channels": 128,
        "out_channels": 6,
        "num_res_blocks": 2,
        "attention_resolutions": (
            512 // 32,
            512 // 16,
            512 // 8,
        ),
        "rnn_resolutions": (1, 2),
        "channel_mult": (0.5, 1, 1, 2, 2, 4, 4),
        "use_fp16": True,
        "num_head_channels": 64,
        "resblock_updown": True,
        "use_scale_shift_norm": True,
        "temporal_block": True,
        "use_checkpoint": True,
    },
}

INIT_FUNC = {
    "x8_bicubic": lambda x: VF.resize(x, (512, 512), VF.InterpolationMode.BICUBIC).clamp(0, 1),
    "x16_bicubic": lambda x: VF.resize(x, (512, 512), VF.InterpolationMode.BICUBIC).clamp(0, 1),
    "gaussian": lambda x: F.interpolate(x, (512, 512), mode="area").clamp(0, 1),
    "jpeg": lambda x: F.interpolate(x, (512, 512), mode="area").clamp(0, 1),
}

CKPT_PATH = {
    "x8_bicubic": "./checkpoints/flair_x8_bicubic.pt",
    "x16_bicubic": "./checkpoints/flair_x16_bicubic.pt",
    "gaussian": "./checkpoints/flair_gaussian.pt",
    "jpeg": "./checkpoints/flair_jpeg.pt",
    "codeformer": "./checkpoints/codeformer.pth",
}



DEFAULT_WEIGHT = 1.0

def bicubic_restore(x, d, A_func):
    return A_func.A_pinv(
        A_func.A(x.reshape(x.shape[0], -1))
        - d.reshape(x.shape[0], -1)
    ).reshape(*x.size())

def gaussian_restore(x, d, A_func, jpeg_qf=-1):
    return A_func.A_pinv(
        rearrange(d, "b t c h w -> (b t) c h w"),
        x,
        jpeg_encode=(lambda img: jpeg_encode(img, jpeg_qf))
        if jpeg_qf != -1
        else None,
        jpeg_decode=(lambda img: jpeg_decode(img, jpeg_qf))
        if jpeg_qf != -1
        else None,
    )

RESTORE_FUNC = {
    "x8_bicubic": bicubic_restore,
    "x16_bicubic": bicubic_restore,
    "gaussian": gaussian_restore,
    "jpeg": gaussian_restore,
}

FRAME_SLICE_LEN = 10
OVERLAP = 3

def get_A_func(task: Literal["x8_bicubic", "x16_bicubic", "gaussian", "jpeg"], device: torch.device):
    if task == "x8_bicubic":
        factor = 8
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return (
                    a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                )
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        A_func = SRConv(
            kernel / kernel.sum(), 3, 512, device, stride=factor
        )
    elif task == "x16_bicubic":
        factor = 16
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return (
                    a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                )
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        A_func = SRConv(
            kernel / kernel.sum(), 3, 512, device, stride=factor
        )
    elif task == "gaussian" or task == "jpeg":
        kernel = hdf5storage.loadmat(
            os.path.join(
                "./miscs/kernels_12.mat"
            )
        )["kernels"]
        factor = int(4)
        pseudoSR_conf = pseudo_sr.Get_pseudoSR_Conf(factor)
        pseudoSR_conf.sigmoid_range_limit = False
        pseudoSR_conf.input_range = np.array(None)
        A_func = pseudo_sr.pseudoSR(
            pseudoSR_conf, upscale_kernel=kernel[0, 3], kernel_indx=10
        )
        A_func = A_func.WrapArchitecture_PyTorch().to(device)
        
    return A_func

@app.default()
def main(
    task: Literal["x8_bicubic", "x16_bicubic", "gaussian", "jpeg"],
    video_path: Annotated[Path, cyclopts.Parameter(validator=validators.Path(exists=True, file_okay=False, dir_okay=True))],
    output_path: Path,
    device: Annotated[torch.device, cyclopts.Parameter(converter=lambda t, v: torch.device(v))]=torch.device("cuda"),
    t_start: int=-1,
    jpeg_qf: int=-1,
    w: float=0.5,
    tau: int=5,
    aligned: bool=False,
    rho: float=0.5,
    noise_level: float=12.75,
    zeta: float=-1,
):  
    """FLAIR video restoration

    Parameters
    ----------
    task : Literal[&quot;x8_bicubic&quot;, &quot;x16_bicubic&quot;, &quot;gaussian&quot;, &quot;jpeg&quot;]
        The task to perform
    video_path : str
        The path to the video frame directory
    output_path : str
        The output directory
    device : 
        The device to use, by default using cuda
    t_start : int, optional
        The start timestep for sampling, by default -1
    jpeg_qf : int, optional
        The jpeg quality factor, by default -1
    w : float, optional
        The weight for the auxiliary model, by default 0.5
    tau : int, optional
        The ending timestep for using auxiliary model, by default 5
    aligned : bool, optional
        Whether to use aligned face restoration, by default False
    rho : float, optional
        The rho value in the paper
    noise_level : float, optional
        The noise level
    zeta : float, optional
        The zeta value in the paper
    """    
    print(f"task: {task}, video_path: {video_path}, output_dir: {output_path}, device: {device}")
    # diffusion config
    diffusion_config = deepcopy(DIFFUSION_CONFIG[task])
    diffusion_steps = diffusion_config.pop("diffusion_steps")
    diffusion_config["use_timesteps"] = space_timesteps(
        diffusion_steps,
        "100",
        "uniform",
    )
    diffusion_config["betas"] = get_named_beta_schedule(
        diffusion_config.get("noise_schedule"),
        diffusion_steps,
    )

    diffusion = SpacedDiffusion(
        **diffusion_config
    )

    model: torch.nn.Module = MODEL_TYPE[task](**MODEL_CONFIG[task]).to(device)
    model.convert_to_fp16()
    model.eval()
    model.load_state_dict(torch.load(CKPT_PATH[task], map_location="cpu"))

    face_helper = FaceRestoreHelper(device=device)

    frame_path_lst = natsorted(video_path.glob("*.[jJpP][pPnN][gG]"))
    print(f"found {len(frame_path_lst)} frames")

    degraded_frames = torch.stack([
        torch.from_numpy(
            cv2.cvtColor(
                cv2.imread(str(frame_path)),
                cv2.COLOR_BGR2RGB
            ).transpose(2, 0, 1)
        ).float() / 255
        for frame_path in frame_path_lst
    ]).unsqueeze(0)

    A_func = get_A_func(task, device)



    gan = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    gan.load_state_dict(torch.load(CKPT_PATH["codeformer"], map_location="cpu")["params_ema"])
    gan.eval()

    windowed_degraded_imgs = [
            torch.stack(list(filter(lambda x: x is not None, w)), 1)
            for w in mit.windowed(
                degraded_frames.unbind(1),
                FRAME_SLICE_LEN,
                step=FRAME_SLICE_LEN - OVERLAP,
            )
        ]
    prev_recon = None
    recon_frames = []
    for sliced_degraded_frames in windowed_degraded_imgs:
        sliced_init_frames: torch.Tensor = INIT_FUNC[task](sliced_degraded_frames.squeeze(0)).unsqueeze(0).clamp(0, 1)
        sliced_degraded_frames = (sliced_degraded_frames - 0.5) / 0.5
        sliced_init_frames = (sliced_init_frames - 0.5) / 0.5

        sliced_degraded_frames = sliced_degraded_frames.to(device)
        sliced_init_frames = sliced_init_frames.to(device)
        if t_start == -1:
            noise = diffusion.q_sample(
                rearrange(sliced_init_frames, "b n c h w -> (b n) c h w"),
                torch.ones(
                    sliced_init_frames.shape[0] * sliced_init_frames.shape[1],
                    dtype=torch.long,
                    device=device,
                )
                * (diffusion.num_timesteps - 1),
            )
        else:
            noise = diffusion.q_sample(
                rearrange(sliced_init_frames, "b n c h w -> (b n) c h w"),
                torch.ones(
                    sliced_init_frames.shape[0] * sliced_init_frames.shape[1],
                    dtype=torch.long,
                    device=device,
                )
                * (t_start),
            )

        model_kwargs = {
            "low_res_input": sliced_init_frames,
            "num_frames": sliced_init_frames.shape[1],
            "enable_cross_frames": True,
            "vsrpp_weights": DEFAULT_WEIGHT,
        }

        if task == "gaussian" or task == "jpeg":
            model_kwargs["rnn_input"] = rearrange(
                VF.normalize(
                    VF.resize(
                        VF.normalize(
                            rearrange(
                                sliced_degraded_frames, "b t c h w -> (b t) c h w"
                            ),
                            0.5,
                            0.5,
                        ),
                        (512, 512),
                        VF.InterpolationMode.BICUBIC,
                    ),
                    -1,
                    2,
                ),
                "(b t) c h w -> b t c h w",
                t=sliced_degraded_frames.shape[1],
            ).clamp(-1, 1)

        if task == "x8_bicubic":
            mask = (
                face_helper.face_parse(sliced_init_frames.squeeze(0))[0].argmax(1, keepdim=True)
                == 0
            ).float()
            weight = mask * 0.93 + (1 - mask) * 1.0
            model_kwargs["vsrpp_weights"] = rearrange(
                weight, "(b t) c h w -> b t c h w", t=sliced_init_frames.shape[1]
            )
        elif task == "x16_bicubic":
            mask = (
                face_helper.face_parse(sliced_init_frames.squeeze(0))[0].argmax(1, keepdim=True)
                == 0
            ).float()
            weight = mask * 0.98 + (1 - mask) * 1.0
            model_kwargs["vsrpp_weights"] = rearrange(
                weight, "(b t) c h w -> b t c h w", t=sliced_init_frames.shape[1]
            )
        
        _, affine_matrices, _ = face_helper.get_crop_face(
            sliced_init_frames.squeeze(0), only_keep_largest=True, eye_dist_threshold=0.1
        )

        def aux_model(x0, *args, **kwargs):
            aux_face = gan(x0, w=1.0, adain=True)[0]
            return aux_face
        restore_fn = RESTORE_FUNC[task]
        if task == "jpeg":
            restore_fn = partial(restore_fn, jpeg_qf=jpeg_qf)
        task_restore_fn = lambda x: restore_fn(x, sliced_degraded_frames, A_func)

        sample = diffusion.sample(
                model,
                noise,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
                clip_denoised=True,
                restore_fn=task_restore_fn,
                post_fn=None,
                face_restore_helper=face_helper,
                aux_model=aux_model,
                w=w,
                tau=tau,
                affine_matrices=affine_matrices,
                aligned=aligned,
                sample_mode="ddpm",
                rho=rho,
                noise_level=noise_level,
                prev_recon=prev_recon,
                zeta=zeta,
                t_start=t_start,
            )
        sample = rearrange(sample.cpu(), "(b t) c h w -> b t c h w", t=sliced_init_frames.shape[1])
        if prev_recon is not None:
            sample = sample[:, OVERLAP:]
        prev_recon = sample[:, -OVERLAP:].clone()
        sample = ((sample.clamp(-1, 1) + 1) / 2).cpu()
        recon_frames.append(sample)

    recon_frames = torch.cat(recon_frames, 1)
    recon_frames = recon_frames.squeeze(0)
    recon_frames = rearrange((recon_frames * 255).byte().cpu().numpy(), "t c h w -> t h w c")
    os.makedirs(output_path, exist_ok=True)
    for i, frame in enumerate(recon_frames):
        cv2.imwrite(str(output_path / f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))






@app.command()
def x8_bicubic_demo(
    device: Annotated[torch.device, cyclopts.Parameter(converter=lambda t, v: torch.device(v))]=torch.device("cuda"),
):
    main(
        task="x8_bicubic",
        video_path=Path("./data/x8_bicubic"),
        output_path=Path("./output/x8_bicubic"),
        device=device,
        w=0.85,
        rho=0.85,
        noise_level=0.0,
    )

@app.command()
def x16_bicubic_demo(
    device: Annotated[torch.device, cyclopts.Parameter(converter=lambda t, v: torch.device(v))]=torch.device("cuda"),
):
    main(
        task="x16_bicubic",
        video_path=Path("./data/x16_bicubic"),
        output_path=Path("./output/x16_bicubic"),
        device=device,
        w=0.7,
        rho=0.85,
        noise_level=0.0,
    )

@app.command()
def gaussian_demo(
    device: Annotated[torch.device, cyclopts.Parameter(converter=lambda t, v: torch.device(v))]=torch.device("cuda"),
):
    main(
        task="gaussian",
        video_path=Path("./data/gaussian"),
        output_path=Path("./output/gaussian"),
        device=device,
        w=0.75,
        rho=0.25,
        noise_level=2.55,
        zeta=1.0,
    )

@app.command()
def jpeg_demo(
    device: Annotated[torch.device, cyclopts.Parameter(converter=lambda t, v: torch.device(v))]=torch.device("cuda"),
):
    main(
        task="jpeg",
        video_path=Path("./data/jpeg"),
        output_path=Path("./output/jpeg"),
        device=device,
        w=0.5,
        rho=0.5,
        noise_level=12.75,
        zeta=1.0,
        jpeg_qf=60,
    )

if __name__ == "__main__":
    app()