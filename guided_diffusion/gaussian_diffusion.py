"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
from einops import rearrange
import numpy as np
import torch as th
from guided_diffusion.facelib.utils.face_restoration_helper import FaceRestoreHelper


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "face_blur":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "face_bicubic":
        return np.linspace(1e-6, 1e-2, 2000, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, self.alphas_cumprod))
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = np.append(
            0.0, np.sqrt(1.0 - self.alphas_cumprod[:-1])
        )
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef3 = self.posterior_mean_coef1 + (
            self.posterior_mean_coef2 * np.sqrt(self.alphas_cumprod)
        )
        self.posterior_mean_coef4 = self.posterior_mean_coef2 * np.sqrt(
            1 - self.alphas_cumprod
        )
        # if lambda_ == -1:
        #     self.rhos = np.ones_like(betas)
        # else:
        #     self.rhos = lambda_ * (
        #         (12.75 / 255) ** 2
        #         / (self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod) ** 2
        #     )
        #     self.rhos[self.rhos >= 1] = 0.991
        #     self.rhos[self.rhos <= 1e-1] = 1e-1
        #     self.rhos = 1 - self.rhos
        # self.rhos = 1 - np.logspace(
        #     np.log10(0.991), np.log10(0.01), self.num_timesteps, dtype=np.float64
        # )
        # self.rhos = (self.rhos - self.rhos.min()) / (self.rhos.max() - self.rhos.min())

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param restore_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["sqrt_recip_alphas_cumprod"] = self.sqrt_recip_alphas_cumprod
        model_kwargs["sqrt_recipm1_alphas_cumprod"] = self.sqrt_recipm1_alphas_cumprod
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            if model_output.shape[1] == 6:
                model_output = model_output[:, :3, ...]
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if clip_denoised:
                x = x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:
            raise NotImplementedError(self.model_mean_type)
        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def sample(
        self,
        model,
        noise,
        model_kwargs,
        restore_fn,
        face_restore_helper,
        aux_model,
        post_fn,
        clip_denoised=True,
        sample_mode="ddpm",
        device=None,
        progress=False,
        w=0.5,
        tau=None,
        aligned=False,
        affine_matrices=None,
        rho=0.35,
        noise_level=None,
        prev_recon=None,
        zeta=-1,
        t_start=-1,
    ):
        if tau is None:
            tau = 0
        if sample_mode == "ddpm":
            pred = self.p_sample_loop(
                model=model,
                shape=noise.shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                progress=progress,
                device=device,
                restore_fn=restore_fn,
                face_restore_helper=face_restore_helper,
                aux_model=aux_model,
                post_fn=post_fn,
                w=w,
                tau=tau,
                aligned=aligned,
                affine_matrices=affine_matrices,
                rho=rho,
                noise_level=noise_level,
                prev_recon=prev_recon,
                zeta=zeta,
                t_start=t_start,
                # **sample_kwargs,
            )
        return pred

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        model_kwargs=None,
        restore_fn=None,
        affine_matrices=None,
        face_restore_helper: FaceRestoreHelper = None,
        aux_model=None,
        w=0.5,
        start_timestep=None,
        tau=None,
        aligned=False,
        rho=0.35,
        prev_recon=None,
        gamma=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if restore_fn is not None:
            out["pred_xstart"] = out["pred_xstart"] - gamma * restore_fn(
                out["pred_xstart"]
            )
            if clip_denoised:
                out["pred_xstart"] = out["pred_xstart"].clamp(-1, 1)
        if (
            aux_model is not None
            and all(t <= start_timestep)
            and all(t >= tau)
        ):
            if not aligned:
                aux_face = face_restore_helper.get_crop_face_from_affine_matrices(
                    out["pred_xstart"], affine_matrices
                )
                aux_xt = face_restore_helper.get_crop_face_from_affine_matrices(
                    x, affine_matrices
                )
            else:
                aux_face = out["pred_xstart"]
                aux_xt = x
            aux_face = aux_model(aux_face, t, aux_xt)
            if not aligned:
                inv_face, inv_mask = face_restore_helper.inverse_faces(
                    aux_face, affine_matrices
                )
                x_with_face = out["pred_xstart"] * (1 - inv_mask) + inv_face * inv_mask
            else:
                x_with_face = aux_face
            if clip_denoised:
                x_with_face = x_with_face.clamp(-1, 1)
            out["pred_xstart"] = w * out["pred_xstart"] + (1 - w) * x_with_face
        if prev_recon is not None:
            out["pred_xstart"] = rearrange(
                out["pred_xstart"],
                "(b t) c h w -> b t c h w",
                t=model_kwargs["num_frames"],
            )
            out["pred_xstart"][:, : prev_recon.shape[1]].copy_(prev_recon)
            out["pred_xstart"] = rearrange(
                out["pred_xstart"], "b t c h w -> (b t) c h w"
            )
        eps = self._predict_eps_from_xstart(x_t=x, t=t, pred_xstart=out["pred_xstart"])
        co_noise = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod_prev, t, x.shape
        )
        sample = _extract_into_tensor(self.sqrt_alphas_cumprod_prev, t, x.shape) * out[
            "pred_xstart"
        ] + nonzero_mask * (
            np.sqrt(1 - rho) * co_noise * eps + np.sqrt(rho) * co_noise * th.randn_like(x)
        )

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        affine_matrices=None,
        restore_fn=None,
        face_restore_helper=None,
        aux_model=None,
        post_fn=None,
        w=0.5,
        tau=None,
        aligned=False,
        rho=0.35,
        noise_level=None,
        prev_recon=None,
        zeta=-1,
        t_start=-1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            restore_fn=restore_fn,
            affine_matrices=affine_matrices,
            face_restore_helper=face_restore_helper,
            aux_model=aux_model,
            w=w,
            tau=tau,
            aligned=aligned,
            rho=rho,
            noise_level=noise_level,
            prev_recon=prev_recon,
            zeta=zeta,
            t_start=t_start,
        ):
            if post_fn is not None:
                post_fn(sample)
            final = sample

        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        affine_matrices=None,
        face_restore_helper=None,
        aux_model=None,
        restore_fn=None,
        w=0.5,
        tau=None,
        aligned=False,
        rho=0.35,
        noise_level=None,
        prev_recon=None,
        zeta=-1,
        t_start=-1,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))
        if t_start != -1:
            if t_start < 0 or t_start >= self.num_timesteps:
                raise ValueError("t_start must be in [0, num_timesteps)")
            indices = indices[: t_start + 1]
        indices = indices[::-1]
        if aux_model is not None:
            start_timestep = indices[0]
            if start_timestep - tau > 0:
                ws = np.linspace(0, 1, start_timestep - tau + 1)
                ws = 1.0 * np.exp(-ws * 1)
                ws = (ws - ws.min()) / (ws.max() - ws.min()) * (1 - w)
                ws = 1 - ws
                ws = np.append(
                    ws, np.ones(self.num_timesteps - start_timestep - 1)
                )
                ws = np.concatenate([np.ones(tau), ws])
            else:
                ws = np.ones(self.num_timesteps) * w
        else:
            ws = np.ones(self.num_timesteps)

        if zeta == -1:
            gammas = np.ones_like(self.betas)
        else:
            gammas = zeta * (
                noise_level**2
                / (self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod) ** 2
            )
            gammas[gammas >= 1] = 0.991
            gammas[gammas <= 1e-1] = 1e-6
            gammas = 1 - gammas

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                w = _extract_into_tensor(ws, t, img.shape)
                gamma = _extract_into_tensor(gammas, t, img.shape)
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    restore_fn=restore_fn,
                    affine_matrices=affine_matrices,
                    face_restore_helper=face_restore_helper,
                    aux_model=aux_model,
                    w=w,
                    start_timestep=start_timestep,
                    tau=tau,
                    aligned=aligned,
                    rho=rho,
                    prev_recon=prev_recon,
                    gamma=gamma,
                )
                img = out["sample"]
                out["t"] = t
                yield out


def _extract_into_tensor(arr, timesteps, broadcast_shape, dtype=th.float32):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
