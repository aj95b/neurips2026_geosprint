"""
GeoSPRINT Trajectory Recording
================================
Model-agnostic wrappers to record full denoising trajectories
from pretrained diffusion / flow models.

Supports: EDM (Karras et al.), diffusers (HuggingFace), custom models.
"""

import torch
import numpy as np
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class TrajectoryBundle:
    """A batch of recorded trajectories with metadata."""
    trajectories: List[np.ndarray]  # B trajectories, each (N+1, d)
    timesteps: np.ndarray           # shape (N+1,), the t values
    initial_noise: List[np.ndarray] # the z_T for each trajectory
    final_samples: List[np.ndarray] # the z_0 for each trajectory
    model_name: str


def record_trajectory_diffusers(
    pipeline,
    prompt: str = "",
    num_inference_steps: int = 50,
    batch_size: int = 1,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
) -> TrajectoryBundle:
    """
    Record full denoising trajectories from a HuggingFace diffusers pipeline.

    Works with StableDiffusionPipeline, DDPMPipeline, etc.
    Hooks into the scheduler to capture z_t at every step.

    Parameters
    ----------
    pipeline : diffusers.DiffusionPipeline
        A loaded diffusers pipeline.
    prompt : str
        Text prompt (for text-conditioned models).
    num_inference_steps : int
        Number of denoising steps.
    batch_size : int
        Number of trajectories to record.
    generator : torch.Generator, optional
        For reproducibility.

    Returns
    -------
    TrajectoryBundle
    """
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps.cpu().numpy()

    all_trajectories = []
    all_noise = []
    all_samples = []

    for b in range(batch_size):
        # Get initial noise
        if hasattr(pipeline, 'unet'):
            shape = pipeline.unet.config.sample_size
            if isinstance(shape, int):
                latent_shape = (1, pipeline.unet.config.in_channels, shape, shape)
            else:
                latent_shape = (1, pipeline.unet.config.in_channels, *shape)
        else:
            latent_shape = (1, 3, 64, 64)  # fallback

        latents = torch.randn(latent_shape, generator=generator, device=device,
                              dtype=pipeline.unet.dtype if hasattr(pipeline, 'unet') else torch.float32)

        trajectory = [latents.detach().cpu().flatten().numpy()]
        all_noise.append(trajectory[0].copy())

        # Encode prompt if text-conditioned
        if prompt and hasattr(pipeline, 'encode_prompt'):
            prompt_embeds, neg_embeds, *_ = pipeline.encode_prompt(
                prompt, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        else:
            prompt_embeds = None

        # Step through denoising
        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                if prompt_embeds is not None:
                    noise_pred = pipeline.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
                else:
                    noise_pred = pipeline.unet(latents, t).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            trajectory.append(latents.detach().cpu().flatten().numpy())

        all_trajectories.append(np.stack(trajectory))  # (N+1, d)
        all_samples.append(trajectory[-1].copy())

    return TrajectoryBundle(
        trajectories=all_trajectories,
        timesteps=np.concatenate([[timesteps[0] + 1], timesteps]),  # include t=T
        initial_noise=all_noise,
        final_samples=all_samples,
        model_name=type(pipeline).__name__,
    )


def record_trajectory_edm(
    model,
    sigma_schedule: np.ndarray,
    batch_size: int = 1,
    image_shape: Tuple[int, ...] = (3, 32, 32),
    device: str = "cuda",
    seed: Optional[int] = None,
) -> TrajectoryBundle:
    """
    Record trajectories from EDM-style models (Karras et al., 2022).

    Parameters
    ----------
    model : callable
        The denoiser D(x; σ) that takes (noisy_images, sigma) and returns denoised.
    sigma_schedule : np.ndarray
        Decreasing sequence of sigma values [σ_max, ..., σ_min].
    batch_size : int
    image_shape : tuple
    """
    if seed is not None:
        torch.manual_seed(seed)

    all_trajectories = []
    all_noise = []
    all_samples = []

    for b in range(batch_size):
        # Initial noise at σ_max
        x = torch.randn(1, *image_shape, device=device) * sigma_schedule[0]
        trajectory = [x.detach().cpu().flatten().numpy()]
        all_noise.append(trajectory[0].copy())

        # EDM sampling (Euler-Maruyama)
        for i in range(len(sigma_schedule) - 1):
            sigma = sigma_schedule[i]
            sigma_next = sigma_schedule[i + 1]

            with torch.no_grad():
                denoised = model(x, torch.tensor([sigma], device=device))

            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma)

            trajectory.append(x.detach().cpu().flatten().numpy())

        all_trajectories.append(np.stack(trajectory))
        all_samples.append(trajectory[-1].copy())

    return TrajectoryBundle(
        trajectories=all_trajectories,
        timesteps=sigma_schedule,
        initial_noise=all_noise,
        final_samples=all_samples,
        model_name="EDM",
    )


def record_trajectory_generic(
    denoise_fn: Callable,
    initial_noise: np.ndarray,
    timesteps: np.ndarray,
) -> np.ndarray:
    """
    Generic trajectory recorder for any denoise function.

    Parameters
    ----------
    denoise_fn : callable
        Takes (z_t, t) → z_{t-1}.
    initial_noise : np.ndarray, shape (d,)
    timesteps : np.ndarray, shape (N,)
        Descending timestep values.

    Returns
    -------
    np.ndarray, shape (N+1, d)
        Full trajectory including initial noise.
    """
    trajectory = [initial_noise.copy()]
    z = initial_noise.copy()

    for t in timesteps:
        z = denoise_fn(z, t)
        trajectory.append(z.copy())

    return np.stack(trajectory)
