"""
GeoSPRINT Sampler
==================
Apply extracted schedules to generate samples from pretrained models.
Supports universal, adaptive, and streaming modes.
"""

import torch
import numpy as np
from typing import Optional, List
from .core import hyperplanarity_residual
from .schedule import UniversalSchedule


def sample_with_schedule(
    pipeline,
    schedule: UniversalSchedule,
    prompt: str = "",
    batch_size: int = 1,
    generator: Optional[torch.Generator] = None,
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    Generate samples using a GeoSPRINT universal schedule.

    Replaces the pipeline's default timestep schedule with the
    GeoSPRINT-selected timesteps.

    Parameters
    ----------
    pipeline : diffusers.DiffusionPipeline
    schedule : UniversalSchedule
    prompt : str
    batch_size : int

    Returns
    -------
    list of np.ndarray
        Generated samples.
    """
    scheduler = pipeline.scheduler

    # Override scheduler timesteps with GeoSPRINT schedule
    custom_timesteps = torch.tensor(schedule.timesteps, device=device, dtype=torch.long)

    samples = []
    for b in range(batch_size):
        if hasattr(pipeline, 'unet'):
            shape = pipeline.unet.config.sample_size
            if isinstance(shape, int):
                latent_shape = (1, pipeline.unet.config.in_channels, shape, shape)
            else:
                latent_shape = (1, pipeline.unet.config.in_channels, *shape)
        else:
            latent_shape = (1, 3, 64, 64)

        latents = torch.randn(latent_shape, generator=generator, device=device,
                              dtype=pipeline.unet.dtype if hasattr(pipeline, 'unet') else torch.float32)

        # Encode prompt
        if prompt and hasattr(pipeline, 'encode_prompt'):
            prompt_embeds, neg_embeds, *_ = pipeline.encode_prompt(
                prompt, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        else:
            prompt_embeds = None

        # Step through using only GeoSPRINT-selected timesteps
        for t in custom_timesteps:
            with torch.no_grad():
                if prompt_embeds is not None:
                    noise_pred = pipeline.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
                else:
                    noise_pred = pipeline.unet(latents, t).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        samples.append(latents.detach().cpu().numpy())

    return samples


def sample_streaming_adaptive(
    pipeline,
    full_timesteps: np.ndarray,
    k: int = 2,
    threshold: float = 1e-3,
    prompt: str = "",
    device: str = "cuda",
    generator: Optional[torch.Generator] = None,
) -> dict:
    """
    Sample-adaptive streaming inference (Algorithm 3).

    Runs the full model at each step but skips steps identified as
    geometrically redundant in real-time.

    NOTE: This still evaluates the model at every step (for the candidate),
    but the trajectory analysis reveals which steps *could* be skipped
    in a cached/precomputed setting. The NFE count reflects actual
    model calls (= full schedule), but the retained_steps shows what
    an oracle schedule would keep.

    For true NFE savings, use the universal schedule mode instead.
    This mode is primarily for:
    - Analyzing per-sample step distributions (Experiment 6)
    - Computing sample-specific projection scores

    Returns
    -------
    dict with keys: sample, retained_steps, nfe, projection_score
    """
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(len(full_timesteps), device=device)

    if hasattr(pipeline, 'unet'):
        shape = pipeline.unet.config.sample_size
        if isinstance(shape, int):
            latent_shape = (1, pipeline.unet.config.in_channels, shape, shape)
        else:
            latent_shape = (1, pipeline.unet.config.in_channels, *shape)
    else:
        latent_shape = (1, 3, 64, 64)

    latents = torch.randn(latent_shape, generator=generator, device=device,
                          dtype=pipeline.unet.dtype if hasattr(pipeline, 'unet') else torch.float32)

    # Encode prompt
    if prompt and hasattr(pipeline, 'encode_prompt'):
        prompt_embeds, *_ = pipeline.encode_prompt(
            prompt, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    else:
        prompt_embeds = None

    retained_window = []  # last k retained latent vectors
    retained_steps = []
    nfe = 0

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            if prompt_embeds is not None:
                noise_pred = pipeline.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
            else:
                noise_pred = pipeline.unet(latents, t).sample
        nfe += 1

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        z_flat = latents.detach().cpu().flatten().numpy()

        if len(retained_window) < k:
            retained_window.append(z_flat)
            retained_steps.append(i)
        else:
            window = np.stack(retained_window[-k:])
            r = hyperplanarity_residual(window, z_flat)

            if r >= threshold:
                retained_window.append(z_flat)
                retained_steps.append(i)
            # else: skip (point is redundant)

    # Always retain last step
    if len(retained_steps) == 0 or retained_steps[-1] != len(full_timesteps) - 1:
        retained_steps.append(len(full_timesteps) - 1)

    return {
        'sample': latents.detach().cpu().numpy(),
        'retained_steps': np.array(retained_steps),
        'nfe': nfe,
        'adaptive_nfe': len(retained_steps),
    }
