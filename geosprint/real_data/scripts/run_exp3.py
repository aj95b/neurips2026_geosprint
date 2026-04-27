"""
Experiment 3: α_traj Across Scheduler Types
=============================================
DDPM (curved) vs DDIM (straighter) vs DPM-Solver++ (straightest)

Usage: python scripts/run_exp3.py --device cuda
Time:  ~2 hours
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search


def record_trajectories_for_scheduler(scheduler_name, num_trajs, num_steps, device):
    """
    Record trajectories with a specific scheduler.
    Creates a FRESH scheduler for every trajectory to reset internal state.
    """
    from diffusers import (
        UNet2DModel, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler,
    )

    model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
    model.eval()

    def make_scheduler():
        if scheduler_name == "DDPM":
            return DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
        elif scheduler_name == "DDIM":
            return DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        elif scheduler_name == "DPM-Solver++":
            # Construct manually. Two critical fixes:
            #   1. from_pretrained ignores solver_order kwarg (loads 3 from JSON)
            #   2. lower_order_final=True prevents IndexError on last step
            return DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                algorithm_type="dpmsolver++",
                solver_order=2,
                lower_order_final=True,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    trajectories = []
    for b in tqdm(range(num_trajs), desc=f"  {scheduler_name}"):
        # CRITICAL: fresh scheduler each trajectory. DPM-Solver++ keeps
        # an internal step_index that accumulates across step() calls.
        # Without reset, trajectory 2 starts with step_index=num_steps
        # and immediately overflows sigmas[step_index+1].
        scheduler = make_scheduler()
        scheduler.set_timesteps(num_steps)

        generator = torch.Generator(device=device).manual_seed(b)
        latents = torch.randn(1, 3, 32, 32, generator=generator, device=device)
        traj = [latents.detach().cpu().numpy().flatten()]

        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(latents, t).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            traj.append(latents.detach().cpu().numpy().flatten())

        trajectories.append(np.stack(traj))

    del model
    torch.cuda.empty_cache()
    return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_trajs", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=100)
    args = parser.parse_args()

    print("Experiment 3: α_traj Across Scheduler Types")
    print("=" * 50)

    scheduler_names = ["DDPM", "DDIM", "DPM-Solver++"]
    results = {}

    for name in scheduler_names:
        print(f"\nRecording: {name}")
        trajectories = record_trajectories_for_scheduler(
            name, args.num_trajs, args.num_steps, args.device
        )

        # Normalize before analysis (same as exp1)
        alphas = []
        reductions = []
        for traj in trajectories:
            mu = traj.mean(axis=0, keepdims=True)
            std = traj.std(axis=0, keepdims=True) + 1e-8
            traj_norm = (traj - mu) / std
            tau, result = threshold_search(traj_norm, k=2, target_alpha=1e-3)
            alphas.append(result.projection_score)
            reductions.append(result.reduction_pct)

        results[name] = {
            'alpha_mean': float(np.mean(alphas)),
            'alpha_std': float(np.std(alphas)),
            'reduction_mean': float(np.mean(reductions)),
            'reduction_std': float(np.std(reductions)),
        }
        print(f"  α_traj = {results[name]['alpha_mean']:.4f} ± {results[name]['alpha_std']:.4f}")
        print(f"  Pruned = {results[name]['reduction_mean']:.1f}% ± {results[name]['reduction_std']:.1f}%")

    alphas_seq = [results[n]['alpha_mean'] for n in scheduler_names]
    monotonic = all(alphas_seq[i] >= alphas_seq[i+1] for i in range(len(alphas_seq)-1))
    print(f"\nMonotonic decrease: {'YES ✓' if monotonic else 'NO ✗'}")

    out_dir = Path("results/exp3")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exp3_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    short = ["DDPM", "DDIM", "DPM++"]
    colors = ['#534AB7', '#1D9E75', '#D85A30']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    m = [results[n]['alpha_mean'] for n in scheduler_names]
    s = [results[n]['alpha_std'] for n in scheduler_names]
    ax1.bar(short, m, yerr=s, capsize=5, color=colors, alpha=0.85)
    ax1.set_ylabel("α_traj (↓ = straighter)")
    ax1.set_title("(a) Trajectory projection score")

    r = [results[n]['reduction_mean'] for n in scheduler_names]
    rs = [results[n]['reduction_std'] for n in scheduler_names]
    ax2.bar(short, r, yerr=rs, capsize=5, color=colors, alpha=0.85)
    ax2.set_ylabel("Steps pruned (%)")
    ax2.set_title("(b) GeoSPRINT reduction rate")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/exp3_rectification.pdf", dpi=200, bbox_inches='tight')
    print(f"\nSaved to figures/exp3_rectification.pdf")


if __name__ == "__main__":
    main()
