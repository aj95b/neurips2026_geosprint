"""
Experiment 1: FID vs NFE Pareto Frontiers
==========================================
The central experiment of the paper.

Sweeps NFE budgets 5-50, compares GeoSPRINT (universal + adaptive)
against DDIM, DPM-Solver++, and other baselines.

Usage:
    python -m experiments.exp1_pareto \
        --model edm_cifar10 \
        --ref_batch 100 \
        --num_samples 50000 \
        --output figures/exp1_cifar10.pdf
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search
from geosprint.schedule import sweep_nfe_budgets
from geosprint.evaluate import EvalResult, format_results_table


def run_cifar10_experiment(args):
    """
    Full Experiment 1 pipeline for CIFAR-10 with EDM.

    Steps:
    1. Load pretrained EDM model
    2. Generate B reference trajectories
    3. Extract universal schedules at NFE = {5, 10, 15, 20, 30, 50}
    4. Generate 50K samples with each schedule
    5. Compute FID for each
    6. Do the same for DDIM and DPM-Solver++ baselines
    7. Plot Pareto frontier
    """
    print("=" * 60)
    print("Experiment 1: FID vs NFE Pareto Frontier")
    print(f"Model: {args.model}")
    print(f"Reference batch: {args.ref_batch}")
    print(f"Num samples: {args.num_samples}")
    print("=" * 60)

    nfe_budgets = [5, 8, 10, 15, 20, 30, 50]
    results = []

    # ---- Phase 1: Record reference trajectories ----
    print("\n[Phase 1] Recording reference trajectories...")
    t0 = time.time()

    # PLACEHOLDER: Replace with actual model loading
    # For CIFAR-10 EDM:
    #   from geosprint.trajectories import record_trajectory_edm
    #   model = load_edm_cifar10()
    #   sigma_schedule = get_edm_sigma_schedule(N=1000)
    #   bundle = record_trajectory_edm(model, sigma_schedule,
    #                                   batch_size=args.ref_batch)
    #   trajectories = bundle.trajectories
    #   timestep_indices = bundle.timesteps

    # --- Demo with synthetic trajectories ---
    print("  (Using synthetic trajectories for demo)")
    d = 3072  # CIFAR-10: 3×32×32
    N = 1000
    trajectories = []
    for b in range(args.ref_batch):
        # Simulate: trajectory is mostly linear in high-noise,
        # curves sharply in low-noise
        z = np.random.randn(d) * 80  # z_T
        traj = [z.copy()]
        for t in range(N - 1, -1, -1):
            # Linear drift + curvature that increases as t→0
            noise_scale = 0.01 * (1 + 5 * np.exp(-t / 50))
            z = z * (1 - 1/N) + np.random.randn(d) * noise_scale
            traj.append(z.copy())
        trajectories.append(np.stack(traj))

    timestep_indices = np.arange(N, -1, -1)
    print(f"  Recorded {len(trajectories)} trajectories in {time.time()-t0:.1f}s")

    # ---- Phase 2: Extract GeoSPRINT schedules ----
    print("\n[Phase 2] Extracting GeoSPRINT schedules...")
    t0 = time.time()

    schedules = sweep_nfe_budgets(
        trajectories=trajectories,
        timestep_indices=timestep_indices,
        nfe_budgets=nfe_budgets,
        k=2,
        target_alpha=1e-3,
    )

    for nfe, sched in schedules.items():
        print(f"  NFE={nfe:3d}: {sched.nfe} steps, "
              f"α_traj={sched.mean_alpha:.2e} ± {sched.std_alpha:.2e}")
        results.append(EvalResult(
            method="GeoSPRINT-univ",
            nfe=sched.nfe,
            mean_alpha=sched.mean_alpha,
        ))

    print(f"  Schedule extraction took {time.time()-t0:.1f}s")

    # ---- Phase 3: Generate samples with each schedule ----
    print("\n[Phase 3] Generating samples (placeholder)...")
    # PLACEHOLDER: For each schedule, generate 50K samples and compute FID
    #
    # for nfe, sched in schedules.items():
    #     samples = sample_with_schedule(pipeline, sched,
    #                                     batch_size=args.num_samples)
    #     fid = compute_fid(samples, ref_stats)
    #     results[nfe].fid = fid

    # ---- Phase 4: Baselines ----
    print("\n[Phase 4] Running baselines (placeholder)...")
    # PLACEHOLDER:
    # for nfe in nfe_budgets:
    #     # DDIM
    #     ddim_samples = ddim_sample(pipeline, nfe)
    #     ddim_fid = compute_fid(ddim_samples, ref_stats)
    #     results.append(EvalResult(method="DDIM", nfe=nfe, fid=ddim_fid))
    #
    #     # DPM-Solver++
    #     dpm_samples = dpm_solver_sample(pipeline, nfe, order=3)
    #     dpm_fid = compute_fid(dpm_samples, ref_stats)
    #     results.append(EvalResult(method="DPM-Solver++", nfe=nfe, fid=dpm_fid))

    # ---- Phase 5: Output ----
    print("\n[Phase 5] Generating outputs...")

    # LaTeX table
    table = format_results_table(results, "CIFAR-10: FID vs NFE")
    print("\n" + table)

    # Save results
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "exp1_results.json", 'w') as f:
        json.dump([{
            'method': r.method,
            'nfe': r.nfe,
            'fid': r.fid,
            'alpha': r.mean_alpha,
        } for r in results], f, indent=2)

    print(f"\nResults saved to {output_dir / 'exp1_results.json'}")

    # ---- Phase 6: Plot ----
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for method in set(r.method for r in results):
            mr = [r for r in results if r.method == method]
            mr.sort(key=lambda x: x.nfe)
            nfes = [r.nfe for r in mr]
            fids = [r.fid for r in mr]
            ax.plot(nfes, fids, 'o-', label=method, markersize=5)

        ax.set_xlabel("NFE")
        ax.set_ylabel("FID ↓")
        ax.set_title("CIFAR-10: FID vs NFE Pareto Frontier")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {args.output}")

    except ImportError:
        print("matplotlib not available — skipping plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="edm_cifar10")
    parser.add_argument("--ref_batch", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--output", default="figures/exp1_cifar10.pdf")
    args = parser.parse_args()

    run_cifar10_experiment(args)
