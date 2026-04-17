"""
Experiment 6: Per-Sample NFE Distribution
==========================================
Under GeoSPRINT-adaptive, different samples get different numbers
of steps. Analyze the distribution and correlate with sample complexity.

Hypothesis: Complex/high-frequency samples get more steps;
simple/low-frequency get fewer.

Usage:
    python -m experiments.exp6_adaptive \
        --traj_dir results/exp1/trajectories \
        --output figures/exp6_adaptive.pdf
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search
from geosprint.evaluate import nfe_statistics


def run(args):
    print("Experiment 6: Per-Sample NFE Distribution")
    print("=" * 50)

    # Load or generate trajectories
    traj_dir = Path(args.traj_dir)
    if traj_dir.exists():
        trajectories = [np.load(f) for f in sorted(traj_dir.glob("traj_*.npy"))]
    else:
        print("Generating synthetic trajectories with variable complexity...")
        d, N = 512, 200
        trajectories = []
        complexities = []
        for b in range(500):
            # Variable complexity: some samples are "easy" (straight), some "hard" (curved)
            complexity = np.random.exponential(1.0)
            complexities.append(complexity)
            z = np.random.randn(d) * 10
            traj = [z.copy()]
            for t in range(N - 1, -1, -1):
                curvature = 0.01 * complexity * (1 + 5 * np.exp(-t / 30))
                z = z * (1 - 1/N) + np.random.randn(d) * curvature
                traj.append(z.copy())
            trajectories.append(np.stack(traj))

    # Run adaptive pruning on each sample
    print(f"\n  Analyzing {len(trajectories)} trajectories...")
    per_sample_nfe = []
    per_sample_alpha = []

    for i, traj in enumerate(trajectories):
        tau, result = threshold_search(traj, k=2, target_alpha=1e-3)
        adaptive_nfe = len(result.retained_indices)
        per_sample_nfe.append(adaptive_nfe)
        per_sample_alpha.append(result.projection_score)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(trajectories)}")

    per_sample_nfe = np.array(per_sample_nfe)
    per_sample_alpha = np.array(per_sample_alpha)

    # Statistics
    stats = nfe_statistics(per_sample_nfe)
    print(f"\n  NFE distribution:")
    for k, v in stats.items():
        print(f"    {k}: {v:.1f}")

    # Save
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "exp6_per_sample_nfe.npy", per_sample_nfe)
    np.save(output_dir / "exp6_per_sample_alpha.npy", per_sample_alpha)
    with open(output_dir / "exp6_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

        # (a) NFE histogram
        axes[0].hist(per_sample_nfe, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
        axes[0].axvline(stats['nfe_mean'], color='coral', linestyle='--', label=f"mean={stats['nfe_mean']:.0f}")
        axes[0].set_xlabel("Adaptive NFE per sample")
        axes[0].set_ylabel("Count")
        axes[0].set_title("(a) NFE distribution")
        axes[0].legend(fontsize=8)

        # (b) NFE vs α_traj
        axes[1].scatter(per_sample_alpha, per_sample_nfe, s=3, alpha=0.3, color='steelblue')
        axes[1].set_xlabel("α_traj (trajectory non-straightness)")
        axes[1].set_ylabel("Adaptive NFE")
        axes[1].set_title("(b) NFE vs trajectory complexity")

        # (c) Sorted NFE (shows spread)
        sorted_nfe = np.sort(per_sample_nfe)
        axes[2].plot(np.arange(len(sorted_nfe)) / len(sorted_nfe) * 100,
                     sorted_nfe, color='steelblue', linewidth=1.5)
        axes[2].fill_between(
            np.arange(len(sorted_nfe)) / len(sorted_nfe) * 100,
            sorted_nfe, alpha=0.2, color='steelblue')
        axes[2].set_xlabel("Percentile of samples")
        axes[2].set_ylabel("NFE")
        axes[2].set_title("(c) NFE cumulative distribution")

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.output}")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="results/exp1/trajectories")
    parser.add_argument("--output", default="figures/exp6_adaptive.pdf")
    run(parser.parse_args())
