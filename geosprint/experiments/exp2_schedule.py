"""
Experiment 2: Schedule Analysis & Visualization
=================================================
Visualize the universal schedule w(t) — where does GeoSPRINT
place steps along the denoising trajectory?

Hypothesis: Steps concentrate at early (global structure) and
late (fine detail) denoising, with middle steps pruned.

Reads saved trajectories from Experiment 1.

Usage:
    python -m experiments.exp2_schedule \
        --traj_dir results/exp1/trajectories \
        --output figures/exp2_schedule.pdf
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search
from geosprint.schedule import extract_universal_schedule


def run(args):
    print("Experiment 2: Schedule Analysis")
    print("=" * 50)

    # Load saved trajectories
    traj_dir = Path(args.traj_dir)
    if traj_dir.exists():
        trajectories = [np.load(f) for f in sorted(traj_dir.glob("traj_*.npy"))]
        timestep_indices = np.load(traj_dir / "timesteps.npy")
        print(f"Loaded {len(trajectories)} trajectories from {traj_dir}")
    else:
        print(f"No saved trajectories at {traj_dir}, generating synthetic...")
        d, N, B = 3072, 1000, 100
        trajectories = []
        for b in range(B):
            z = np.random.randn(d) * 80
            traj = [z.copy()]
            for t in range(N - 1, -1, -1):
                noise_scale = 0.01 * (1 + 5 * np.exp(-t / 50))
                z = z * (1 - 1/N) + np.random.randn(d) * noise_scale
                traj.append(z.copy())
            trajectories.append(np.stack(traj))
        timestep_indices = np.arange(N, -1, -1)

    N = len(timestep_indices)

    # Extract schedules at multiple budgets
    nfe_budgets = [10, 20, 30, 50]
    schedules = {}
    for budget in nfe_budgets:
        sched = extract_universal_schedule(
            trajectories, timestep_indices, nfe_budget=budget,
            k=2, target_alpha=1e-3,
        )
        schedules[budget] = sched

    # Analyze retention frequency
    w = schedules[nfe_budgets[-1]].retention_freq  # highest-budget w(t)

    # Identify high-retention zones
    top_20pct = np.percentile(w, 80)
    high_retention = np.where(w >= top_20pct)[0]
    early = high_retention[high_retention < N * 0.2]
    late = high_retention[high_retention > N * 0.8]
    middle = high_retention[(high_retention >= N * 0.2) & (high_retention <= N * 0.8)]

    print(f"\nRetention frequency analysis (NFE={nfe_budgets[-1]}):")
    print(f"  High-retention steps in early 20%: {len(early)}")
    print(f"  High-retention steps in middle 60%: {len(middle)}")
    print(f"  High-retention steps in late 20%: {len(late)}")

    # Save analysis
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = {
        'retention_freq': w.tolist(),
        'high_early': len(early),
        'high_middle': len(middle),
        'high_late': len(late),
    }
    with open(output_dir / "exp2_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs

        fig = plt.figure(figsize=(10, 6))
        grid = gs.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # (a) Retention frequency heatmap
        ax1 = fig.add_subplot(grid[0, :])
        t_normalized = np.linspace(1, 0, N)
        ax1.bar(t_normalized, w, width=1.0/N, color='steelblue', alpha=0.7)
        ax1.set_xlabel("Normalized time (t/T → 0)")
        ax1.set_ylabel("Retention frequency w(t)")
        ax1.set_title("(a) Where GeoSPRINT keeps steps")
        ax1.set_xlim(0, 1)

        # (b) Schedule comparison across NFE budgets
        ax2 = fig.add_subplot(grid[1, 0])
        for i, budget in enumerate(nfe_budgets):
            sched = schedules[budget]
            y = np.zeros(N)
            for ts in sched.timesteps:
                idx = np.searchsorted(-timestep_indices, -ts)
                if idx < N:
                    y[idx] = 1
            ax2.scatter(t_normalized[y > 0], [budget]*int(y.sum()),
                       s=3, alpha=0.8)
        ax2.set_xlabel("Normalized time")
        ax2.set_ylabel("NFE budget")
        ax2.set_title("(b) Selected steps by budget")

        # (c) Cumulative retention
        ax3 = fig.add_subplot(grid[1, 1])
        cumw = np.cumsum(w) / np.sum(w)
        ax3.plot(t_normalized, cumw, 'k-', linewidth=1.5)
        ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        median_idx = np.searchsorted(cumw, 0.5)
        ax3.axvline(t_normalized[median_idx], color='coral', linestyle='--', alpha=0.7)
        ax3.set_xlabel("Normalized time")
        ax3.set_ylabel("Cumulative retention")
        ax3.set_title("(c) Cumulative step distribution")

        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.output}")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="results/exp1/trajectories")
    parser.add_argument("--output", default="figures/exp2_schedule.pdf")
    run(parser.parse_args())
