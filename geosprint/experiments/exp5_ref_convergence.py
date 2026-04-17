"""
Experiment 5: Reference Set Sample Complexity
===============================================
How many reference trajectories B are needed for the universal
schedule to converge?

Hypothesis: Convergence at B ≈ 50-100.

Usage:
    python -m experiments.exp5_ref_convergence \
        --traj_dir results/exp1/trajectories \
        --output figures/exp5_convergence.pdf
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.schedule import extract_universal_schedule


def schedule_similarity(s1, s2):
    """Jaccard similarity between two timestep sets."""
    set1, set2 = set(s1.tolist()), set(s2.tolist())
    if len(set1 | set2) == 0:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def run(args):
    print("Experiment 5: Reference Set Convergence")
    print("=" * 50)

    # Load or generate trajectories
    traj_dir = Path(args.traj_dir)
    if traj_dir.exists():
        all_trajectories = [np.load(f) for f in sorted(traj_dir.glob("traj_*.npy"))]
        timestep_indices = np.load(traj_dir / "timesteps.npy")
    else:
        print("Generating synthetic trajectories...")
        d, N, B_max = 512, 200, 500
        all_trajectories = []
        for b in range(B_max):
            z = np.random.randn(d) * 10
            traj = [z.copy()]
            for t in range(N - 1, -1, -1):
                z = z * (1 - 1/N) + np.random.randn(d) * 0.01 * (1 + 3*np.exp(-t/30))
                traj.append(z.copy())
            all_trajectories.append(np.stack(traj))
        timestep_indices = np.arange(N, -1, -1)

    B_values = [10, 20, 50, 100, 200, 500]
    B_values = [b for b in B_values if b <= len(all_trajectories)]
    nfe_budget = 20
    num_trials = 5

    # Reference schedule from ALL trajectories
    ref_schedule = extract_universal_schedule(
        all_trajectories, timestep_indices, nfe_budget=nfe_budget,
        k=2, target_alpha=1e-3,
    )

    results = {}
    for B in B_values:
        similarities = []
        for trial in range(num_trials):
            # Random subsample of B trajectories
            indices = np.random.choice(len(all_trajectories), B, replace=False)
            subset = [all_trajectories[i] for i in indices]

            sched = extract_universal_schedule(
                subset, timestep_indices, nfe_budget=nfe_budget,
                k=2, target_alpha=1e-3,
            )
            sim = schedule_similarity(sched.timesteps, ref_schedule.timesteps)
            similarities.append(sim)

        results[B] = {
            'jaccard_mean': float(np.mean(similarities)),
            'jaccard_std': float(np.std(similarities)),
        }
        print(f"  B={B:4d}: Jaccard similarity = "
              f"{results[B]['jaccard_mean']:.3f} ± {results[B]['jaccard_std']:.3f}")

    # Save
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp5_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bs = list(results.keys())
        means = [results[b]['jaccard_mean'] for b in bs]
        stds = [results[b]['jaccard_std'] for b in bs]
        ax.errorbar(bs, means, yerr=stds, fmt='o-', capsize=4, color='steelblue')
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
        ax.set_xlabel("Reference batch size B")
        ax.set_ylabel("Jaccard similarity to full schedule")
        ax.set_title("Schedule convergence vs reference set size")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.output}")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="results/exp1/trajectories")
    parser.add_argument("--output", default="figures/exp5_convergence.pdf")
    run(parser.parse_args())
