"""
Experiment 3: α_traj as Rectification Diagnostic
==================================================
For rectified flow models (0-, 1-, 2-rectified), show that
α_traj decreases with each round of reflow.

Hypothesis: α_traj correlates with few-step FID, providing
a training-free metric for flow-matching quality.

Usage:
    python -m experiments.exp3_rectification \
        --output figures/exp3_rectification.pdf
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search, prune_trajectory


def simulate_rectified_trajectories(N=100, d=512, B=50, rectification_rounds=0):
    """
    Simulate trajectories with controllable straightness.
    More rectification rounds → straighter paths → lower α_traj.
    
    Replace this with actual rectified flow models:
        - 0-rect: standard flow matching
        - 1-rect: one round of reflow
        - 2-rect: two rounds of reflow
    """
    trajectories = []
    # Curvature decreases with rectification
    curvature = 1.0 / (1.0 + rectification_rounds * 3.0)
    
    for b in range(B):
        start = np.random.randn(d)
        end = np.random.randn(d) * 0.1
        traj = np.zeros((N, d))
        for i in range(N):
            t = i / (N - 1)
            # Linear interpolation + curvature perturbation
            traj[i] = start * (1 - t) + end * t + \
                      curvature * np.sin(t * np.pi) * np.random.randn(d) * 0.1
        trajectories.append(traj)
    return trajectories


def run(args):
    print("Experiment 3: α_traj as Rectification Diagnostic")
    print("=" * 50)

    rect_rounds = [0, 1, 2]
    N, d, B = 100, 512, 50
    results = {}

    for rr in rect_rounds:
        print(f"\n  Rectification rounds: {rr}")
        
        # PLACEHOLDER: Replace with actual rectified flow models
        # from rectified_flow import load_model
        # model = load_model(f"rectified_flow_{rr}rect.pt")
        # trajectories = [record_trajectory(model) for _ in range(B)]
        
        trajectories = simulate_rectified_trajectories(N, d, B, rr)
        
        alphas = []
        reductions = []
        for traj in trajectories:
            tau, result = threshold_search(traj, k=2, target_alpha=1e-3)
            alphas.append(result.projection_score)
            reductions.append(result.reduction_pct)
        
        results[rr] = {
            'alpha_mean': float(np.mean(alphas)),
            'alpha_std': float(np.std(alphas)),
            'reduction_mean': float(np.mean(reductions)),
            'reduction_std': float(np.std(reductions)),
        }
        print(f"    α_traj = {results[rr]['alpha_mean']:.4f} ± {results[rr]['alpha_std']:.4f}")
        print(f"    Reduction = {results[rr]['reduction_mean']:.1f}% ± {results[rr]['reduction_std']:.1f}%")

    # Verify monotonic decrease
    alphas_seq = [results[rr]['alpha_mean'] for rr in rect_rounds]
    monotonic = all(alphas_seq[i] >= alphas_seq[i+1] for i in range(len(alphas_seq)-1))
    print(f"\n  Monotonic decrease in α_traj: {'✓ YES' if monotonic else '✗ NO'}")

    # Save
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp3_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

        means = [results[rr]['alpha_mean'] for rr in rect_rounds]
        stds = [results[rr]['alpha_std'] for rr in rect_rounds]
        ax1.bar(rect_rounds, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax1.set_xlabel("Rectification rounds")
        ax1.set_ylabel("α_traj (↓ = straighter)")
        ax1.set_title("(a) Trajectory projection score")
        ax1.set_xticks(rect_rounds)

        reds = [results[rr]['reduction_mean'] for rr in rect_rounds]
        red_stds = [results[rr]['reduction_std'] for rr in rect_rounds]
        ax2.bar(rect_rounds, reds, yerr=red_stds, capsize=5, color='coral', alpha=0.8)
        ax2.set_xlabel("Rectification rounds")
        ax2.set_ylabel("Steps pruned (%)")
        ax2.set_title("(b) GeoSPRINT reduction rate")
        ax2.set_xticks(rect_rounds)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.output}")
    except ImportError:
        print("matplotlib not available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/exp3_rectification.pdf")
    run(parser.parse_args())
