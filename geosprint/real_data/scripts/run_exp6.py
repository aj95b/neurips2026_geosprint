"""
Experiment 6 (Improved): Per-Sample NFE Distribution
======================================================
Sweeps multiple α_traj targets to show how sample complexity
varies at different fidelity levels. Also compares normalized
vs raw trajectories to show the effect of normalization.

Usage: python scripts/run_exp6.py
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search
from geosprint.evaluate import nfe_statistics


def main():
    print("Experiment 6: Per-Sample NFE Distribution")
    print("=" * 50)

    traj_dir = Path("results/exp1/trajectories")
    if not traj_dir.exists():
        print("ERROR: Run run_exp1.py first!")
        return

    files = sorted(traj_dir.glob("traj_*.npy"))
    trajectories = [np.load(f) for f in files]
    print(f"Loaded {len(trajectories)} trajectories")

    # Normalize
    normalized = []
    for traj in trajectories:
        mu = traj.mean(0, keepdims=True)
        s = traj.std(0, keepdims=True) + 1e-8
        normalized.append((traj - mu) / s)

    # Sweep multiple alpha targets
    alpha_targets = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    all_sweep = {}
    print(f"\n{'α_target':>10s} {'mean':>6s} {'std':>6s} {'min':>5s} {'max':>5s} {'ratio':>6s}")
    print("-" * 45)

    for alpha in alpha_targets:
        nfes = []
        alphas = []
        for traj in normalized:
            tau, result = threshold_search(traj, k=2, target_alpha=alpha)
            nfes.append(len(result.retained_indices))
            alphas.append(result.projection_score)

        nfes = np.array(nfes)
        alphas_arr = np.array(alphas)
        ratio = nfes.max() / max(nfes.min(), 1)

        all_sweep[f"{alpha:.0e}"] = {
            'nfes': nfes.tolist(),
            'alphas': alphas_arr.tolist(),
            'mean': float(nfes.mean()),
            'std': float(nfes.std()),
            'min': int(nfes.min()),
            'max': int(nfes.max()),
            'ratio': float(ratio),
        }

        print(f"{alpha:>10.0e} {nfes.mean():>6.1f} {nfes.std():>6.1f} "
              f"{nfes.min():>5d} {nfes.max():>5d} {ratio:>6.1f}×")

    # Also compute on raw (unnormalized) trajectories for comparison
    print(f"\nRaw (unnormalized) trajectories:")
    print(f"{'α_target':>10s} {'mean':>6s} {'std':>6s} {'min':>5s} {'max':>5s} {'ratio':>6s}")
    print("-" * 45)

    raw_sweep = {}
    for alpha in [1e-3, 1e-4, 1e-5]:
        nfes = []
        for traj in trajectories:
            tau, result = threshold_search(traj, k=2, target_alpha=alpha)
            nfes.append(len(result.retained_indices))
        nfes = np.array(nfes)
        ratio = nfes.max() / max(nfes.min(), 1)
        raw_sweep[f"{alpha:.0e}"] = {
            'mean': float(nfes.mean()),
            'std': float(nfes.std()),
            'min': int(nfes.min()),
            'max': int(nfes.max()),
            'ratio': float(ratio),
        }
        print(f"{alpha:>10.0e} {nfes.mean():>6.1f} {nfes.std():>6.1f} "
              f"{nfes.min():>5d} {nfes.max():>5d} {ratio:>6.1f}×")

    # Save
    out_dir = Path("results/exp6")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'normalized': {k: {kk: vv for kk, vv in v.items() if kk != 'nfes' and kk != 'alphas'}
                       for k, v in all_sweep.items()},
        'raw': raw_sweep,
    }
    with open(out_dir / "exp6_stats.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    # Save arrays for detailed analysis
    for alpha_key, data in all_sweep.items():
        np.save(out_dir / f"nfes_{alpha_key}.npy", np.array(data['nfes']))
        np.save(out_dir / f"alphas_{alpha_key}.npy", np.array(data['alphas']))

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) NFE distribution at multiple alpha targets
    # Pick 3 representative thresholds
    colors = {'1e-02': '#1D9E75', '1e-03': '#534AB7', '1e-05': '#D85A30'}
    labels = {'1e-02': 'α=1e-2 (loose)', '1e-03': 'α=1e-3 (moderate)', '1e-05': 'α=1e-5 (strict)'}
    for akey in ['1e-02', '1e-03', '1e-05']:
        if akey in all_sweep:
            nfes = np.array(all_sweep[akey]['nfes'])
            axes[0].hist(nfes, bins=range(nfes.min(), nfes.max() + 2),
                         alpha=0.5, color=colors[akey], edgecolor='white',
                         label=labels[akey])
    axes[0].set_xlabel("NFE per sample")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) NFE distribution by threshold")
    axes[0].legend(fontsize=7)

    # (b) NFE vs α_traj scatter at moderate threshold
    mid_key = '1e-04'
    if mid_key in all_sweep:
        nfes = np.array(all_sweep[mid_key]['nfes'])
        alps = np.array(all_sweep[mid_key]['alphas'])
        axes[1].scatter(alps, nfes, s=15, alpha=0.6, color='#534AB7')
        # Add trend line
        if len(set(nfes)) > 1:
            z = np.polyfit(alps, nfes, 1)
            x_line = np.linspace(alps.min(), alps.max(), 100)
            axes[1].plot(x_line, np.polyval(z, x_line), 'r--', linewidth=1, alpha=0.7)
        axes[1].set_xlabel(f"α_traj (at target={mid_key})")
        axes[1].set_ylabel("NFE")
        axes[1].set_title("(b) NFE vs trajectory complexity")

    # (c) Max/min ratio vs alpha target
    alphas_sorted = sorted(all_sweep.keys(), key=lambda x: float(x), reverse=True)
    ratios = [all_sweep[a]['ratio'] for a in alphas_sorted]
    means = [all_sweep[a]['mean'] for a in alphas_sorted]
    alpha_vals = [float(a) for a in alphas_sorted]

    ax_ratio = axes[2]
    ax_mean = ax_ratio.twinx()

    l1 = ax_ratio.plot(range(len(alphas_sorted)), ratios, 'D-', color='#D85A30',
                       markersize=6, linewidth=1.5, label='Max/Min ratio')
    l2 = ax_mean.plot(range(len(alphas_sorted)), means, 's--', color='#534AB7',
                      markersize=5, linewidth=1, label='Mean NFE')
    ax_ratio.set_xticks(range(len(alphas_sorted)))
    ax_ratio.set_xticklabels([f"{float(a):.0e}" for a in alphas_sorted], rotation=45, fontsize=7)
    ax_ratio.set_xlabel("α_traj target")
    ax_ratio.set_ylabel("Max/Min NFE ratio", color='#D85A30')
    ax_mean.set_ylabel("Mean NFE", color='#534AB7')
    ax_ratio.set_title("(c) Sample complexity spread")
    lines = l1 + l2
    ax_ratio.legend(lines, [l.get_label() for l in lines], fontsize=7)

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/exp6_adaptive.pdf", dpi=200, bbox_inches='tight')
    print("\nSaved to figures/exp6_adaptive.pdf")


if __name__ == "__main__":
    main()
