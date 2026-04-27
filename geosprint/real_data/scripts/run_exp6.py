"""
Experiment 6 (Final): Per-Sample Geometric Complexity
======================================================
Shows:
  (a) Per-sample curvature profiles overlaid with universal profile
      (the universal profile is what feeds into LogSNR+curvature schedule)
  (b) NFE distribution at multiple thresholds
  (c) How well individual samples agree with the universal curvature
      (high agreement = universal schedule works well for all samples)

Usage: python scripts/run_exp6.py
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import threshold_search, prune_trajectory


def find_threshold_for_K(traj, target_K, tol=3, max_iter=30):
    tau_lo, tau_hi = 0.0, np.max(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 3
    for _ in range(max_iter):
        tau_mid = (tau_lo + tau_hi) / 2
        result = prune_trajectory(traj, k=2, threshold=tau_mid)
        K = len(result.retained_indices)
        if abs(K - target_K) <= tol: return tau_mid, result
        elif K < target_K: tau_hi = tau_mid
        else: tau_lo = tau_mid
    return tau_mid, result


def main():
    print("Experiment 6: Per-Sample Geometric Complexity")
    print("=" * 50)

    traj_dir = Path("results/exp1/trajectories")
    if not traj_dir.exists():
        print("ERROR: Run run_exp1.py first!"); return

    files = sorted(traj_dir.glob("traj_*.npy"))
    trajectories = [np.load(f) for f in files]
    dts = np.load(traj_dir / "ddim_timesteps.npy")
    print(f"Loaded {len(trajectories)} trajectories, {len(dts)} pool timesteps")

    normalized = []
    for traj in trajectories:
        mu = traj.mean(0, keepdims=True)
        s = traj.std(0, keepdims=True) + 1e-8
        normalized.append((traj - mu) / s)

    # ── Analysis 1: NFE at multiple alpha targets ──
    alpha_targets = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    sweep = {}

    print(f"\n{'alpha':>10s} {'mean':>6s} {'std':>6s} {'min':>5s} {'max':>5s} {'ratio':>6s}")
    print("-" * 45)
    for alpha in alpha_targets:
        nfes = []
        for traj in normalized:
            tau, result = threshold_search(traj, k=2, target_alpha=alpha)
            nfes.append(len(result.retained_indices))
        nfes = np.array(nfes)
        ratio = nfes.max() / max(nfes.min(), 1)
        sweep[f"{alpha:.0e}"] = {
            'nfes': nfes.tolist(),
            'mean': float(nfes.mean()), 'std': float(nfes.std()),
            'min': int(nfes.min()), 'max': int(nfes.max()),
            'ratio': float(ratio),
        }
        print(f"{alpha:>10.0e} {nfes.mean():>6.1f} {nfes.std():>6.1f} "
              f"{nfes.min():>5d} {nfes.max():>5d} {ratio:>6.1f}x")

    # ── Analysis 2: Per-sample curvature profiles ──
    print(f"\nComputing per-sample curvature profiles...")
    N_pool = len(dts)
    all_profiles = []
    all_K = []
    all_alpha_vals = []

    for traj in tqdm(normalized, desc="Profiles"):
        N_traj = N_pool + 1
        tau, res = find_threshold_for_K(traj, 50, tol=5)
        profile = np.zeros(N_traj)
        for idx in res.retained_indices:
            if idx < N_traj: profile[idx] = 1.0
        all_profiles.append(profile[1:N_pool+1])
        all_K.append(len(res.retained_indices))
        all_alpha_vals.append(res.projection_score)

    profiles = np.stack(all_profiles)
    all_K = np.array(all_K)
    universal = profiles.mean(axis=0)

    # Per-sample correlation with universal
    per_corr = [float(np.corrcoef(profiles[i], universal)[0, 1]) for i in range(len(profiles))]
    mean_corr = float(np.mean(per_corr))

    print(f"\n  Per-sample K (target=50): mean={all_K.mean():.1f} +/- {all_K.std():.1f}, "
          f"range=[{all_K.min()}, {all_K.max()}]")
    print(f"  Mean correlation with universal curvature: {mean_corr:.3f}")
    print(f"  (High correlation = universal schedule works well for all samples)")

    # ── Save ──
    out_dir = Path("results/exp6")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'sweep': {k: {kk: vv for kk, vv in v.items() if kk != 'nfes'}
                  for k, v in sweep.items()},
        'per_sample_K_mean': float(all_K.mean()),
        'per_sample_K_std': float(all_K.std()),
        'per_sample_K_range': [int(all_K.min()), int(all_K.max())],
        'curvature_agreement_mean': mean_corr,
        'curvature_agreement_std': float(np.std(per_corr)),
    }
    with open(out_dir / "exp6_stats.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    np.save(out_dir / "curvature_profiles.npy", profiles)
    np.save(out_dir / "universal_curvature.npy", universal)

    # ── Plot ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Per-sample curvature profiles overlaid with universal
    t_norm = np.linspace(1, 0, N_pool)
    for i in range(min(20, len(profiles))):
        axes[0].plot(t_norm, profiles[i], alpha=0.15, color='#534AB7', linewidth=0.5)
    axes[0].plot(t_norm, universal, color='#D85A30', linewidth=2, label='Universal (mean)')
    axes[0].set_xlabel("Normalized time (t/T → 0)")
    axes[0].set_ylabel("Retention probability")
    axes[0].set_title("(a) Per-sample vs universal curvature")
    axes[0].legend(fontsize=8)

    # (b) NFE distribution at three thresholds
    colors = {'1e-02': '#1D9E75', '1e-03': '#534AB7', '1e-05': '#D85A30'}
    labels_map = {'1e-02': 'α=1e-2 (loose)', '1e-03': 'α=1e-3 (moderate)', '1e-05': 'α=1e-5 (strict)'}
    for akey in ['1e-02', '1e-03', '1e-05']:
        if akey in sweep:
            nfes = np.array(sweep[akey]['nfes'])
            axes[1].hist(nfes, bins=range(nfes.min(), nfes.max() + 2),
                         alpha=0.5, color=colors[akey], edgecolor='white',
                         label=labels_map[akey])
    axes[1].set_xlabel("NFE per sample")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) NFE distribution by threshold")
    axes[1].legend(fontsize=7)

    # (c) Curvature agreement histogram
    axes[2].hist(per_corr, bins=20, color='#534AB7', alpha=0.85, edgecolor='white')
    axes[2].axvline(mean_corr, color='#D85A30', linestyle='--',
                    label=f'mean={mean_corr:.3f}')
    axes[2].set_xlabel("Correlation with universal curvature")
    axes[2].set_ylabel("Count")
    axes[2].set_title("(c) Curvature profile agreement")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/exp6_adaptive.pdf", dpi=200, bbox_inches='tight')
    print("\nSaved to figures/exp6_adaptive.pdf")


if __name__ == "__main__":
    main()
