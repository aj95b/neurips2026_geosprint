"""
Rebuttal experiment: window size k ablation.

Answers reviewer point 5 (7G9F, eXSn): "higher-order hyperplanarity (k>2)
is not thoroughly investigated."

For k in {2, 3, 5}, we re-run the pruning on the SAME CIFAR-10 reference
trajectories and report:
  - how the retained-step count changes at a fixed threshold-search target
  - how alpha_traj changes
  - how much the resulting curvature profile / schedule shifts (correlation
    with the k=2 profile)

This is CPU-only and reuses existing trajectories (no regeneration, no FID).

Usage:
    python scripts/run_kablation.py
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory


def find_threshold_for_K(traj, target_K, k, tol=3, max_iter=30):
    tau_lo, tau_hi = 0.0, np.max(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 3
    for _ in range(max_iter):
        tau_mid = (tau_lo + tau_hi) / 2
        result = prune_trajectory(traj, k=k, threshold=tau_mid)
        Kr = len(result.retained_indices)
        if abs(Kr - target_K) <= tol:
            return tau_mid, result
        elif Kr < target_K:
            tau_hi = tau_mid
        else:
            tau_lo = tau_mid
    return tau_mid, result


def retention_profile(trajs, pool_len, k, target_K=50):
    """Curvature density profile at window size k."""
    N = pool_len + 1
    counts = np.zeros(N)
    alphas = []
    Ks = []
    for traj in trajs:
        tau, res = find_threshold_for_K(traj, target_K, k=k, tol=5)
        alphas.append(res.projection_score)
        Ks.append(len(res.retained_indices))
        for idx in res.retained_indices:
            if idx < N:
                counts[idx] += 1
    profile = counts[1:pool_len + 1] / len(trajs)
    return profile, float(np.mean(alphas)), float(np.mean(Ks))


def main():
    traj_dir = Path("results/exp1/trajectories")
    if not (traj_dir / "ddim_timesteps.npy").exists():
        print("ERROR: run_exp1.py trajectories not found.")
        return

    trajs = [np.load(f) for f in sorted(traj_dir.glob("traj_*.npy"))]
    dts = np.load(traj_dir / "ddim_timesteps.npy")
    print(f"Loaded {len(trajs)} CIFAR-10 trajectories, pool={len(dts)}")

    # Normalize (same as main pipeline)
    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True)
        s = t.std(0, keepdims=True) + 1e-8
        normalized.append((t - mu) / s)

    k_values = [2, 3, 5]
    profiles = {}
    results = {}

    print(f"\n{'k':>3s} {'mean_K':>8s} {'mean_alpha':>12s} {'corr_vs_k2':>12s}")
    print("-" * 40)

    for k in k_values:
        profile, mean_alpha, mean_K = retention_profile(normalized, len(dts), k=k, target_K=50)
        profiles[k] = profile
        # Correlation of curvature profile with k=2
        if k == 2:
            corr = 1.0
        else:
            if profile.std() > 0 and profiles[2].std() > 0:
                corr = float(np.corrcoef(profile, profiles[2])[0, 1])
            else:
                corr = float('nan')
        results[k] = {"mean_K": mean_K, "mean_alpha": mean_alpha, "corr_vs_k2": corr}
        print(f"{k:>3d} {mean_K:>8.1f} {mean_alpha:>12.2e} {corr:>12.3f}")

    out = Path("results/kablation")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "kablation.json", "w") as f:
        json.dump(results, f, indent=2)
    for k in k_values:
        np.save(out / f"profile_k{k}.npy", profiles[k])

    print(f"\nInterpretation:")
    print(f"  - High corr_vs_k2 means the curvature profile is stable w.r.t. k")
    print(f"  - k=2 (collinearity) is the cheapest and the default")
    print(f"\nSaved to results/kablation/kablation.json")


if __name__ == "__main__":
    main()
