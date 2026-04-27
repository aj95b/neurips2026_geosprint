"""
Experiment 5 (Improved): Schedule Convergence vs B
====================================================
Uses fuzzy Jaccard (timesteps within ±pool_spacing count as match)
and retention-frequency correlation as convergence metrics.

Usage: python scripts/run_exp5.py
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.schedule import extract_universal_schedule


def fuzzy_jaccard(a, b, tolerance):
    """Count a match if any element in b is within ±tolerance of an element in a."""
    a, b = set(a.tolist()), set(b.tolist())
    matches = 0
    for x in a:
        if any(abs(x - y) <= tolerance for y in b):
            matches += 1
    for y in b:
        if any(abs(y - x) <= tolerance for x in a):
            matches += 1
    # Fuzzy intersection / fuzzy union
    fuzzy_inter = matches / 2  # counted from both sides, average
    fuzzy_union = len(a) + len(b) - fuzzy_inter
    return fuzzy_inter / max(fuzzy_union, 1)


def exact_jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


def retention_correlation(sched_a, sched_b, N):
    """Correlation between retention frequency vectors."""
    vec_a = np.zeros(N)
    vec_b = np.zeros(N)
    for t in sched_a:
        idx = min(int(t), N - 1)
        vec_a[idx] = 1
    for t in sched_b:
        idx = min(int(t), N - 1)
        vec_b[idx] = 1
    if vec_a.std() == 0 or vec_b.std() == 0:
        return 1.0
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def main():
    print("Experiment 5: Schedule Convergence vs B")
    print("=" * 50)

    traj_dir = Path("results/exp1/trajectories")
    if not traj_dir.exists():
        print("ERROR: Run run_exp1.py first!")
        return

    files = sorted(traj_dir.glob("traj_*.npy"))
    all_trajs = [np.load(f) for f in files]
    timesteps = np.load(traj_dir / "ddim_timesteps.npy")
    timesteps = np.concatenate([[timesteps[0] + 20], timesteps])
    B_total = len(all_trajs)
    N = len(timesteps)
    print(f"Loaded {B_total} trajectories, {N} timesteps")

    # Pool spacing
    raw_ts = np.load(traj_dir / "ddim_timesteps.npy")
    pool_spacing = int(np.median(np.abs(np.diff(raw_ts))))
    print(f"Pool spacing: {pool_spacing}")

    # Normalize
    normalized = []
    for traj in all_trajs:
        mu = traj.mean(0, keepdims=True)
        s = traj.std(0, keepdims=True) + 1e-8
        normalized.append((traj - mu) / s)

    # Reference = full set
    nfe_budget = 20
    ref = extract_universal_schedule(normalized, timesteps, nfe_budget=nfe_budget, k=2, target_alpha=1e-3)

    B_values = [b for b in [5, 10, 15, 20, 30, 40, 50, 60, 75, 100] if b <= B_total]
    num_trials = 10

    results = {}
    for B in B_values:
        exact_sims = []
        fuzzy_sims = []
        corrs = []
        for trial in range(num_trials):
            idx = np.random.choice(B_total, B, replace=False)
            subset = [normalized[i] for i in idx]
            s = extract_universal_schedule(subset, timesteps, nfe_budget=nfe_budget, k=2, target_alpha=1e-3)
            exact_sims.append(exact_jaccard(s.timesteps, ref.timesteps))
            fuzzy_sims.append(fuzzy_jaccard(s.timesteps, ref.timesteps, tolerance=pool_spacing))
            corrs.append(retention_correlation(s.timesteps, ref.timesteps, N))

        results[B] = {
            'exact_mean': float(np.mean(exact_sims)),
            'exact_std': float(np.std(exact_sims)),
            'fuzzy_mean': float(np.mean(fuzzy_sims)),
            'fuzzy_std': float(np.std(fuzzy_sims)),
            'corr_mean': float(np.mean(corrs)),
            'corr_std': float(np.std(corrs)),
        }
        print(f"  B={B:3d}: Exact={results[B]['exact_mean']:.3f}  "
              f"Fuzzy={results[B]['fuzzy_mean']:.3f}  "
              f"Corr={results[B]['corr_mean']:.3f}")

    out_dir = Path("results/exp5")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exp5_results.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bs = sorted(results.keys())

    # Left: Jaccard (exact and fuzzy)
    ax1.errorbar(bs, [results[b]['exact_mean'] for b in bs],
                 yerr=[results[b]['exact_std'] for b in bs],
                 fmt='s--', capsize=4, color='#888780', markersize=5, label='Exact Jaccard')
    ax1.errorbar(bs, [results[b]['fuzzy_mean'] for b in bs],
                 yerr=[results[b]['fuzzy_std'] for b in bs],
                 fmt='o-', capsize=4, color='#534AB7', markersize=5, label=f'Fuzzy Jaccard (±{pool_spacing})')
    ax1.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='95% threshold')
    ax1.set_xlabel("Reference batch size B")
    ax1.set_ylabel("Similarity to full schedule")
    ax1.set_title("(a) Schedule similarity vs B")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 1.05)

    # Right: Correlation
    ax2.errorbar(bs, [results[b]['corr_mean'] for b in bs],
                 yerr=[results[b]['corr_std'] for b in bs],
                 fmt='D-', capsize=4, color='#D85A30', markersize=5)
    ax2.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='95% threshold')
    ax2.set_xlabel("Reference batch size B")
    ax2.set_ylabel("Correlation with full schedule")
    ax2.set_title("(b) Retention frequency correlation vs B")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/exp5_convergence.pdf", dpi=200, bbox_inches='tight')
    print("\nSaved to figures/exp5_convergence.pdf")


if __name__ == "__main__":
    main()
