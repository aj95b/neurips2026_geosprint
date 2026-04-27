"""
Experiment 2: Schedule w(t) Analysis
=====================================
Reads saved trajectories from Experiment 1. No GPU needed.

Usage: python scripts/run_exp2.py
Time:  ~10 minutes
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.schedule import extract_universal_schedule


def main():
    print("Experiment 2: Schedule w(t) Analysis")
    print("=" * 50)

    traj_dir = Path("results/exp1/trajectories")
    if not traj_dir.exists():
        print("ERROR: Run 'python scripts/run_exp1.py' first!")
        return

    files = sorted(traj_dir.glob("traj_*.npy"))
    trajectories = [np.load(f) for f in files]
    timesteps = np.load(traj_dir / "ddim_timesteps.npy")
    timesteps = np.concatenate([[timesteps[0] + 20], timesteps])
    N = len(timesteps)
    print(f"Loaded {len(trajectories)} trajectories, {N} timesteps")

    # Normalize (same as exp1)
    normalized = []
    for traj in trajectories:
        mu = traj.mean(axis=0, keepdims=True)
        std = traj.std(axis=0, keepdims=True) + 1e-8
        normalized.append((traj - mu) / std)

    sched = extract_universal_schedule(normalized, timesteps, nfe_budget=50, k=2, target_alpha=1e-3)
    w = sched.retention_freq

    early = w[:int(N * 0.2)]
    middle = w[int(N * 0.2):int(N * 0.8)]
    late = w[int(N * 0.8):]

    print(f"\nRetention frequency by zone:")
    print(f"  Early  (t/T > 0.8): mean w = {early.mean():.3f}")
    print(f"  Middle (0.2-0.8):   mean w = {middle.mean():.3f}")
    print(f"  Late   (t/T < 0.2): mean w = {late.mean():.3f}")

    out_dir = Path("results/exp2")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "retention_freq.npy", w)
    json.dump({'early': float(early.mean()), 'middle': float(middle.mean()), 'late': float(late.mean())},
              open(out_dir / "exp2_analysis.json", 'w'), indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    t_norm = np.linspace(1, 0, N)

    axes[0].bar(t_norm, w, width=1.0/N, color='#534AB7', alpha=0.7)
    axes[0].set_xlabel("Normalized time (t/T → 0)")
    axes[0].set_ylabel("Retention freq w(t)")
    axes[0].set_title("(a) Step retention frequency")

    for budget in [10, 20, 50]:
        s = extract_universal_schedule(normalized, timesteps, nfe_budget=budget, k=2, target_alpha=1e-3)
        y = np.zeros(N)
        for ts in s.timesteps:
            idx = np.argmin(np.abs(timesteps - ts))
            if idx < N: y[idx] = 1
        kept = np.where(y > 0)[0]
        axes[1].scatter(t_norm[kept], [budget]*len(kept), s=8, alpha=0.9, label=f"K={budget}")
    axes[1].set_xlabel("Normalized time")
    axes[1].set_ylabel("NFE budget K")
    axes[1].set_title("(b) Selected steps by budget")
    axes[1].legend(fontsize=8)

    cumw = np.cumsum(w) / (np.sum(w) + 1e-10)
    axes[2].plot(t_norm, cumw, 'k-', linewidth=1.5)
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Normalized time")
    axes[2].set_ylabel("Cumulative retention")
    axes[2].set_title("(c) Cumulative distribution")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/exp2_schedule.pdf", dpi=200, bbox_inches='tight')
    print("\nSaved to figures/exp2_schedule.pdf")


if __name__ == "__main__":
    main()
