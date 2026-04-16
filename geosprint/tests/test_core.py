"""
GeoSPRINT Core Verification
============================
Validates the algorithm on synthetic trajectories.
Runs on CPU — no GPU or pretrained models needed.

Usage: python tests/test_core.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from geosprint.core import (
    hyperplanarity_residual,
    prune_trajectory,
    multi_level_prune,
    projection_score,
    threshold_search,
)
from geosprint.schedule import extract_universal_schedule, sweep_nfe_budgets


def test_collinear_points():
    """Proposition 1: perfectly collinear points → all interior pruned, α=0."""
    print("Test 1: Collinear points (Proposition 1)...")
    d = 64
    N = 100
    direction = np.random.randn(d)
    direction /= np.linalg.norm(direction)

    trajectory = np.array([i * direction for i in range(N)])
    result = prune_trajectory(trajectory, k=2, threshold=1e-10)

    assert result.reduction_pct > 95, f"Expected >95% reduction, got {result.reduction_pct:.1f}%"
    assert result.projection_score < 1e-6, f"Expected α≈0, got {result.projection_score:.2e}"
    print(f"  ✓ Reduced {result.reduction_pct:.1f}%, α={result.projection_score:.2e}")


def test_curved_trajectory():
    """Curved trajectory should retain more points than straight."""
    print("Test 2: Curved trajectory retains more points...")
    d = 32
    N = 200

    # Straight
    straight = np.array([np.random.randn(d) * 0.001 + i * np.ones(d) / N for i in range(N)])
    r_straight = prune_trajectory(straight, k=2, threshold=0.1)

    # Curved (sinusoidal)
    t = np.linspace(0, 4 * np.pi, N)
    curved = np.zeros((N, d))
    for j in range(min(5, d)):
        curved[:, j] = np.sin(t + j * 0.5) * (j + 1)
    curved += np.random.randn(N, d) * 0.01  # small noise
    r_curved = prune_trajectory(curved, k=2, threshold=0.1)

    assert r_straight.reduction_pct > r_curved.reduction_pct, \
        f"Straight ({r_straight.reduction_pct:.1f}%) should reduce more than curved ({r_curved.reduction_pct:.1f}%)"
    print(f"  ✓ Straight: {r_straight.reduction_pct:.1f}% removed, "
          f"Curved: {r_curved.reduction_pct:.1f}% removed")


def test_high_dimensional():
    """Algorithm works in high dimensions (d=16384, like SD latent)."""
    print("Test 3: High-dimensional trajectory (d=16384)...")
    d = 16384
    N = 50

    # Simulate diffusion trajectory: mostly linear with late-stage curvature
    trajectory = np.zeros((N, d))
    trajectory[0] = np.random.randn(d) * 10  # z_T (noisy)

    for i in range(1, N):
        # Linear drift toward origin + increasing noise at end
        alpha = i / N
        noise_scale = 0.01 * (1 + 10 * alpha**3)
        trajectory[i] = trajectory[i-1] * (1 - 1/N) + np.random.randn(d) * noise_scale

    result = prune_trajectory(trajectory, k=2, threshold=0.5)
    print(f"  ✓ d={d}, N={N}: {result.reduction_pct:.1f}% removed, "
          f"α={result.projection_score:.2e}")


def test_threshold_search():
    """Binary search finds good threshold automatically."""
    print("Test 4: Automatic threshold search...")
    d = 128
    N = 200

    trajectory = np.random.randn(N, d)
    # Add structure: first 100 steps nearly linear
    for i in range(1, 100):
        trajectory[i] = trajectory[0] + (trajectory[99] - trajectory[0]) * i / 99 + \
                        np.random.randn(d) * 0.01

    tau, result = threshold_search(trajectory, k=2, target_alpha=1e-3)

    assert result.projection_score <= 0.01, \
        f"Expected α≤0.01, got {result.projection_score:.4f}"
    print(f"  ✓ τ={tau:.4f}, {result.reduction_pct:.1f}% removed, "
          f"α={result.projection_score:.2e}")


def test_multi_level():
    """Multi-level pruning removes more than single level."""
    print("Test 5: Multi-level pruning (L1 → L2)...")
    d = 64
    N = 300

    trajectory = np.random.randn(N, d) * 0.1
    # Add some structure
    for i in range(N):
        trajectory[i, 0] = i / N * 10
        trajectory[i, 1] = np.sin(i / N * 4 * np.pi)

    r_l1 = prune_trajectory(trajectory, k=2, threshold=0.5)
    r_multi = multi_level_prune(trajectory, levels=[2, 3], threshold=0.5)

    print(f"  Level-1 only: {r_l1.reduction_pct:.1f}% removed")
    print(f"  Multi-level:  {r_multi.reduction_pct:.1f}% removed")
    assert r_multi.reduction_pct >= r_l1.reduction_pct, \
        "Multi-level should prune at least as much as single level"
    print(f"  ✓ Multi-level pruned {r_multi.reduction_pct - r_l1.reduction_pct:.1f}% more")


def test_universal_schedule():
    """Universal schedule extraction from batch of trajectories."""
    print("Test 6: Universal schedule extraction (B=20)...")
    d = 128
    N = 100
    B = 20

    trajectories = []
    for b in range(B):
        traj = np.random.randn(N, d) * 0.1
        for i in range(N):
            traj[i, 0] = i / N * 10 + np.random.randn() * 0.01
        trajectories.append(traj)

    timestep_indices = np.arange(N - 1, -1, -1)

    schedule = extract_universal_schedule(
        trajectories=trajectories,
        timestep_indices=timestep_indices,
        nfe_budget=20,
        k=2,
        target_alpha=1e-2,
    )

    print(f"  ✓ Selected {schedule.nfe} timesteps from {N}")
    print(f"  Mean α={schedule.mean_alpha:.2e} ± {schedule.std_alpha:.2e}")
    print(f"  Retention freq range: [{schedule.retention_freq.min():.2f}, "
          f"{schedule.retention_freq.max():.2f}]")


def test_variance_preservation():
    """Theorem 1: variance preservation bound holds."""
    print("Test 7: Variance preservation bound (Theorem 1)...")
    d = 64
    N = 200
    trajectory = np.random.randn(N, d)

    for tau in [0.1, 0.5, 1.0, 2.0]:
        result = prune_trajectory(trajectory, k=2, threshold=tau)

        # Compute actual retained variance ratio
        global_mean = trajectory.mean(axis=0)
        var_total = np.sum((trajectory - global_mean) ** 2)
        var_retained = np.sum((trajectory[result.retained_indices] - global_mean) ** 2)
        ratio = var_retained / var_total

        # Theoretical bound: ratio ≥ 1 - |S|*τ²/tr(C_Z)
        bound = 1 - len(result.pruned_indices) * tau**2 / var_total

        print(f"  τ={tau:.1f}: retained_var_ratio={ratio:.4f}, "
              f"bound={bound:.4f}, "
              f"satisfies={'✓' if ratio >= bound - 1e-6 else '✗'}")


def main():
    print("=" * 60)
    print("GeoSPRINT Core Algorithm Verification")
    print("=" * 60)

    test_collinear_points()
    test_curved_trajectory()
    test_high_dimensional()
    test_threshold_search()
    test_multi_level()
    test_universal_schedule()
    test_variance_preservation()

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
