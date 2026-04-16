"""
GeoSPRINT Core Algorithm
========================
Hyperplanarity test, multi-level trajectory pruning, and projection score.

This is the heart of the method — generalizes Joshi & Haspel (2020)
from 2-3 PCA dimensions to arbitrary d-dimensional latent spaces.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PruneResult:
    """Result of trajectory pruning."""
    retained_indices: np.ndarray    # indices of kept steps
    pruned_indices: np.ndarray      # indices of removed steps
    residuals: np.ndarray           # residual distance at each point
    projection_score: float         # α_traj
    threshold: float                # τ used
    reduction_pct: float            # % of steps removed


def hyperplanarity_residual(
    window: np.ndarray,
    point: np.ndarray
) -> float:
    """
    Compute the residual distance of `point` from the affine subspace
    spanned by the points in `window`.

    Uses QR factorization for numerical stability and efficiency.

    Parameters
    ----------
    window : np.ndarray, shape (k, d)
        The k points defining the affine subspace.
        The subspace passes through window[0] and is spanned by
        the directions (window[i] - window[0]) for i=1..k-1.
    point : np.ndarray, shape (d,)
        The point to test.

    Returns
    -------
    float
        Residual distance (0 = perfectly in the subspace).
    """
    anchor = window[0]
    directions = (window[1:] - anchor)  # shape: (k-1, d)

    if directions.shape[0] == 0:
        # Only one point in window — residual is just distance
        return np.linalg.norm(point - anchor)

    # Thin QR factorization: directions.T = Q @ R
    # Q columns span the same subspace as the direction vectors
    Q, _ = np.linalg.qr(directions.T, mode='reduced')  # Q: (d, k-1)

    # Project (point - anchor) onto orthogonal complement of span
    diff = point - anchor
    projection = Q @ (Q.T @ diff)
    residual_vec = diff - projection

    return np.linalg.norm(residual_vec)


def prune_trajectory(
    trajectory: np.ndarray,
    k: int = 2,
    threshold: float = 1e-3,
) -> PruneResult:
    """
    Prune geometrically redundant points from an ordered trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, d)
        Ordered sequence of latent states (e.g., z_T, z_{T-1}, ..., z_0).
    k : int
        Window size for hyperplanarity test.
        k=2: collinearity (3 consecutive points)
        k=3: coplanarity (4 consecutive points)
        k=l: test against (l-1)-dim affine subspace
    threshold : float
        Residual distance threshold τ. Points with residual < τ are pruned.

    Returns
    -------
    PruneResult
        Contains retained/pruned indices, residuals, and projection score.
    """
    N, d = trajectory.shape
    assert k >= 2, "Window size must be at least 2"
    assert k <= min(N - 1, d), f"Window size {k} too large for N={N}, d={d}"

    retained = list(range(k))  # Always keep first k points
    pruned = []
    residuals = np.zeros(N)

    for i in range(k, N):
        # Build window from last k retained points
        window_indices = retained[-k:]
        window = trajectory[window_indices]

        r = hyperplanarity_residual(window, trajectory[i])
        residuals[i] = r

        if r < threshold:
            pruned.append(i)
        else:
            retained.append(i)

    # Always keep the last point (z_0 = final sample)
    if N - 1 not in retained:
        if N - 1 in pruned:
            pruned.remove(N - 1)
        retained.append(N - 1)
        retained.sort()

    retained = np.array(retained)
    pruned = np.array(pruned) if pruned else np.array([], dtype=int)

    alpha = projection_score(trajectory, pruned, residuals=residuals)
    reduction = len(pruned) / N * 100

    return PruneResult(
        retained_indices=retained,
        pruned_indices=pruned,
        residuals=residuals,
        projection_score=alpha,
        threshold=threshold,
        reduction_pct=reduction,
    )


def multi_level_prune(
    trajectory: np.ndarray,
    levels: List[int] = [2, 3],
    threshold: float = 1e-3,
) -> PruneResult:
    """
    Progressive multi-level pruning: apply Level-1 (collinearity),
    then Level-2 (coplanarity) on surviving points, etc.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, d)
    levels : list of int
        Window sizes to apply sequentially [2, 3, ...].
    threshold : float
        Residual threshold τ (same for all levels).

    Returns
    -------
    PruneResult
    """
    current_indices = np.arange(len(trajectory))
    all_pruned = []

    for k in levels:
        if len(current_indices) <= k + 1:
            break

        current_traj = trajectory[current_indices]
        result = prune_trajectory(current_traj, k=k, threshold=threshold)

        # Map back to original indices
        pruned_original = current_indices[result.pruned_indices]
        all_pruned.extend(pruned_original.tolist())

        current_indices = current_indices[result.retained_indices]

    all_pruned = np.array(sorted(set(all_pruned)), dtype=int)
    retained = np.array([i for i in range(len(trajectory)) if i not in all_pruned])

    alpha = projection_score(trajectory, all_pruned)
    reduction = len(all_pruned) / len(trajectory) * 100

    return PruneResult(
        retained_indices=retained,
        pruned_indices=all_pruned,
        residuals=np.zeros(len(trajectory)),  # Simplified for multi-level
        projection_score=alpha,
        threshold=threshold,
        reduction_pct=reduction,
    )


def projection_score(
    trajectory: np.ndarray,
    pruned_indices: np.ndarray,
    residuals: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the trajectory projection score α_traj.

    α_traj = (sum of squared residuals of pruned points) / tr(C_Z)

    This measures the fraction of trajectory variance that is LOST
    by pruning — i.e., the residual variance orthogonal to what
    the retained neighbors capture.

    For collinear trajectories: α_traj = 0 (Proposition 1).
    For curved trajectories: α_traj > 0, proportional to curvature.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, d)
    pruned_indices : np.ndarray
        Indices of removed points.
    residuals : np.ndarray, optional
        Pre-computed residual distances from prune_trajectory().
        If None, uses a fallback based on distance to nearest retained neighbor.

    Returns
    -------
    float
        α_traj ∈ [0, 1]. Lower = less information lost.
    """
    if len(pruned_indices) == 0:
        return 0.0

    # Total variance
    global_mean = trajectory.mean(axis=0)
    centered_all = trajectory - global_mean
    var_total = np.sum(centered_all ** 2)

    if var_total == 0:
        return 0.0

    if residuals is not None and len(residuals) > 0:
        # Use pre-computed residuals: sum of squared residuals
        residual_variance = np.sum(residuals[pruned_indices] ** 2)
    else:
        # Fallback: compute residuals as distance to linearly interpolated
        # position between nearest retained neighbors
        retained = sorted(set(range(len(trajectory))) - set(pruned_indices))
        residual_variance = 0.0
        for idx in pruned_indices:
            # Find nearest retained neighbors before and after
            before = [r for r in retained if r < idx]
            after = [r for r in retained if r > idx]
            if before and after:
                b, a = before[-1], after[0]
                # Linear interpolation
                alpha = (idx - b) / (a - b)
                interpolated = trajectory[b] * (1 - alpha) + trajectory[a] * alpha
                residual_variance += np.sum((trajectory[idx] - interpolated) ** 2)
            elif before:
                residual_variance += np.sum((trajectory[idx] - trajectory[before[-1]]) ** 2)
            elif after:
                residual_variance += np.sum((trajectory[idx] - trajectory[after[0]]) ** 2)

    return residual_variance / var_total


def threshold_search(
    trajectory: np.ndarray,
    k: int = 2,
    target_alpha: float = 1e-3,
    target_reduction_min: float = 30.0,
    max_iters: int = 20,
) -> Tuple[float, PruneResult]:
    """
    Binary search for optimal threshold τ that achieves:
    - projection_score ≤ target_alpha
    - reduction_pct ≥ target_reduction_min

    Following the procedure in Joshi & Haspel (2020), Section 2.4.

    Returns
    -------
    (optimal_tau, PruneResult)
    """
    # Compute scale of the trajectory to set initial bounds
    diffs = np.diff(trajectory, axis=0)
    step_norms = np.linalg.norm(diffs, axis=1)
    median_step = np.median(step_norms)

    tau_low = 0.0
    tau_high = median_step * 2.0
    best_result = None
    best_tau = tau_high

    for _ in range(max_iters):
        tau_mid = (tau_low + tau_high) / 2.0
        result = prune_trajectory(trajectory, k=k, threshold=tau_mid)

        if result.projection_score <= target_alpha and \
           result.reduction_pct >= target_reduction_min:
            # Good: try to prune even more (increase threshold)
            best_result = result
            best_tau = tau_mid
            tau_low = tau_mid
        elif result.projection_score > target_alpha:
            # Too aggressive: reduce threshold
            tau_high = tau_mid
        else:
            # Not enough reduction: increase threshold
            tau_low = tau_mid

    if best_result is None:
        best_result = prune_trajectory(trajectory, k=k, threshold=best_tau)

    return best_tau, best_result


# ---- Convenience / batch operations ----

def batch_prune(
    trajectories: List[np.ndarray],
    k: int = 2,
    target_alpha: float = 1e-3,
) -> List[PruneResult]:
    """Prune a batch of trajectories with automatic threshold search."""
    results = []
    for traj in trajectories:
        tau, result = threshold_search(traj, k=k, target_alpha=target_alpha)
        results.append(result)
    return results
