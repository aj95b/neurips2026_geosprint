"""
GeoSPRINT Schedule Extraction
==============================
Universal schedule aggregation across reference trajectories,
and sample-adaptive scheduling.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from .core import prune_trajectory, multi_level_prune, threshold_search, PruneResult


@dataclass
class UniversalSchedule:
    """A fixed step schedule derived from reference trajectories."""
    timesteps: np.ndarray           # retained timestep indices, sorted
    retention_freq: np.ndarray      # w(t) for all T timesteps
    nfe: int                        # number of function evaluations
    mean_alpha: float               # mean projection score across refs
    std_alpha: float                # std of projection scores


def extract_universal_schedule(
    trajectories: List[np.ndarray],
    timestep_indices: np.ndarray,
    nfe_budget: int,
    k: int = 2,
    target_alpha: float = 1e-3,
    levels: Optional[List[int]] = None,
) -> UniversalSchedule:
    """
    Extract a universal schedule from B reference trajectories.

    Algorithm 1 from the paper:
    1. Prune each trajectory independently
    2. Compute retention frequency w(t) for each timestep
    3. Select top-K timesteps by w(t)

    Parameters
    ----------
    trajectories : list of np.ndarray, each shape (N, d)
        B reference trajectories (full denoising paths).
    timestep_indices : np.ndarray, shape (N,)
        The actual timestep values [T, T-1, ..., 0].
    nfe_budget : int
        Target number of function evaluations K.
    k : int
        Window size for hyperplanarity test.
    target_alpha : float
        Target projection score for threshold search.
    levels : list of int, optional
        If provided, use multi-level pruning with these window sizes.

    Returns
    -------
    UniversalSchedule
    """
    B = len(trajectories)
    N = len(timestep_indices)
    retention_counts = np.zeros(N, dtype=float)
    alphas = []

    for traj in trajectories:
        if levels is not None:
            result = multi_level_prune(traj, levels=levels, threshold=0.0)
            # Need to search for threshold — use level-1 for search
            tau, result = threshold_search(traj, k=k, target_alpha=target_alpha)
        else:
            tau, result = threshold_search(traj, k=k, target_alpha=target_alpha)

        # Mark retained timesteps
        for idx in result.retained_indices:
            retention_counts[idx] += 1

        alphas.append(result.projection_score)

    retention_freq = retention_counts / B

    # Select top-K timesteps by retention frequency
    # Always include first (t=T) and last (t=0)
    mandatory = {0, N - 1}
    top_k_indices = np.argsort(-retention_freq)

    selected = set()
    for idx in top_k_indices:
        selected.add(idx)
        if len(selected) >= nfe_budget:
            break
    selected |= mandatory

    selected_sorted = np.sort(list(selected))
    selected_timesteps = timestep_indices[selected_sorted]

    return UniversalSchedule(
        timesteps=selected_timesteps,
        retention_freq=retention_freq,
        nfe=len(selected_sorted),
        mean_alpha=np.mean(alphas),
        std_alpha=np.std(alphas),
    )


def sweep_nfe_budgets(
    trajectories: List[np.ndarray],
    timestep_indices: np.ndarray,
    nfe_budgets: List[int],
    k: int = 2,
    target_alpha: float = 1e-3,
) -> Dict[int, UniversalSchedule]:
    """
    Extract universal schedules for multiple NFE budgets.
    Reuses the same pruning results — only the top-K selection changes.

    Returns dict mapping nfe_budget → UniversalSchedule.
    """
    B = len(trajectories)
    N = len(timestep_indices)
    retention_counts = np.zeros(N, dtype=float)
    alphas = []

    # Prune all trajectories once
    for traj in trajectories:
        tau, result = threshold_search(traj, k=k, target_alpha=target_alpha)
        for idx in result.retained_indices:
            retention_counts[idx] += 1
        alphas.append(result.projection_score)

    retention_freq = retention_counts / B
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)

    # For each budget, select top-K
    results = {}
    top_k_indices = np.argsort(-retention_freq)

    for budget in nfe_budgets:
        selected = set()
        for idx in top_k_indices:
            selected.add(idx)
            if len(selected) >= budget:
                break
        selected |= {0, N - 1}
        selected_sorted = np.sort(list(selected))

        results[budget] = UniversalSchedule(
            timesteps=timestep_indices[selected_sorted],
            retention_freq=retention_freq,
            nfe=len(selected_sorted),
            mean_alpha=mean_alpha,
            std_alpha=std_alpha,
        )

    return results


def adaptive_schedule_for_sample(
    trajectory: np.ndarray,
    k: int = 2,
    target_alpha: float = 1e-3,
) -> PruneResult:
    """
    Compute a sample-specific adaptive schedule.
    This is for the per-sample mode (Experiment 6).

    Returns the PruneResult which contains the retained indices.
    """
    tau, result = threshold_search(trajectory, k=k, target_alpha=target_alpha)
    return result
