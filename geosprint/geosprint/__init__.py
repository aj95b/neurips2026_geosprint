"""
GeoSPRINT: Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories
"""

from .core import (
    hyperplanarity_residual,
    prune_trajectory,
    multi_level_prune,
    projection_score,
    threshold_search,
    batch_prune,
    PruneResult,
)
from .schedule import (
    extract_universal_schedule,
    sweep_nfe_budgets,
    adaptive_schedule_for_sample,
    UniversalSchedule,
)
from .evaluate import (
    compute_fid,
    compute_inception_score,
    projection_score_statistics,
    nfe_statistics,
    format_results_table,
    EvalResult,
)

__version__ = "0.1.0"
