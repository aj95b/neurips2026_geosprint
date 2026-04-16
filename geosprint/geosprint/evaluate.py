"""
GeoSPRINT Evaluation
=====================
FID, Inception Score, projection score statistics, NFE tracking.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Evaluation results for a single method at a single NFE budget."""
    method: str
    nfe: int
    fid: float = 0.0
    fid_std: float = 0.0
    inception_score: float = 0.0
    is_std: float = 0.0
    mean_alpha: float = 0.0
    wall_clock_sec: float = 0.0
    num_samples: int = 0
    per_sample_nfe: Optional[np.ndarray] = None


def compute_fid(
    generated: np.ndarray,
    reference_stats: dict,
    device: str = "cuda",
) -> float:
    """
    Compute FID between generated samples and reference statistics.

    Parameters
    ----------
    generated : np.ndarray
        Generated images, shape (N, C, H, W) or (N, H, W, C).
    reference_stats : dict
        Pre-computed {'mu': np.ndarray, 'sigma': np.ndarray} from the
        training set. Use compute_reference_stats() to generate.

    Returns
    -------
    float
        FID score (lower is better).
    """
    try:
        from pytorch_fid.fid_score import calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3
        import torch

        # Get Inception features
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device).eval()

        if isinstance(generated, np.ndarray):
            generated = torch.from_numpy(generated).float()
        if generated.shape[-1] in [1, 3]:  # NHWC → NCHW
            generated = generated.permute(0, 3, 1, 2)
        generated = generated.to(device)

        # Normalize to [0, 1] if needed
        if generated.max() > 1.0:
            generated = generated / 255.0

        # Resize to 299x299 for Inception
        generated = torch.nn.functional.interpolate(generated, size=(299, 299), mode='bilinear')

        features = []
        batch_size = 64
        for i in range(0, len(generated), batch_size):
            batch = generated[i:i+batch_size]
            with torch.no_grad():
                feat = model(batch)[0].squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())

        features = np.concatenate(features, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        fid = calculate_frechet_distance(
            mu, sigma,
            reference_stats['mu'], reference_stats['sigma']
        )
        return fid

    except ImportError:
        print("WARNING: pytorch_fid not installed. Run: pip install pytorch-fid")
        print("Returning dummy FID=0.0")
        return 0.0


def compute_reference_stats(
    dataset_path: str,
    device: str = "cuda",
) -> dict:
    """
    Pre-compute Inception statistics for a reference dataset.
    Save with np.savez() and load for repeated FID evaluations.
    """
    try:
        from pytorch_fid.fid_score import compute_statistics_of_path

        mu, sigma = compute_statistics_of_path(
            dataset_path,
            model=None,
            batch_size=64,
            dims=2048,
            device=device,
        )
        return {'mu': mu, 'sigma': sigma}
    except ImportError:
        print("WARNING: pytorch_fid not installed.")
        return {'mu': np.zeros(2048), 'sigma': np.eye(2048)}


def compute_inception_score(
    generated: np.ndarray,
    splits: int = 10,
    device: str = "cuda",
) -> tuple:
    """
    Compute Inception Score.

    Returns
    -------
    (mean_IS, std_IS)
    """
    try:
        import torch
        from torchvision.models import inception_v3
        import torch.nn.functional as F

        model = inception_v3(pretrained=True, transform_input=False).to(device).eval()

        if isinstance(generated, np.ndarray):
            generated = torch.from_numpy(generated).float()
        if generated.shape[-1] in [1, 3]:
            generated = generated.permute(0, 3, 1, 2)
        generated = generated.to(device)

        if generated.max() > 1.0:
            generated = generated / 255.0

        generated = F.interpolate(generated, size=(299, 299), mode='bilinear')

        preds = []
        batch_size = 64
        for i in range(0, len(generated), batch_size):
            batch = generated[i:i+batch_size]
            with torch.no_grad():
                logits = model(batch)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Split and compute IS
        scores = []
        chunk_size = len(preds) // splits
        for i in range(splits):
            chunk = preds[i*chunk_size:(i+1)*chunk_size]
            p_y = np.mean(chunk, axis=0, keepdims=True)
            kl = chunk * (np.log(chunk + 1e-10) - np.log(p_y + 1e-10))
            scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

        return float(np.mean(scores)), float(np.std(scores))

    except ImportError:
        print("WARNING: torchvision not installed.")
        return 0.0, 0.0


def projection_score_statistics(
    results: List,
) -> Dict[str, float]:
    """Aggregate projection score stats across a batch of PruneResults."""
    alphas = [r.projection_score for r in results]
    reductions = [r.reduction_pct for r in results]

    return {
        'alpha_mean': np.mean(alphas),
        'alpha_std': np.std(alphas),
        'alpha_median': np.median(alphas),
        'reduction_mean': np.mean(reductions),
        'reduction_std': np.std(reductions),
    }


def nfe_statistics(per_sample_nfe: np.ndarray) -> Dict[str, float]:
    """Compute NFE distribution statistics for adaptive scheduling."""
    return {
        'nfe_mean': np.mean(per_sample_nfe),
        'nfe_std': np.std(per_sample_nfe),
        'nfe_median': np.median(per_sample_nfe),
        'nfe_q25': np.percentile(per_sample_nfe, 25),
        'nfe_q75': np.percentile(per_sample_nfe, 75),
        'nfe_min': np.min(per_sample_nfe),
        'nfe_max': np.max(per_sample_nfe),
    }


def format_results_table(
    results: List[EvalResult],
    caption: str = "FID vs NFE comparison",
) -> str:
    """
    Format evaluation results as a LaTeX booktabs table.
    Ready to paste into the paper.
    """
    lines = [
        f"% {caption}",
        r"\begin{table}[ht]",
        f"  \\caption{{{caption}}}",
        r"  \centering",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Method & NFE & FID $\downarrow$ & $\alpha_{\mathrm{traj}}$ \\",
        r"    \midrule",
    ]

    for r in sorted(results, key=lambda x: (x.nfe, x.fid)):
        alpha_str = f"{r.mean_alpha:.1e}" if r.mean_alpha > 0 else "---"
        lines.append(
            f"    {r.method} & {r.nfe} & "
            f"{r.fid:.2f} $\\pm$ {r.fid_std:.2f} & {alpha_str} \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
