<h1 align="center">GeoSPRINT</h1>
<p align="center">
  <b>Geometric Step Pruning for Inference in Diffusion Trajectories</b><br>
  <i>A training-free way to build non-uniform diffusion sampling schedules from trajectory geometry.</i>
</p>

<p align="center">
  <a href="#"><img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS-2026-8A2BE2"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg"></a>
</p>

---

## TL;DR

Diffusion sampling is slow because it takes many sequential neural function evaluations (NFEs). GeoSPRINT looks at the **geometry of the denoising trajectory** — which steps actually bend the path and which just trace a straight line — and places sampling steps where they matter. It requires **no retraining, no distillation, and no changes to the model**. A short offline pass over a handful of reference trajectories yields a schedule you reuse for all future samples.

On three models spanning pixel- and latent-space diffusion, GeoSPRINT consistently improves FID over uniform DDIM at matched NFE, and — above a crossover budget of ~30 steps — a first-order DDIM solver on a GeoSPRINT schedule matches or beats the second-order DPM-Solver++ on its own default schedule.

> **Accepted at NeurIPS 2026.** Single-author work by Arpita Joshi (The Scripps Research Institute).

---

## Why it works

A denoising trajectory `{z_T, …, z_0}` is an ordered sequence of latent states. In smooth regions the score changes slowly and consecutive states trace a nearly **straight line** — those steps are geometrically redundant. Where the model resolves fine structure the trajectory **curves sharply** — every step there carries new directional information.

GeoSPRINT detects this with a **hyperplanarity test**: for a sliding window of retained points, it measures the residual distance of the next point from their affine span (via a thin QR factorization, `O(d·k)` per step). Averaging retained steps over a few reference trajectories gives a **curvature-density profile** `w(t)`. That profile is then blended with log-SNR spacing to build the final schedule:

```
ρ(t) = (1 − β) · ρ_logSNR(t)  +  β · ρ_curv(t)
```

Steps are placed at uniform quantiles of the cumulative density. Neither piece alone is enough — log-SNR spacing ignores geometry, pure curvature spacing leaves coverage gaps. The blend (with **β = 0.6**, a broad and stable optimum) captures both solver-error structure and trajectory curvature.

We also introduce **α_traj**, the *trajectory projection score*: the fraction of trajectory variance lost to pruning. It is a training-free measure of how straight a trajectory is (it drops ~450× from stochastic DDPM to deterministic DDIM), and it doubles as a diagnostic for trajectory rectification quality.

---

## Results

FID (↓), mean over 3 seeds. GeoSPRINT uses LogSNR+curvature scheduling (β = 0.6).

**CIFAR-10 (32×32)** — baseline DDIM-200 FID = 12.65

| NFE | GeoSPRINT | DDIM  | Δ vs DDIM |
|----:|:---------:|:-----:|:---------:|
| 49  | **15.95** | 16.82 | −0.87 |
| 60  | **15.25** | 15.95 | −0.70 |
| 70  | **14.80** | 15.69 | −0.89 |
| 80  | **14.38** | 15.30 | −0.92 |
| 89  | **14.10** | 15.17 | −1.07 |

*GeoSPRINT at 60 steps matches DDIM at 80 — a 25% NFE reduction.*

**LSUN Church (256×256)**

| NFE | GeoSPRINT | DDIM |
|----:|:---------:|:----:|
| 52  | **1.26** | 1.48 |
| 74  | **0.87** | 1.72 |
| 79  | **0.79** | 1.43 |

**Stable Diffusion v1.5 (512×512 latent)** — FID vs 50-step DDIM reference (relative trajectory fidelity)

| NFE | GeoSPRINT | DDIM | DPM-Solver++ |
|----:|:---------:|:----:|:------------:|
| 15  | 7.91 | 8.72 | **6.07** |
| 24  | 4.60 | 5.92 | **4.37** |
| 29  | **4.01** | 4.92 | 4.07 |
| 34  | **3.66** | 4.23 | 4.01 |
| 44  | **3.45** | 5.38 | 3.95 |

**The crossover.** Higher-order integration (DPM-Solver++) wins at low NFE; schedule quality (GeoSPRINT) wins above ~30 steps. This crossover is consistent across CIFAR-10 and SD v1.5. GeoSPRINT is a *schedule*, so it **composes with any solver** rather than competing with them.

---

## Installation

```bash
git clone https://github.com/<your-username>/geosprint.git
cd geosprint
pip install -e .

# For the full experiment suite (diffusion models, FID):
pip install torch torchvision diffusers accelerate transformers safetensors clean-fid scipy matplotlib tqdm

# Verify the core algorithms:
python tests/test_core.py     # 7 tests
```

Core requirements are just `numpy` and `scipy`; the heavy dependencies are only needed to reproduce the experiments.

---

## Quick start

```python
import numpy as np
from geosprint import prune_trajectory, threshold_search

# A denoising trajectory: (N steps) x (d latent dims)
trajectory = np.load("my_trajectory.npy")

# Prune geometrically redundant steps at a fixed threshold
result = prune_trajectory(trajectory, k=2, threshold=0.01)
print(f"Retained {len(result.retained_indices)}/{len(trajectory)} steps")
print(f"alpha_traj = {result.projection_score:.4e}")

# ...or search for a threshold that hits a target straightness score
tau, result = threshold_search(trajectory, k=2, target_alpha=1e-3)
```

Building a schedule from reference trajectories:

```python
from geosprint.schedule import compute_retention_frequency, logsnr_curvature_schedule

w = compute_retention_frequency(reference_trajectories, pool_timesteps, target_K=50)
schedule = logsnr_curvature_schedule(alphas_cumprod, w, pool_timesteps, K=50, blend=0.6)
```

---

## Reproducing the paper

```bash
# 1. Reference FID statistics (downloads CIFAR-10)
python scripts/setup_fid_stats.py

# 2. CIFAR-10: FID vs NFE, DPM-Solver++ comparison, ablations
python scripts/run_exp1.py         --device cuda --num_samples 10000
python scripts/run_exp4.py         --device cuda --num_samples 10000
python scripts/run_exp4_higher.py  --device cuda --num_samples 10000
python scripts/test_blend_sweep.py --device cuda --num_samples 10000
python scripts/run_exp3.py         --device cuda      # alpha_traj diagnostic
python scripts/run_exp5.py                            # reference-set convergence (CPU)
python scripts/run_exp6.py                            # per-sample complexity (CPU)

# 3. Higher resolution
python scripts/run_church256_logsnr.py --device cuda:0 --num_samples 10000
python scripts/run_sd15_logsnr.py      --device cuda:0 --num_samples 5000
```

SLURM submission scripts for each dataset live in `slurm/`.

> **Environment tips (hard-won):** set `export PYTHONNOUSERSITE=1` and `export HF_HUB_OFFLINE=1` to avoid package-shadowing and HuggingFace disconnects on HPC nodes. DPM-Solver++ needs a fresh scheduler instance per batch to reset internal state, and non-uniform DDIM requires the manual update formula (the library's `DDIMScheduler.step()` assumes uniform spacing).

---

## Repository layout

```
geosprint/
├── geosprint/                 # Core library (numpy/scipy only)
│   ├── core.py               # Hyperplanarity test, pruning, alpha_traj
│   ├── schedule.py           # LogSNR + curvature schedule construction
│   └── evaluate.py           # FID / results utilities
├── scripts/                   # Experiment scripts (CIFAR-10, Church, SD v1.5)
├── slurm/                     # Cluster submission scripts
├── tests/                     # Unit tests (7 passing)
├── paper/                     # NeurIPS 2026 paper source + figures
└── setup.py
```

---

## How GeoSPRINT relates to other methods

| Method | Signal used | Training? | Granularity |
|---|---|---|---|
| **DDIM** | none (uniform) | no | fixed schedule |
| **DPM-Solver++ / UniPC** | local truncation error | no | per-step, online |
| **Distillation / Consistency** | learned | **yes** | per-budget model |
| **GeoSPRINT** | **global trajectory geometry** | **no** | fixed schedule, composes with any solver |

GeoSPRINT is complementary to high-order solvers: it decides *where* to step; they decide *how* to step between those points.

---

## Citation

```bibtex
@inproceedings{joshi2026geosprint,
  title     = {GeoSPRINT: Geometric Step Pruning for Inference in Diffusion Trajectories},
  author    = {Joshi, Arpita},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

Building on the geometric instance-reduction method it generalizes:

```bibtex
@article{joshi2020instance,
  title   = {A Novel Data Instance Reduction Technique Using Linear Feature Reduction},
  author  = {Joshi, Arpita and Haspel, Nurit},
  journal = {Journal of Artificial Intelligence and Systems},
  volume  = {2},
  pages   = {191--206},
  year    = {2020}
}
```

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Supported by NIH grant 1R01DA063157-01 and the High Performance Computing group at The Scripps Research Institute, San Diego, CA.# GeoSPRINT

**Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories**

---

## Project Structure

```
geosprint/
├── geosprint/           # Core library (no GPU needed)
│   ├── core.py          # Hyperplanarity test, pruning, projection score
│   ├── schedule.py      # Universal + adaptive schedule extraction
│   ├── sampler.py       # Apply schedules to generate samples
│   ├── trajectories.py  # Record denoising paths from pretrained models
│   └── evaluate.py      # FID, IS, metrics, LaTeX table generator
├── scripts/             # Experiment runners
│   ├── setup_fid_stats.py  # Download CIFAR-10, compute Inception stats
│   ├── run_exp1.py         # FID vs NFE Pareto frontier (GPU)
│   ├── run_exp2.py         # Schedule w(t) analysis (CPU)
│   ├── run_exp3.py         # α_traj across schedulers (GPU)
│   ├── run_exp4.py         # GeoSPRINT + DPM-Solver++ composability (GPU)
│   ├── run_exp5.py         # Reference set convergence (CPU)
│   ├── run_exp6.py         # Per-sample NFE distribution (CPU)
│   └── collect.py          # Collect all results
├── tests/
│   └── test_core.py     # 7 verification tests (CPU, 10 sec)
├── slurm/
│   └── submit_all.sh    # HPC submission (gpu + highmem partitions)
├── figures/             # Output PDFs (paper-ready)
├── results/             # Output JSON + numpy
└── setup.py
```

---

## End-to-End Instructions

### Option A: Laptop / Single GPU

```bash
# ─── Step 0: Install (2 min) ───
cd geosprint
pip install -e .
pip install diffusers accelerate transformers safetensors
pip install clean-fid pytorch-fid torchvision
pip install matplotlib scipy tqdm

# Verify core algorithm (no GPU, 10 seconds)
python tests/test_core.py


# ─── Step 1: FID reference stats (10 min, downloads CIFAR-10) ───
python scripts/setup_fid_stats.py


# ─── Step 2: Experiment 1 — FID vs NFE Pareto (4-6 hrs GPU) ───
python scripts/run_exp1.py --device cuda --num_samples 10000

# If you need to restart (reuses saved trajectories):
python scripts/run_exp1.py --device cuda --num_samples 10000 --skip_record


# ─── Step 3: Experiment 3 — α_traj diagnostic (5 min GPU) ───
python scripts/run_exp3.py --device cuda


# ─── Step 4: Experiment 4 — GeoSPRINT + DPM-Solver++ (2-3 hrs GPU) ───
# Requires exp1 trajectories. Tests composability.
python scripts/run_exp4.py --device cuda --num_samples 10000


# ─── Step 5: CPU experiments (30 min total, no GPU) ───
python scripts/run_exp2.py
python scripts/run_exp5.py
python scripts/run_exp6.py


# ─── Step 6: Collect all results ───
python scripts/collect.py
```

### Option B: HPC Cluster (SLURM)

```bash
cd geosprint
bash slurm/submit_all.sh
```

This submits 5 chained jobs:
```
Job 0 (setup, 30 min)
  ├── Job 1 (exp1, 8 hrs GPU)     ← the big one
  │     └── Job 2 (exp2+5+6, 1 hr CPU)
  └── Job 3 (exp3, 3 hrs GPU)     ← parallel with Job 1
        └── Job 4 (collect)
```

Monitor: `squeue -u $USER` and `tail -f logs/exp1_<JOBID>.out`

---

## What Each Experiment Produces

| Exp | Figure | Key Result |
|-----|--------|------------|
| 1 | `figures/exp1_pareto.pdf` | FID vs NFE curves for GeoSPRINT, DDIM, DPM-Solver++ |
| 2 | `figures/exp2_schedule.pdf` | Where GeoSPRINT places steps (retention heatmap) |
| 3 | `figures/exp3_rectification.pdf` | α_traj: DDPM >> DDIM >> DPM++ (confirms Corollary) |
| 4 | `figures/exp4_compose.pdf` | DPM++ with GeoSPRINT schedule vs default schedule |
| 5 | `figures/exp5_convergence.pdf` | Schedule converges at B ≈ 50-100 reference trajectories |
| 6 | `figures/exp6_adaptive.pdf` | Per-sample NFE histogram (complex samples get more steps) |

All results also saved as JSON in `results/exp*/` and a LaTeX table in `results/exp1/exp1_table.tex`.

---

## Troubleshooting

**`torch.xpu` AttributeError**: Your PyTorch is too old. Run:
```bash
pip install --upgrade torch torchvision
pip install --upgrade diffusers accelerate
```

**DPM-Solver++ IndexError**: Already fixed in this codebase. The fixes:
1. Manual `DPMSolverMultistepScheduler(...)` constructor (bypasses `from_pretrained` ignoring kwargs)
2. `solver_order=2` + `lower_order_final=True` (prevents last-step overflow)
3. Fresh scheduler per trajectory (resets internal `step_index` counter)

**Exp1 `--skip_record`**: Reuses trajectories from a previous run. Safe to use after any crash in Phase 3+.

---

## API Quick Reference

```python
from geosprint import prune_trajectory, threshold_search, extract_universal_schedule

# Prune a single trajectory
result = prune_trajectory(trajectory, k=2, threshold=0.01)
print(f"Removed {result.reduction_pct:.1f}%, α={result.projection_score:.2e}")

# Auto-find optimal threshold
tau, result = threshold_search(trajectory, k=2, target_alpha=1e-3)

# Extract universal schedule from B reference trajectories
schedule = extract_universal_schedule(trajectories, timesteps, nfe_budget=20)
```
