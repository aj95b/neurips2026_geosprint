# GeoSPRINT

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
