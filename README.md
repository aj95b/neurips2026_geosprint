# GeoSPRINT

**Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories**

A training-free, model-agnostic framework for accelerating diffusion model sampling.

## Quickstart

```bash
# Install
pip install -e .

# Verify core algorithm (CPU only, no models needed)
python tests/test_core.py

# Run Experiment 1 demo (synthetic trajectories)
python -m experiments.exp1_pareto --ref_batch 50 --output figures/exp1_demo.pdf
```

## Project Structure

```
geosprint/
├── geosprint/              # Core library
│   ├── core.py             # Hyperplanarity test, pruning, projection score
│   ├── schedule.py         # Universal + adaptive schedule extraction
│   ├── sampler.py          # Apply schedules to generate samples
│   ├── trajectories.py     # Record denoising paths from pretrained models
│   └── evaluate.py         # FID, IS, projection score stats, LaTeX tables
├── experiments/            # One script per paper experiment
│   └── exp1_pareto.py      # Exp 1: FID vs NFE Pareto frontiers
├── tests/
│   └── test_core.py        # Core algorithm verification (7 tests)
├── figures/                # Output plots and tables
├── configs/                # Experiment configs (YAML)
└── setup.py
```

## Experiment Execution Order

Run in this order to build results for the paper:

| # | Experiment | Script | GPU needed | Approx time |
|---|-----------|--------|-----------|-------------|
| 0 | Verify core | `tests/test_core.py` | No | 10 sec |
| 1 | FID vs NFE Pareto | `experiments/exp1_pareto.py` | Yes | 6-12 hrs |
| 2 | Schedule w(t) analysis | `experiments/exp2_schedule.py` | No* | 30 min |
| 3 | α_traj rectification diagnostic | `experiments/exp3_rectification.py` | Yes | 2-4 hrs |
| 4 | GeoSPRINT + DPM-Solver++ | `experiments/exp4_compose.py` | Yes | 4-8 hrs |
| 5 | Reference batch size B | `experiments/exp5_ref_convergence.py` | No* | 1 hr |
| 6 | Per-sample NFE distribution | `experiments/exp6_adaptive.py` | Yes | 4-8 hrs |
| 7 | Domain transfer (scRNA) | `experiments/exp7_domain.py` | Yes | 2-4 hrs |

*Experiments 2 and 5 operate on saved trajectories from Experiment 1.

## Connecting to Real Models

Replace the synthetic trajectory generation in `exp1_pareto.py` with:

### CIFAR-10 (EDM)
```python
# pip install edm  (or clone https://github.com/NVlabs/edm)
from geosprint.trajectories import record_trajectory_edm
model = load_edm_model('edm-cifar10-32x32-uncond-vp.pkl')
sigma_schedule = edm_sigma_schedule(num_steps=1000)
bundle = record_trajectory_edm(model, sigma_schedule, batch_size=100)
```

### Stable Diffusion v1.5 (HuggingFace diffusers)
```python
from diffusers import StableDiffusionPipeline
from geosprint.trajectories import record_trajectory_diffusers

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

bundle = record_trajectory_diffusers(
    pipe, prompt="a photo of a cat", num_inference_steps=50, batch_size=100
)
```

### ImageNet 64×64 (ADM / guided-diffusion)
```python
# Clone https://github.com/openai/guided-diffusion
from geosprint.trajectories import record_trajectory_generic

def denoise_fn(z, t):
    return guided_diffusion_step(model, z, t)

traj = record_trajectory_generic(denoise_fn, initial_noise, timesteps)
```

## Key API

```python
from geosprint import (
    prune_trajectory,        # Single trajectory → PruneResult
    multi_level_prune,       # Progressive L1→L2→... pruning
    threshold_search,        # Auto-find optimal τ
    extract_universal_schedule,  # B trajectories → UniversalSchedule
    sweep_nfe_budgets,       # Multiple NFE budgets at once
    projection_score,        # α_traj computation
    format_results_table,    # → LaTeX booktabs table
)

# Core usage
result = prune_trajectory(trajectory, k=2, threshold=0.01)
print(f"Removed {result.reduction_pct:.1f}%, α={result.projection_score:.2e}")

# Universal schedule
schedule = extract_universal_schedule(trajectories, timesteps, nfe_budget=20)
print(f"Selected {schedule.nfe} steps, α={schedule.mean_alpha:.2e}")
```

## Citation

```bibtex
@inproceedings{geosprint2026,
  title={GeoSPRINT: Geometric Redundancy-Aware Step Pruning 
         for Inference in Diffusion Trajectories},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```
