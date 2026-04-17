"""
Experiment 4: GeoSPRINT + DPM-Solver++ Composability
======================================================
Use GeoSPRINT to select WHICH timesteps, then use DPM-Solver++
as the ODE solver at those timesteps (replacing its built-in
logSNR/time-uniform heuristics).

Hypothesis: GeoSPRINT schedule + DPM-Solver++ > DPM-Solver++ alone.

Usage:
    python -m experiments.exp4_compose \
        --model stable-diffusion-v1-5 \
        --output figures/exp4_compose.pdf
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.schedule import extract_universal_schedule
from geosprint.evaluate import EvalResult, format_results_table


def run(args):
    print("Experiment 4: GeoSPRINT + DPM-Solver++ Composability")
    print("=" * 50)

    nfe_budgets = [10, 15, 20]
    results = []

    # PLACEHOLDER: Load model and generate reference trajectories
    # pipe = StableDiffusionPipeline.from_pretrained(args.model).to("cuda")
    # bundle = record_trajectory_diffusers(pipe, num_inference_steps=50, batch_size=100)
    # trajectories = bundle.trajectories

    print("\n  Step 1: Extract GeoSPRINT schedules from reference trajectories")
    print("  Step 2: For each NFE budget, compare:")
    print("    (a) DPM-Solver++ with logSNR schedule (baseline)")
    print("    (b) DPM-Solver++ with time-uniform schedule (baseline)")
    print("    (c) DPM-Solver++ with GeoSPRINT schedule (ours)")
    print("  Step 3: Generate 50K samples with each, compute FID")

    # TEMPLATE for real execution:
    # for nfe in nfe_budgets:
    #     # (a) DPM-Solver++ logSNR
    #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #         pipe.scheduler.config, algorithm_type="dpmsolver++",
    #         solver_order=3, timestep_spacing="logSNR")
    #     samples_a = generate_50k(pipe, nfe)
    #     fid_a = compute_fid(samples_a, ref_stats)
    #
    #     # (b) DPM-Solver++ time-uniform
    #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #         pipe.scheduler.config, timestep_spacing="uniform")
    #     samples_b = generate_50k(pipe, nfe)
    #     fid_b = compute_fid(samples_b, ref_stats)
    #
    #     # (c) DPM-Solver++ with GeoSPRINT timesteps
    #     geo_schedule = schedules[nfe]
    #     pipe.scheduler.set_timesteps(custom_timesteps=geo_schedule.timesteps)
    #     samples_c = generate_50k(pipe, nfe)
    #     fid_c = compute_fid(samples_c, ref_stats)

    for nfe in nfe_budgets:
        results.append(EvalResult(method="DPM++ logSNR", nfe=nfe))
        results.append(EvalResult(method="DPM++ uniform", nfe=nfe))
        results.append(EvalResult(method="DPM++ GeoSPRINT", nfe=nfe))

    table = format_results_table(results, "DPM-Solver++ schedule comparison")
    print("\n" + table)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp4_results.json", 'w') as f:
        json.dump([{'method': r.method, 'nfe': r.nfe, 'fid': r.fid}
                   for r in results], f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output", default="figures/exp4_compose.pdf")
    run(parser.parse_args())
