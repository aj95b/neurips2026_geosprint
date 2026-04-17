"""
GeoSPRINT Master Runner
=========================
Execute all experiments in order and generate paper figures.

Usage:
    # Run all experiments with synthetic data (no GPU)
    python run_all.py --synthetic

    # Run with real models (GPU required)
    python run_all.py --model edm_cifar10

    # Run specific experiment only
    python run_all.py --only exp3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = {
    'exp1': {
        'script': 'experiments/exp1_pareto.py',
        'desc': 'FID vs NFE Pareto frontiers',
        'gpu': True,
    },
    'exp2': {
        'script': 'experiments/exp2_schedule.py',
        'desc': 'Schedule w(t) analysis',
        'gpu': False,
    },
    'exp3': {
        'script': 'experiments/exp3_rectification.py',
        'desc': 'α_traj rectification diagnostic',
        'gpu': True,
    },
    'exp4': {
        'script': 'experiments/exp4_compose.py',
        'desc': 'GeoSPRINT + DPM-Solver++',
        'gpu': True,
    },
    'exp5': {
        'script': 'experiments/exp5_ref_convergence.py',
        'desc': 'Reference set convergence',
        'gpu': False,
    },
    'exp6': {
        'script': 'experiments/exp6_adaptive.py',
        'desc': 'Per-sample NFE distribution',
        'gpu': False,
    },
}


def run_experiment(name, info, extra_args=None):
    """Run a single experiment as a subprocess."""
    print(f"\n{'='*60}")
    print(f"Running {name}: {info['desc']}")
    print(f"{'='*60}")

    cmd = [sys.executable, '-m', info['script'].replace('/', '.').replace('.py', '')]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
    print(f"\n{name}: {status} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="GeoSPRINT Master Runner")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data (no GPU needed)")
    parser.add_argument("--model", default=None,
                       help="Model identifier for real experiments")
    parser.add_argument("--only", default=None,
                       help="Run only this experiment (e.g., exp3)")
    args = parser.parse_args()

    print("GeoSPRINT Experiment Suite")
    print("=" * 60)
    print(f"Mode: {'synthetic' if args.synthetic else 'real models'}")

    # Ensure output dirs exist
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Select experiments
    if args.only:
        if args.only not in EXPERIMENTS:
            print(f"Unknown experiment: {args.only}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
        to_run = {args.only: EXPERIMENTS[args.only]}
    else:
        to_run = EXPERIMENTS

    # Run
    results = {}
    t_total = time.time()
    for name, info in to_run.items():
        ok = run_experiment(name, info)
        results[name] = ok

    elapsed_total = time.time() - t_total

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {EXPERIMENTS[name]['desc']}")
    print(f"\nTotal time: {elapsed_total:.0f}s")

    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")

    # Check for output figures
    fig_dir = Path("figures")
    figs = list(fig_dir.glob("*.pdf")) + list(fig_dir.glob("*.png"))
    if figs:
        print(f"\nGenerated figures:")
        for f in figs:
            print(f"  {f}")


if __name__ == "__main__":
    main()
