"""
Collect all experiment outputs and print summary.

Usage: python scripts/collect.py
"""

import json
from pathlib import Path


def main():
    print("=" * 60)
    print("GeoSPRINT Results Summary")
    print("=" * 60)

    figs = sorted(Path("figures").glob("*.pdf"))
    print(f"\nFigures ({len(figs)}):")
    for f in figs:
        print(f"  ✓ {f}")

    for exp, path in [("Exp 1 — FID vs NFE", "results/exp1/exp1_results.json"),
                       ("Exp 2 — Schedule zones", "results/exp2/exp2_analysis.json"),
                       ("Exp 3 — α_traj by scheduler", "results/exp3/exp3_results.json"),
                       ("Exp 4 — DPM++ composability", "results/exp4/exp4_results.json"),
                       ("Exp 5 — Convergence", "results/exp5/exp5_results.json"),
                       ("Exp 6 — Per-sample NFE", "results/exp6/exp6_stats.json")]:
        p = Path(path)
        if p.exists():
            data = json.load(open(p))
            print(f"\n{exp}:")
            if isinstance(data, list):
                for r in data:
                    alpha_str = f"α={r.get('alpha', 0):.2e}" if r.get('alpha') else ""
                    print(f"  {r['method']:16s} NFE={r['nfe']:3d}  FID={r['fid']:.2f}  {alpha_str}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        print(f"  {k}: {v}")
                    else:
                        print(f"  {k}: {v}")

    tex = Path("results/exp1/exp1_table.tex")
    if tex.exists():
        print(f"\nLaTeX table (paste into paper):")
        print(open(tex).read())

    print("\nDone. Copy figures/*.pdf into your paper.")


if __name__ == "__main__":
    main()
