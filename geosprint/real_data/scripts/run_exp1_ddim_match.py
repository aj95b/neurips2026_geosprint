"""
Exp1 Final: Generate matching DDIM baselines + merge ALL exp1 results.

Reads:
  results/exp1/exp1_results.json          (GeoSPRINT raw + DDIM at those Ks)
  results/exp1/exp1_results_with_gap.json (adds GeoSPRINT+gap)

Does:
  1. Generates DDIM at every GeoSPRINT+gap NFE that's missing
  2. Merges everything into one table
  3. Saves final JSON, LaTeX table, and plot

Outputs:
  results/exp1/exp1_final.json
  results/exp1/exp1_final_table.tex
  figures/exp1_final.pdf

Usage: python3 scripts/run_exp1_ddim_match.py
"""

import json, numpy as np, torch, shutil
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

_model_cache = {}

def get_model(device):
    if device not in _model_cache:
        from diffusers import UNet2DModel
        m = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
        m.eval()
        _model_cache[device] = m
    return _model_cache[device]

def get_alphas(device):
    from diffusers import DDIMScheduler
    return DDIMScheduler.from_pretrained("google/ddpm-cifar10-32").alphas_cumprod.to(device)

def to_images(latents):
    return ((latents.clamp(-1,1)+1)/2*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

def ddim_step(x, eps, a_t, a_prev):
    x0 = ((x - (1-a_t).sqrt()*eps) / a_t.sqrt()).clamp(-1,1)
    return a_prev.sqrt()*x0 + (1-a_prev).sqrt()*eps

def generate(ts_use, num, device, bs=64, desc="Gen"):
    model = get_model(device)
    alphas = get_alphas(device)
    valid = sorted([int(t) for t in ts_use if 0<=int(t)<1000], reverse=True)
    samples = []
    for bi in tqdm(range((num+bs-1)//bs), desc=desc):
        b = min(bs, num-bi*bs)
        g = torch.Generator(device=device).manual_seed(10000+bi)
        z = torch.randn(b,3,32,32, generator=g, device=device)
        for i, t in enumerate(valid):
            with torch.no_grad():
                eps = model(z, torch.tensor(t, device=device)).sample
            a_t = alphas[t]
            a_prev = alphas[valid[i+1]] if i+1<len(valid) else torch.tensor(1.0, device=device)
            z = ddim_step(z, eps, a_t, a_prev)
        samples.append(to_images(z))
    return np.concatenate(samples)[:num]

def save_imgs(imgs, d):
    from PIL import Image
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        Image.fromarray(im).save(d/f"{i:05d}.png")

def compute_fid(gd, rd):
    try:
        from cleanfid import fid as cf
        return cf.compute_fid(str(gd), str(rd))
    except ImportError:
        import subprocess
        r = subprocess.run(["python","-m","pytorch_fid",str(rd),str(gd)],
                           capture_output=True, text=True)
        for l in r.stdout.split('\n'):
            if 'FID' in l: return float(l.split(':')[-1].strip())
        return float('nan')


def main():
    rdir = Path("results/exp1")
    sdir = rdir / "samples"
    fdir = Path("figures")
    fdir.mkdir(exist_ok=True)
    ref_dir = Path("results/fid_stats/cifar10_train_images")

    # ── Step 1: Load all existing results ──
    print("[Step 1] Loading existing results...")

    all_records = []
    seen = set()  # (method, nfe) to deduplicate

    for fname in ["exp1_results.json", "exp1_results_with_gap.json"]:
        fpath = rdir / fname
        if fpath.exists():
            data = json.load(open(fpath))
            for r in data:
                key = (r['method'], r['nfe'])
                if key not in seen and not np.isnan(r.get('fid', float('nan'))):
                    all_records.append(r)
                    seen.add(key)
            print(f"  Loaded {len(data)} records from {fname}")
        else:
            print(f"  WARNING: {fname} not found")

    print(f"  Total unique records: {len(all_records)}")

    # ── Step 2: Find missing DDIM matches ──
    gap_nfes = set(r['nfe'] for r in all_records if r['method'] == 'GeoSPRINT+gap')
    ddim_nfes = set(r['nfe'] for r in all_records if r['method'].startswith('DDIM'))
    missing = sorted(gap_nfes - ddim_nfes)

    if missing:
        print(f"\n[Step 2] Generating DDIM at missing NFEs: {missing}")
        from diffusers import DDIMScheduler

        for K in missing:
            print(f"\n  --- DDIM K={K} ---")
            d = sdir / f"ddim_match_K{K}"
            if not d.exists():
                s = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
                s.set_timesteps(K)
                ts = s.timesteps.cpu().numpy().tolist()
                imgs = generate(ts, 10000, "cuda", 64, f"DDIM K={K}")
                save_imgs(imgs, d)
                del imgs

            fid_score = compute_fid(d, ref_dir)
            print(f"    DDIM K={K}: FID={fid_score:.2f}")
            all_records.append({'method': 'DDIM', 'nfe': K, 'fid': fid_score, 'alpha': 0})
            torch.cuda.empty_cache()
    else:
        print("\n[Step 2] All DDIM matches already exist.")

    # ── Step 3: Build final table ──
    print("\n[Step 3] Building final results...")

    # Sort by NFE then method
    all_records.sort(key=lambda x: (x['nfe'], x['method']))

    # Save JSON
    with open(rdir / "exp1_final.json", 'w') as f:
        json.dump(all_records, f, indent=2)

    # Print table
    print("\n" + "=" * 75)
    print("EXPERIMENT 1: COMPLETE RESULTS")
    print("=" * 75)
    print(f"\n  {'Method':<25s} {'NFE':>5s} {'FID':>8s} {'α_traj':>12s} {'vs DDIM':>10s}")
    print(f"  {'-'*65}")

    # Group by NFE for side-by-side comparison
    from collections import defaultdict
    by_nfe = defaultdict(dict)
    for r in all_records:
        by_nfe[r['nfe']][r['method']] = r

    for nfe in sorted(by_nfe.keys()):
        group = by_nfe[nfe]
        # Find DDIM FID at this NFE for comparison
        ddim_fid = None
        for mname, r in group.items():
            if mname.startswith('DDIM'):
                ddim_fid = r['fid']
                break

        for method in ['DDIM-200 (baseline)', 'DDIM-50', 'DDIM', 'GeoSPRINT', 'GeoSPRINT+gap']:
            if method in group:
                r = group[method]
                astr = f"{r['alpha']:.1e}" if r.get('alpha', 0) > 0 else "---"
                if method in ('GeoSPRINT', 'GeoSPRINT+gap') and ddim_fid is not None:
                    diff = r['fid'] - ddim_fid
                    diff_str = f"{diff:+.2f}"
                elif method.startswith('DDIM') and not method.startswith('DDIM-'):
                    diff_str = "ref"
                else:
                    diff_str = "---"
                print(f"  {method:<25s} {nfe:>5d} {r['fid']:>8.2f} {astr:>12s} {diff_str:>10s}")
        if len(group) > 1:
            print()

    # Save LaTeX table
    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("  \\caption{CIFAR-10: GeoSPRINT step-budget oracle. DDIM-200 is the zero-loss baseline (FID=12.94). GeoSPRINT decides the step budget K; DDIM at the same K is shown for comparison.}")
    latex.append("  \\centering")
    latex.append("  \\begin{tabular}{llccc}")
    latex.append("    \\toprule")
    latex.append("    Method & NFE & FID $\\downarrow$ & $\\alpha_{\\mathrm{traj}}$ & $\\Delta$ vs DDIM \\\\")
    latex.append("    \\midrule")

    for nfe in sorted(by_nfe.keys()):
        group = by_nfe[nfe]
        ddim_fid = None
        for mname, r in group.items():
            if mname.startswith('DDIM') and not mname.startswith('DDIM-'):
                ddim_fid = r['fid']

        for method in ['DDIM-200 (baseline)', 'DDIM-50', 'DDIM', 'GeoSPRINT', 'GeoSPRINT+gap']:
            if method in group:
                r = group[method]
                astr = f"{r['alpha']:.1e}" if r.get('alpha', 0) > 0 else "---"
                if method in ('GeoSPRINT', 'GeoSPRINT+gap') and ddim_fid:
                    diff = r['fid'] - ddim_fid
                    dstr = f"{diff:+.2f}"
                else:
                    dstr = "---"
                latex.append(f"    {method} & {nfe} & {r['fid']:.2f} & {astr} & {dstr} \\\\")
        latex.append("    \\midrule")

    latex[-1] = "    \\bottomrule"
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")

    tex_str = "\n".join(latex)
    with open(rdir / "exp1_final_table.tex", 'w') as f:
        f.write(tex_str)
    print(f"\nLaTeX table saved to results/exp1/exp1_final_table.tex")

    # ── Step 4: Plot ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: FID vs NFE for all three methods
    styles = {
        'GeoSPRINT':     {'m': 'o', 'c': '#534AB7', 'ls': '-',  'label': 'GeoSPRINT (raw)'},
        'GeoSPRINT+gap': {'m': 'D', 'c': '#D85A30', 'ls': '-',  'label': 'GeoSPRINT+gap'},
        'DDIM':          {'m': 's', 'c': '#888780', 'ls': '--', 'label': 'DDIM (uniform)'},
    }
    for method, sty in styles.items():
        pts = sorted([r for r in all_records if r['method'] == method and not np.isnan(r['fid'])],
                     key=lambda x: x['nfe'])
        if pts:
            ax1.plot([r['nfe'] for r in pts], [r['fid'] for r in pts],
                     f'{sty["m"]}{sty["ls"]}', label=sty['label'], color=sty['c'],
                     markersize=6, linewidth=1.5)

    # Add baseline reference lines
    baselines = [r for r in all_records if r['method'].startswith('DDIM-')]
    for bl in baselines:
        ax1.axhline(bl['fid'], color='#1D9E75', linestyle=':', linewidth=0.8, alpha=0.7)
        ax1.annotate(f"{bl['method']} ({bl['fid']:.1f})",
                     xy=(max(r['nfe'] for r in all_records), bl['fid']),
                     fontsize=7, color='#1D9E75', va='bottom')

    ax1.set_xlabel("NFE (number of function evaluations)")
    ax1.set_ylabel("FID (lower is better)")
    ax1.set_title("(a) FID vs NFE")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Right: FID difference (GeoSPRINT - DDIM) at matched NFEs
    geo_raw = {r['nfe']: r['fid'] for r in all_records if r['method'] == 'GeoSPRINT'}
    geo_gap = {r['nfe']: r['fid'] for r in all_records if r['method'] == 'GeoSPRINT+gap'}
    ddim_all = {r['nfe']: r['fid'] for r in all_records if r['method'] == 'DDIM'}

    # GeoSPRINT raw vs DDIM
    shared_raw = sorted(set(geo_raw.keys()) & set(ddim_all.keys()))
    if shared_raw:
        diffs_raw = [geo_raw[k] - ddim_all[k] for k in shared_raw]
        ax2.bar([k - 1.5 for k in shared_raw], diffs_raw, width=3,
                color='#534AB7', alpha=0.7, label='GeoSPRINT raw − DDIM')

    # GeoSPRINT+gap vs DDIM
    shared_gap = sorted(set(geo_gap.keys()) & set(ddim_all.keys()))
    if shared_gap:
        diffs_gap = [geo_gap[k] - ddim_all[k] for k in shared_gap]
        ax2.bar([k + 1.5 for k in shared_gap], diffs_gap, width=3,
                color='#D85A30', alpha=0.7, label='GeoSPRINT+gap − DDIM')

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("NFE")
    ax2.set_ylabel("ΔFID (negative = GeoSPRINT wins)")
    ax2.set_title("(b) FID difference vs DDIM at same NFE")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(fdir / "exp1_final.pdf", dpi=200, bbox_inches='tight')
    print(f"Plot saved to figures/exp1_final.pdf")
    print(f"JSON saved to results/exp1/exp1_final.json")


if __name__ == "__main__":
    main()
