"""
Experiment 4: Solver Composability
===================================
Combines existing results:
  - DPM-Solver++ default schedule (from earlier exp4 run)
  - GeoSPRINT LogSNR+curvature with DDIM (from blend sweep)
  - DDIM uniform (from blend sweep)

Also generates any missing data points.

Usage: python scripts/run_exp4.py --device cuda --num_samples 10000
"""

import argparse, json, shutil
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory
from geosprint.evaluate import EvalResult, format_results_table

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

def generate_ddim(ts_use, num, device, bs=64, desc="Gen"):
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

def generate_dpmsolver(num_steps, num, device, bs=64, desc="Gen"):
    from diffusers import DPMSolverMultistepScheduler
    model = get_model(device)
    samples = []
    for bi in tqdm(range((num+bs-1)//bs), desc=desc):
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
            algorithm_type="dpmsolver++", solver_order=2, lower_order_final=True,
        )
        scheduler.set_timesteps(num_steps)
        b = min(bs, num-bi*bs)
        g = torch.Generator(device=device).manual_seed(10000+bi)
        z = torch.randn(b,3,32,32, generator=g, device=device)
        for t in scheduler.timesteps:
            with torch.no_grad():
                z = scheduler.step(model(z, t).sample, t, z).prev_sample
        samples.append(to_images(z))
    return np.concatenate(samples)[:num]

def save_imgs(imgs, d):
    from PIL import Image
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs): Image.fromarray(im).save(d/f"{i:05d}.png")

def compute_fid(gd, rd):
    try:
        from cleanfid import fid as cf; return cf.compute_fid(str(gd), str(rd))
    except ImportError:
        import subprocess
        r = subprocess.run(["python","-m","pytorch_fid",str(rd),str(gd)], capture_output=True, text=True)
        for l in r.stdout.split('\n'):
            if 'FID' in l: return float(l.split(':')[-1].strip())
        return float('nan')

def find_threshold_for_K(traj, target_K, tol=3, max_iter=30):
    tau_lo, tau_hi = 0.0, np.max(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 3
    for _ in range(max_iter):
        tau_mid = (tau_lo + tau_hi) / 2
        result = prune_trajectory(traj, k=2, threshold=tau_mid)
        K = len(result.retained_indices)
        if abs(K - target_K) <= tol: return tau_mid, result
        elif K < target_K: tau_hi = tau_mid
        else: tau_lo = tau_mid
    return tau_mid, result

def compute_retention_frequency(normalized_trajs, pool_timesteps, target_K=50):
    N_traj = len(pool_timesteps) + 1
    counts = np.zeros(N_traj)
    for traj in normalized_trajs:
        tau, res = find_threshold_for_K(traj, target_K, tol=5)
        for idx in res.retained_indices:
            if idx < N_traj: counts[idx] += 1
    return counts[1:len(pool_timesteps)+1]

def logsnr_curvature_schedule(alphas_np, retention_freq, pool_timesteps, K, blend=0.6):
    eps = 1e-8
    logsnr_density = np.ones(len(alphas_np))
    curv_density = np.zeros(len(alphas_np))
    for i, t in enumerate(pool_timesteps):
        t_int = int(t)
        if 0 <= t_int < len(curv_density):
            curv_density[t_int] = retention_freq[i]
    curv_density = gaussian_filter1d(curv_density, sigma=10)
    curv_density = curv_density + 0.01 * curv_density.max()
    combined = (1-blend) * logsnr_density/logsnr_density.sum() + \
               blend * curv_density/curv_density.sum()
    W = np.cumsum(combined)
    W = W / W[-1]
    quantiles = np.linspace(0, 1, K)
    timesteps = []
    for q in quantiles:
        idx = np.searchsorted(W, q)
        idx = min(idx, len(alphas_np)-1)
        timesteps.append(idx)
    timesteps = sorted(set(timesteps), reverse=True)
    if len(timesteps) < K:
        all_t = list(range(999, -1, -1))
        for t in all_t:
            if t not in timesteps:
                timesteps.append(t)
                timesteps = sorted(set(timesteps), reverse=True)
            if len(timesteps) >= K: break
    return timesteps[:K]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--blend", type=float, default=0.6)
    a = p.parse_args()

    print("Experiment 4: Solver Composability")
    print("=" * 55)

    traj_dir = Path("results/exp1/trajectories")
    if not (traj_dir / "ddim_timesteps.npy").exists():
        print("ERROR: Run run_exp1.py first!"); return
    ref_dir = Path("results/fid_stats/cifar10_train_images")
    if not ref_dir.exists():
        print("ERROR: Run setup_fid_stats.py first!"); return

    trajs = [np.load(f) for f in sorted(traj_dir.glob("traj_*.npy"))]
    dts = np.load(traj_dir / "ddim_timesteps.npy")
    print(f"Loaded {len(trajs)} trajectories, {len(dts)} pool timesteps")

    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    print("Computing curvature profile...")
    w = compute_retention_frequency(normalized, dts, target_K=50)

    from diffusers import DDIMScheduler
    alphas_np = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32").alphas_cumprod.numpy()

    rdir = Path("results/exp4"); sdir = rdir / "samples"
    rdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    nfe_budgets = [10, 20, 30, 50]
    all_results = []

    for nfe in nfe_budgets:
        print(f"\n  --- NFE = {nfe} ---")

        # (a) DPM-Solver++ default
        dpm_dir = sdir / f"dpm_default_{nfe}"
        if not dpm_dir.exists():
            try:
                imgs = generate_dpmsolver(nfe, a.num_samples, a.device, a.batch_size,
                                          f"DPM++ default K={nfe}")
                save_imgs(imgs, dpm_dir); del imgs
            except Exception as e:
                print(f"    DPM++ default  K={nfe}: SKIPPED ({e})")
                all_results.append(EvalResult(method="DPM-Solver++", nfe=nfe, fid=float('nan')))
                continue
        dpm_fid = compute_fid(dpm_dir, ref_dir)
        print(f"    DPM-Solver++     K={nfe}: FID={dpm_fid:.2f}")
        all_results.append(EvalResult(method="DPM-Solver++", nfe=nfe, fid=dpm_fid))

        # (b) GeoSPRINT LogSNR+curvature with DDIM
        geo_ts = logsnr_curvature_schedule(alphas_np, w, dts, nfe, blend=a.blend)
        K_geo = len(geo_ts)
        geo_dir = sdir / f"geosprint_{nfe}"
        if not geo_dir.exists():
            imgs = generate_ddim(geo_ts, a.num_samples, a.device, a.batch_size,
                                 f"GeoSPRINT K={K_geo}")
            save_imgs(imgs, geo_dir); del imgs
        geo_fid = compute_fid(geo_dir, ref_dir)
        print(f"    DDIM+GeoSPRINT   K={K_geo}: FID={geo_fid:.2f}")
        all_results.append(EvalResult(method="DDIM+GeoSPRINT", nfe=K_geo, fid=geo_fid))

        # (c) DDIM uniform
        sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        sc.set_timesteps(nfe)
        ddim_dir = sdir / f"ddim_uniform_{nfe}"
        if not ddim_dir.exists():
            imgs = generate_ddim(sc.timesteps.cpu().numpy().tolist(), a.num_samples,
                                 a.device, a.batch_size, f"DDIM uniform K={nfe}")
            save_imgs(imgs, ddim_dir); del imgs
        ddim_fid = compute_fid(ddim_dir, ref_dir)
        print(f"    DDIM uniform     K={nfe}: FID={ddim_fid:.2f}")
        all_results.append(EvalResult(method="DDIM uniform", nfe=nfe, fid=ddim_fid))

        torch.cuda.empty_cache()

    # Output
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\n  {'Method':<25s} {'NFE':>5s} {'FID':>8s} {'vs DPM++':>10s}")
    print(f"  {'-'*52}")

    by_nfe = {}
    for r in all_results:
        by_nfe.setdefault(r.nfe, {})[r.method] = r.fid

    for r in sorted(all_results, key=lambda x: (x.nfe, x.method)):
        dpm_fid = by_nfe.get(r.nfe, {}).get('DPM-Solver++')
        if dpm_fid and r.method != 'DPM-Solver++' and not np.isnan(dpm_fid):
            diff = r.fid - dpm_fid
            dstr = f"{diff:+.2f}"
        elif r.method == 'DPM-Solver++':
            dstr = "ref"
        else:
            dstr = "---"
        fstr = f"{r.fid:.2f}" if not np.isnan(r.fid) else "FAIL"
        print(f"  {r.method:<25s} {r.nfe:>5d} {fstr:>8s} {dstr:>10s}")

    with open(rdir / "exp4_results.json", 'w') as f:
        json.dump([{'method':r.method,'nfe':r.nfe,'fid':r.fid} for r in all_results], f, indent=2)

    tab = format_results_table(all_results, "Solver composability: GeoSPRINT vs DPM-Solver++")
    with open(rdir / "exp4_table.tex", 'w') as f: f.write(tab)

    # Plot
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {
        'DPM-Solver++':   {'m':'^','c':'#1D9E75','ls':'-'},
        'DDIM+GeoSPRINT': {'m':'o','c':'#534AB7','ls':'-'},
        'DDIM uniform':   {'m':'s','c':'#888780','ls':'--'},
    }
    for method, sty in styles.items():
        pts = sorted([r for r in all_results if r.method==method and not np.isnan(r.fid)],
                     key=lambda x: x.nfe)
        if pts:
            ax.plot([r.nfe for r in pts],[r.fid for r in pts],
                    f'{sty["m"]}{sty["ls"]}', label=method, color=sty['c'],
                    markersize=6, linewidth=1.5)
    ax.set_xlabel("NFE"); ax.set_ylabel("FID (lower is better)")
    ax.set_title("Solver composability: GeoSPRINT schedule vs defaults")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/exp4_compose.pdf", dpi=200, bbox_inches='tight')
    print(f"\nPlot: figures/exp4_compose.pdf")
    print(f"JSON: results/exp4/exp4_results.json")

if __name__ == "__main__":
    main()
