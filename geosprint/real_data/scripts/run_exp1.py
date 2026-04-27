"""
Experiment 1 (Final): GeoSPRINT Step-Budget Oracle
====================================================
Uses LogSNR + curvature blended schedule (blend=0.6).

1. Record 100 DDIM trajectories at 200 steps
2. Compute curvature retention frequency
3. GeoSPRINT budget search from K=50 upward
4. At each K: generate with GeoSPRINT (LogSNR+curv) and DDIM uniform
5. Baselines: DDIM-200, DDIM-50

Usage:
    python scripts/run_exp1.py --device cuda --num_samples 10000
    python scripts/run_exp1.py --device cuda --num_samples 10000 --skip_record
"""

import argparse, json, shutil, time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory, threshold_search
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

def record_trajectories(B, N, device, save_dir):
    from diffusers import DDIMScheduler
    print(f"\n[Phase 1] Recording {B} DDIM trajectories at {N} steps...")
    model = get_model(device)
    alphas = get_alphas(device)
    sched = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
    sched.set_timesteps(N)
    ts = sched.timesteps.cpu().numpy()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    trajs = []
    for b in tqdm(range(B), desc="Recording"):
        g = torch.Generator(device=device).manual_seed(b)
        z = torch.randn(1,3,32,32, generator=g, device=device)
        path = [z.detach().cpu().numpy().flatten()]
        tl = ts.tolist()
        for i, t in enumerate(tl):
            with torch.no_grad():
                eps = model(z, torch.tensor(t, device=device)).sample
            a_t = alphas[t]
            a_prev = alphas[tl[i+1]] if i+1<len(tl) else torch.tensor(1.0, device=device)
            z = ddim_step(z, eps, a_t, a_prev)
            path.append(z.detach().cpu().numpy().flatten())
        arr = np.stack(path); trajs.append(arr)
        np.save(save_dir / f"traj_{b:04d}.npy", arr)
    np.save(save_dir / "ddim_timesteps.npy", ts)
    print(f"  Shape: {trajs[0].shape}, Timesteps: {ts[:5]}...{ts[-5:]}")
    return trajs, ts

def load_trajectories(d):
    d = Path(d)
    trajs = [np.load(f) for f in sorted(d.glob("traj_*.npy"))]
    ts = np.load(d / "ddim_timesteps.npy")
    print(f"  Loaded {len(trajs)} trajectories")
    return trajs, ts

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

def geosprint_budget_search(normalized_trajs, ddim_ts, min_K=50):
    print(f"\n  Starting from K={min_K}, searching upward...")
    target_Ks = list(range(min_K, len(ddim_ts)+1, 10))
    if target_Ks[-1] != len(ddim_ts): target_Ks.append(len(ddim_ts))
    results = []
    for target_K in target_Ks:
        all_K = []; all_alpha = []
        for traj in normalized_trajs:
            tau, res = find_threshold_for_K(traj, target_K, tol=3)
            all_K.append(len(res.retained_indices))
            all_alpha.append(res.projection_score)
        median_K = int(np.median(all_K))
        mean_alpha = float(np.mean(all_alpha))
        results.append({'target_K': target_K, 'K': median_K, 'alpha': mean_alpha})
        print(f"  target_K={target_K:>4d}  K={median_K:>4d}  alpha={mean_alpha:.2e}")
        if mean_alpha < 1e-6:
            print("  -> diminishing returns, stopping"); break
    return results

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_batch", type=int, default=100)
    p.add_argument("--pool_steps", type=int, default=200)
    p.add_argument("--min_K", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_record", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--blend", type=float, default=0.6)
    a = p.parse_args()

    rdir = Path("results/exp1"); tdir = rdir/"trajectories"; sdir = rdir/"samples"
    fdir = Path("figures")
    for d in [rdir, sdir, fdir]: d.mkdir(parents=True, exist_ok=True)

    ref_dir = Path("results/fid_stats/cifar10_train_images")
    if not ref_dir.exists():
        print("ERROR: Run setup_fid_stats.py first!"); return

    # Phase 1
    if a.skip_record and (tdir / "ddim_timesteps.npy").exists():
        trajs, dts = load_trajectories(tdir)
    else:
        if tdir.exists(): shutil.rmtree(tdir)
        trajs, dts = record_trajectories(a.ref_batch, a.pool_steps, a.device, tdir)

    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    # Phase 2
    print(f"\n[Phase 2] Curvature analysis + budget search...")
    t0 = time.time()
    w = compute_retention_frequency(normalized, dts, target_K=50)
    print(f"  Retention freq: range=[{w.min():.1f}, {w.max():.1f}]")
    budget_results = geosprint_budget_search(normalized, dts, min_K=a.min_K)
    print(f"  Analysis: {time.time()-t0:.1f}s")

    from diffusers import DDIMScheduler
    alphas_np = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32").alphas_cumprod.numpy()

    # Phase 3
    print(f"\n[Phase 3] Generating {a.num_samples} samples (blend={a.blend})...")
    if sdir.exists():
        for d in sdir.iterdir():
            if d.is_dir(): shutil.rmtree(d)

    all_results = []

    # Baseline DDIM-200
    print(f"\n  --- Baseline: DDIM-{len(dts)} ---")
    bd = sdir / "baseline_full"
    imgs = generate(dts.tolist(), a.num_samples, a.device, a.batch_size, f"DDIM-{len(dts)}")
    save_imgs(imgs, bd); del imgs
    base_fid = compute_fid(bd, ref_dir)
    print(f"    DDIM-{len(dts)}: FID={base_fid:.2f}")
    all_results.append(EvalResult(method=f"DDIM-{len(dts)} (baseline)", nfe=len(dts), fid=base_fid))

    # Reference DDIM-50
    print(f"\n  --- Reference: DDIM-{a.min_K} ---")
    sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
    sc.set_timesteps(a.min_K)
    rd = sdir / f"ddim_{a.min_K}"
    imgs = generate(sc.timesteps.cpu().numpy().tolist(), a.num_samples, a.device, a.batch_size, f"DDIM-{a.min_K}")
    save_imgs(imgs, rd); del imgs
    ref50_fid = compute_fid(rd, ref_dir)
    print(f"    DDIM-{a.min_K}: FID={ref50_fid:.2f}")
    all_results.append(EvalResult(method=f"DDIM-{a.min_K}", nfe=a.min_K, fid=ref50_fid))

    # GeoSPRINT + DDIM at each K
    for br in budget_results:
        K = br['K']; alpha = br['alpha']
        print(f"\n  --- K={K} (alpha={alpha:.2e}) ---")

        geo_ts = logsnr_curvature_schedule(alphas_np, w, dts, K, blend=a.blend)
        K_actual = len(geo_ts)
        gd = sdir / f"geosprint_K{K_actual}"
        imgs = generate(geo_ts, a.num_samples, a.device, a.batch_size, f"GeoSPRINT K={K_actual}")
        save_imgs(imgs, gd); del imgs
        geo_fid = compute_fid(gd, ref_dir)
        print(f"    GeoSPRINT    K={K_actual}: FID={geo_fid:.2f}")
        all_results.append(EvalResult(method="GeoSPRINT", nfe=K_actual, fid=geo_fid, mean_alpha=alpha))

        sc.set_timesteps(K_actual)
        dd = sdir / f"ddim_K{K_actual}"
        imgs = generate(sc.timesteps.cpu().numpy().tolist(), a.num_samples, a.device, a.batch_size, f"DDIM K={K_actual}")
        save_imgs(imgs, dd); del imgs
        ddim_fid = compute_fid(dd, ref_dir)
        print(f"    DDIM uniform K={K_actual}: FID={ddim_fid:.2f}")
        all_results.append(EvalResult(method="DDIM", nfe=K_actual, fid=ddim_fid))

        torch.cuda.empty_cache()

    # Output
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline:  DDIM-{len(dts)} FID = {base_fid:.2f}")
    print(f"  Reference: DDIM-{a.min_K}  FID = {ref50_fid:.2f}")
    print(f"  Blend:     {a.blend}\n")
    print(f"  {'Method':<25s} {'NFE':>5s} {'FID':>8s} {'alpha':>10s} {'vs DDIM':>10s}")
    print(f"  {'-'*63}")
    for r in sorted(all_results, key=lambda x: (x.nfe, x.method)):
        astr = f"{r.mean_alpha:.1e}" if r.mean_alpha > 0 else "---"
        ddim_match = [x for x in all_results if x.method == "DDIM" and x.nfe == r.nfe]
        if ddim_match and r.method == "GeoSPRINT":
            diff = r.fid - ddim_match[0].fid
            dstr = f"{diff:+.2f}"
        else:
            dstr = "---"
        print(f"  {r.method:<25s} {r.nfe:>5d} {r.fid:>8.2f} {astr:>10s} {dstr:>10s}")

    with open(rdir / "exp1_results.json", 'w') as f:
        json.dump([{'method':r.method,'nfe':r.nfe,'fid':r.fid,'alpha':r.mean_alpha} for r in all_results], f, indent=2)

    tab = format_results_table(all_results, "CIFAR-10: GeoSPRINT (LogSNR+curvature)")
    with open(rdir / "exp1_table.tex", 'w') as f: f.write(tab)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, m, c, ls in [('GeoSPRINT','o','#534AB7','-'), ('DDIM','s','#888780','--')]:
        pts = sorted([r for r in all_results if r.method==method and not np.isnan(r.fid)], key=lambda x:x.nfe)
        if pts:
            ax1.plot([r.nfe for r in pts],[r.fid for r in pts], f'{m}{ls}', label=method, color=c, markersize=6, linewidth=1.5)
    ax1.axhline(base_fid, color='#1D9E75', linestyle=':', linewidth=0.8, alpha=0.7)
    ax1.axhline(ref50_fid, color='#D85A30', linestyle=':', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("NFE"); ax1.set_ylabel("FID (lower is better)"); ax1.set_title("(a) FID vs NFE")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)
    geo_pts = sorted([r for r in all_results if r.method=="GeoSPRINT"], key=lambda x:x.nfe)
    if geo_pts:
        ax2.plot([r.nfe for r in geo_pts],[r.mean_alpha for r in geo_pts], 'o-', color='#534AB7', markersize=6, linewidth=1.5)
        ax2.set_xlabel("NFE"); ax2.set_ylabel("alpha_traj"); ax2.set_title("(b) Variance preservation vs NFE")
        ax2.set_yscale('log'); ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fdir / "exp1_pareto.pdf", dpi=200, bbox_inches='tight')
    print(f"\nPlot: figures/exp1_pareto.pdf")
    print(f"JSON: results/exp1/exp1_results.json")

if __name__ == "__main__":
    main()
