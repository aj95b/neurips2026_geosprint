"""
Church 256: LogSNR+curvature GeoSPRINT (reuses saved trajectories/baselines)

Usage: python scripts/run_church256_logsnr.py --device cuda:0 --num_samples 10000 --skip_record
"""

import argparse, json, shutil, time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory

MODEL_ID = "google/ddpm-ema-church-256"
RES = 256; CH = 3; T = 1000

_model_cache = {}

def get_model(device):
    if device not in _model_cache:
        from diffusers import UNet2DModel
        m = UNet2DModel.from_pretrained(MODEL_ID).to(device); m.eval()
        _model_cache[device] = m
    return _model_cache[device]

def get_alphas(device):
    from diffusers import DDIMScheduler
    return DDIMScheduler.from_pretrained(MODEL_ID).alphas_cumprod.to(device)

def to_images(latents):
    return ((latents.clamp(-1,1)+1)/2*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

def ddim_step(x, eps, a_t, a_prev):
    x0 = ((x - (1-a_t).sqrt()*eps) / a_t.sqrt()).clamp(-1,1)
    return a_prev.sqrt()*x0 + (1-a_prev).sqrt()*eps

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
    W = np.cumsum(combined); W = W / W[-1]
    quantiles = np.linspace(0, 1, K)
    timesteps = []
    for q in quantiles:
        idx = np.searchsorted(W, q); idx = min(idx, len(alphas_np)-1)
        timesteps.append(idx)
    timesteps = sorted(set(timesteps), reverse=True)
    if len(timesteps) < K:
        for t in range(999, -1, -1):
            if t not in timesteps:
                timesteps.append(t)
                timesteps = sorted(set(timesteps), reverse=True)
            if len(timesteps) >= K: break
    return timesteps[:K]

def record_trajectories(B, N, device, save_dir):
    from diffusers import DDIMScheduler
    print(f"\n[Phase 1] Recording {B} DDIM trajectories at {N} steps...")
    model = get_model(device); alphas = get_alphas(device)
    sched = DDIMScheduler.from_pretrained(MODEL_ID); sched.set_timesteps(N)
    ts = sched.timesteps.cpu().numpy()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    trajs = []
    for b in tqdm(range(B), desc="Recording"):
        g = torch.Generator(device=device).manual_seed(b)
        z = torch.randn(1, CH, RES, RES, generator=g, device=device)
        path = [z.detach().cpu().numpy().flatten()]
        tl = ts.tolist()
        for i, t in enumerate(tl):
            with torch.no_grad(): eps = model(z, torch.tensor(t, device=device)).sample
            a_t = alphas[t]; a_prev = alphas[tl[i+1]] if i+1<len(tl) else torch.tensor(1.0, device=device)
            z = ddim_step(z, eps, a_t, a_prev)
            path.append(z.detach().cpu().numpy().flatten())
        arr = np.stack(path); trajs.append(arr)
        np.save(save_dir / f"traj_{b:04d}.npy", arr)
    np.save(save_dir / "ddim_timesteps.npy", ts)
    print(f"  Shape: {trajs[0].shape}")
    return trajs, ts

def load_trajectories(d):
    d = Path(d)
    trajs = [np.load(f) for f in sorted(d.glob("traj_*.npy"))]
    ts = np.load(d / "ddim_timesteps.npy")
    print(f"  Loaded {len(trajs)} trajectories")
    return trajs, ts

def generate(ts_use, num, device, bs=8, desc="Gen"):
    model = get_model(device); alphas = get_alphas(device)
    valid = sorted([int(t) for t in ts_use if 0<=int(t)<1000], reverse=True)
    samples = []
    for bi in tqdm(range((num+bs-1)//bs), desc=desc):
        b = min(bs, num-bi*bs)
        g = torch.Generator(device=device).manual_seed(10000+bi)
        z = torch.randn(b, CH, RES, RES, generator=g, device=device)
        for i, t in enumerate(valid):
            with torch.no_grad(): eps = model(z, torch.tensor(t, device=device)).sample
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
    from cleanfid import fid as cf
    return cf.compute_fid(str(gd), str(rd))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_batch", type=int, default=50)
    p.add_argument("--pool_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_record", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--blend", type=float, default=0.6)
    a = p.parse_args()

    rdir = Path("results/church256"); tdir = rdir/"trajectories"
    sdir = rdir/"samples"; fdir = Path("figures")
    for d in [rdir, sdir, fdir]: d.mkdir(parents=True, exist_ok=True)

    if a.skip_record and (tdir / "ddim_timesteps.npy").exists():
        trajs, dts = load_trajectories(tdir)
    else:
        if tdir.exists(): shutil.rmtree(tdir)
        trajs, dts = record_trajectories(a.ref_batch, a.pool_steps, a.device, tdir)

    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    print(f"\n[Phase 2] Curvature analysis...")
    t0 = time.time()
    w = compute_retention_frequency(normalized, dts, target_K=50)
    print(f"  Retention freq range: [{w.min():.1f}, {w.max():.1f}]")

    from diffusers import DDIMScheduler
    alphas_np = DDIMScheduler.from_pretrained(MODEL_ID).alphas_cumprod.numpy()

    # Budget search
    target_Ks = list(range(50, min(len(dts)+1, 201), 10))
    budget_results = []
    for target_K in target_Ks:
        all_K = []; all_alpha = []
        for traj in normalized:
            tau, res = find_threshold_for_K(traj, target_K, tol=5)
            all_K.append(len(res.retained_indices))
            all_alpha.append(res.projection_score)
        median_K = int(np.median(all_K))
        mean_alpha = float(np.mean(all_alpha))
        budget_results.append({'target_K': target_K, 'K': median_K, 'alpha': mean_alpha})
        print(f"  target_K={target_K:>4d}  K={median_K:>4d}  alpha={mean_alpha:.2e}")
        if mean_alpha < 1e-6:
            print("  -> diminishing returns"); break
    print(f"  Analysis: {time.time()-t0:.1f}s")

    # Generate
    print(f"\n[Phase 3] Generating {a.num_samples} samples (blend={a.blend})...")

    ref_dir = sdir / "baseline"
    all_results = []

    # Baseline (reuse if exists)
    if not ref_dir.exists():
        print(f"\n  --- Baseline: DDIM-{len(dts)} ---")
        imgs = generate(dts.tolist(), a.num_samples, a.device, a.batch_size, f"DDIM-{len(dts)}")
        save_imgs(imgs, ref_dir); del imgs
    base_fid_file = rdir / "baseline_fid.json"
    if base_fid_file.exists():
        base_fid = json.load(open(base_fid_file))['fid']
    else:
        # Need a real reference for FID - use baseline as reference
        # On LSUN Church we compute FID against our own full-step generations
        base_fid = 0.0  # self-reference
    print(f"  Baseline exists: {ref_dir.exists()}")

    for br in budget_results:
        K = br['K']; alpha = br['alpha']
        print(f"\n  --- K={K} (alpha={alpha:.2e}) ---")

        # GeoSPRINT LogSNR+curvature
        geo_ts = logsnr_curvature_schedule(alphas_np, w, dts, K, blend=a.blend)
        K_actual = len(geo_ts)
        gd = sdir / f"geosprint_logsnr_K{K_actual}"
        if not gd.exists():
            imgs = generate(geo_ts, a.num_samples, a.device, a.batch_size,
                           f"GeoSPRINT K={K_actual}")
            save_imgs(imgs, gd); del imgs
        geo_fid = compute_fid(gd, ref_dir)
        print(f"    GeoSPRINT    K={K_actual}: FID={geo_fid:.2f}")
        all_results.append({'method': 'GeoSPRINT', 'nfe': K_actual, 'fid': geo_fid, 'alpha': alpha})

        # DDIM uniform (reuse if exists from old run)
        sc = DDIMScheduler.from_pretrained(MODEL_ID); sc.set_timesteps(K_actual)
        dd = sdir / f"ddim_logsnr_K{K_actual}"
        if not dd.exists():
            imgs = generate(sc.timesteps.cpu().numpy().tolist(), a.num_samples,
                           a.device, a.batch_size, f"DDIM K={K_actual}")
            save_imgs(imgs, dd); del imgs
        ddim_fid = compute_fid(dd, ref_dir)
        print(f"    DDIM uniform K={K_actual}: FID={ddim_fid:.2f}")
        all_results.append({'method': 'DDIM', 'nfe': K_actual, 'fid': ddim_fid, 'alpha': 0})

        torch.cuda.empty_cache()

    # Output
    print("\n" + "=" * 55)
    print(f"  {'Method':<20s} {'NFE':>5s} {'FID':>8s} {'vs DDIM':>10s}")
    print(f"  {'-'*47}")
    by_nfe = {}
    for r in all_results: by_nfe.setdefault(r['nfe'], {})[r['method']] = r['fid']
    for r in sorted(all_results, key=lambda x: (x['nfe'], x['method'])):
        ddim_f = by_nfe.get(r['nfe'], {}).get('DDIM')
        if ddim_f and r['method'] == 'GeoSPRINT':
            dstr = f"{r['fid']-ddim_f:+.2f}"
        else:
            dstr = "ref" if r['method'] == 'DDIM' else "---"
        print(f"  {r['method']:<20s} {r['nfe']:>5d} {r['fid']:>8.2f} {dstr:>10s}")

    with open(rdir / "results_logsnr.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, m, c, ls in [('GeoSPRINT','o','#534AB7','-'),('DDIM','s','#888780','--')]:
        pts = sorted([r for r in all_results if r['method']==method], key=lambda x:x['nfe'])
        if pts:
            ax.plot([r['nfe'] for r in pts],[r['fid'] for r in pts], f'{m}{ls}',
                    label=method, color=c, markersize=6, linewidth=1.5)
    ax.set_xlabel("NFE"); ax.set_ylabel("FID vs baseline")
    ax.set_title("LSUN Church 256: GeoSPRINT vs DDIM")
    ax.legend(); ax.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(fdir/"church256_pareto.pdf", dpi=200, bbox_inches='tight')
    print(f"\nPlot: figures/church256_pareto.pdf")
    print(f"JSON: results/church256/results_logsnr.json")

if __name__ == "__main__":
    main()
