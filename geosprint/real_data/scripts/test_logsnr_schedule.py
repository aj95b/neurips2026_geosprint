"""
Test: log-SNR spacing + GeoSPRINT curvature refinement
========================================================
Three schedules compared:
  1. DDIM uniform (baseline) — uniform in t
  2. Log-SNR uniform — uniform in log(alpha/(1-alpha))
  3. Log-SNR + GeoSPRINT — log-SNR base shifted by curvature

Usage: python scripts/test_logsnr_schedule.py --device cuda --num_samples 10000
"""

import argparse, json, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory

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
    for i, im in enumerate(imgs): Image.fromarray(im).save(d/f"{i:05d}.png")

def compute_fid(gd, rd):
    from cleanfid import fid as cf
    return cf.compute_fid(str(gd), str(rd))

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


def logsnr_uniform_schedule(alphas_cumprod_np, K):
    """Place K timesteps uniformly in log-SNR space."""
    # log-SNR = log(alpha / (1 - alpha))
    eps = 1e-8
    logsnr = np.log(alphas_cumprod_np / (1 - alphas_cumprod_np + eps) + eps)

    # Uniform quantiles in log-SNR
    logsnr_min = logsnr[-1]  # t=999, lowest SNR
    logsnr_max = logsnr[0]   # t=0, highest SNR
    targets = np.linspace(logsnr_max, logsnr_min, K)

    timesteps = []
    for target in targets:
        idx = np.argmin(np.abs(logsnr - target))
        timesteps.append(idx)

    # Deduplicate
    timesteps = sorted(set(timesteps), reverse=True)

    # Fill if we lost steps
    if len(timesteps) < K:
        all_t = list(range(999, -1, -1))
        for t in all_t:
            if t not in timesteps:
                timesteps.append(t)
                timesteps = sorted(set(timesteps), reverse=True)
            if len(timesteps) >= K:
                break

    return timesteps[:K]


def logsnr_curvature_schedule(alphas_cumprod_np, retention_freq, pool_timesteps, K, blend=0.3):
    """
    Hybrid: log-SNR base with GeoSPRINT curvature shift.

    Places steps uniformly in a blended space:
        spacing ∝ (1-blend) * uniform_logSNR + blend * curvature_density

    blend=0: pure log-SNR uniform
    blend=1: pure curvature-weighted
    blend=0.3: mostly log-SNR, nudged by curvature
    """
    eps = 1e-8
    logsnr = np.log(alphas_cumprod_np / (1 - alphas_cumprod_np + eps) + eps)

    # Build density from log-SNR (uniform = constant density)
    logsnr_density = np.ones(len(alphas_cumprod_np))

    # Build density from curvature (map pool retention to full t range)
    curv_density = np.zeros(len(alphas_cumprod_np))
    for i, t in enumerate(pool_timesteps):
        t_int = int(t)
        if 0 <= t_int < len(curv_density):
            curv_density[t_int] = retention_freq[i]
    # Smooth it
    from scipy.ndimage import gaussian_filter1d
    curv_density = gaussian_filter1d(curv_density, sigma=10)
    curv_density = curv_density + 0.01 * curv_density.max()  # floor

    # Blend
    combined = (1 - blend) * logsnr_density / logsnr_density.sum() + \
               blend * curv_density / curv_density.sum()

    # Cumulative
    W = np.cumsum(combined)
    W = W / W[-1]

    # Place K steps at uniform quantiles
    quantiles = np.linspace(0, 1, K)
    timesteps = []
    for q in quantiles:
        idx = np.searchsorted(W, q)
        idx = min(idx, len(alphas_cumprod_np) - 1)
        timesteps.append(idx)

    timesteps = sorted(set(timesteps), reverse=True)

    # Fill if dedup reduced count
    if len(timesteps) < K:
        all_t = list(range(999, -1, -1))
        for t in all_t:
            if t not in timesteps:
                timesteps.append(t)
                timesteps = sorted(set(timesteps), reverse=True)
            if len(timesteps) >= K:
                break

    return timesteps[:K]


def compute_retention_frequency(normalized_trajs, pool_timesteps, target_K=50):
    N_traj = len(pool_timesteps) + 1
    counts = np.zeros(N_traj)
    for traj in normalized_trajs:
        tau, res = find_threshold_for_K(traj, target_K, tol=5)
        for idx in res.retained_indices:
            if idx < N_traj: counts[idx] += 1
    return counts[1:len(pool_timesteps)+1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()

    traj_dir = Path("results/exp1/trajectories")
    if not (traj_dir / "ddim_timesteps.npy").exists():
        print("ERROR: Run run_exp1.py first!"); return
    ref_dir = Path("results/fid_stats/cifar10_train_images")

    trajs = [np.load(f) for f in sorted(traj_dir.glob("traj_*.npy"))]
    dts = np.load(traj_dir / "ddim_timesteps.npy")
    print(f"Loaded {len(trajs)} trajectories, {len(dts)} pool timesteps")

    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    print("Computing retention frequency...")
    w = compute_retention_frequency(normalized, dts, target_K=50)

    # Get alphas_cumprod as numpy
    from diffusers import DDIMScheduler
    sched = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
    alphas_np = sched.alphas_cumprod.numpy()

    sdir = Path("results/test_logsnr/samples")
    sdir.mkdir(parents=True, exist_ok=True)

    nfe_budgets = [10, 20, 30, 50]
    all_results = []

    for nfe in nfe_budgets:
        print(f"\n  === NFE = {nfe} ===")

        schedules = {}

        # (a) DDIM uniform
        sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        sc.set_timesteps(nfe)
        schedules['DDIM uniform'] = sc.timesteps.cpu().numpy().tolist()

        # (b) Log-SNR uniform
        schedules['Log-SNR uniform'] = logsnr_uniform_schedule(alphas_np, nfe)

        # (c) Log-SNR + curvature (blend=0.3)
        schedules['LogSNR+curv 0.3'] = logsnr_curvature_schedule(
            alphas_np, w, dts, nfe, blend=0.3)

        # (d) Log-SNR + curvature (blend=0.5)
        schedules['LogSNR+curv 0.5'] = logsnr_curvature_schedule(
            alphas_np, w, dts, nfe, blend=0.5)

        for method, ts in schedules.items():
            K = len(ts)
            tag = method.replace(' ', '_').replace('+', '_')
            print(f"    {method}: K={K}, ts={ts[:5]}...")

            sd = sdir / f"{tag}_{nfe}"
            if not sd.exists():
                imgs = generate(ts, a.num_samples, a.device, a.batch_size,
                               f"{method} K={K}")
                save_imgs(imgs, sd); del imgs
            fid_score = compute_fid(sd, ref_dir)
            print(f"    {method:25s} K={K}: FID={fid_score:.2f}")
            all_results.append({'method': method, 'nfe': K, 'fid': fid_score})

            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print(f"  {'Method':<25s} {'NFE':>5s} {'FID':>8s} {'vs DDIM':>10s}")
    print(f"  {'-'*52}")

    by_nfe = {}
    for r in all_results:
        by_nfe.setdefault(r['nfe'], {})[r['method']] = r['fid']

    for r in sorted(all_results, key=lambda x: (x['nfe'], x['method'])):
        nfe = r['nfe']
        ddim_fid = by_nfe.get(nfe, {}).get('DDIM uniform')
        if ddim_fid and r['method'] != 'DDIM uniform':
            delta = r['fid'] - ddim_fid
            dstr = f"{delta:+.2f}"
        else:
            dstr = "ref"
        print(f"  {r['method']:<25s} {r['nfe']:>5d} {r['fid']:>8.2f} {dstr:>10s}")

        # Print blank line between NFE groups
        next_idx = sorted(all_results, key=lambda x: (x['nfe'], x['method'])).index(r) + 1
        if next_idx < len(all_results):
            next_r = sorted(all_results, key=lambda x: (x['nfe'], x['method']))[next_idx]
            if next_r['nfe'] != r['nfe']:
                print()

    with open(Path("results/test_logsnr/results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/test_logsnr/results.json")


if __name__ == "__main__":
    main()
