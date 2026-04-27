"""
Sweep blend values for LogSNR + GeoSPRINT curvature schedule.
Tests blend=0.5 to 1.0 at NFE=20,30,50.

Usage: python scripts/test_blend_sweep.py --device cuda --num_samples 10000
"""

import argparse, json, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
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

def compute_retention_frequency(normalized_trajs, pool_timesteps, target_K=50):
    N_traj = len(pool_timesteps) + 1
    counts = np.zeros(N_traj)
    for traj in normalized_trajs:
        tau, res = find_threshold_for_K(traj, target_K, tol=5)
        for idx in res.retained_indices:
            if idx < N_traj: counts[idx] += 1
    return counts[1:len(pool_timesteps)+1]

def logsnr_curvature_schedule(alphas_np, retention_freq, pool_timesteps, K, blend=0.3):
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

    from diffusers import DDIMScheduler
    sched = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
    alphas_np = sched.alphas_cumprod.numpy()

    sdir = Path("results/test_blend/samples")
    sdir.mkdir(parents=True, exist_ok=True)

    blends = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nfes = [20, 30, 50]
    all_results = []

    for blend in blends:
        for nfe in nfes:
            ts = logsnr_curvature_schedule(alphas_np, w, dts, nfe, blend=blend)
            K = len(ts)
            tag = f"b{blend:.1f}_nfe{nfe}"
            print(f"\n  blend={blend:.1f}  NFE={nfe}  K={K}")

            sd = sdir / tag
            if not sd.exists():
                imgs = generate(ts, a.num_samples, a.device, a.batch_size,
                               f"blend={blend} K={K}")
                save_imgs(imgs, sd); del imgs
            f = compute_fid(sd, ref_dir)
            print(f"    FID={f:.2f}")
            all_results.append({'blend': blend, 'nfe': K, 'fid': f})
            torch.cuda.empty_cache()

    # DDIM baselines for comparison
    print("\n  DDIM baselines:")
    for nfe in nfes:
        sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        sc.set_timesteps(nfe)
        dd = sdir / f"ddim_{nfe}"
        if not dd.exists():
            imgs = generate(sc.timesteps.cpu().numpy().tolist(), a.num_samples,
                           a.device, a.batch_size, f"DDIM K={nfe}")
            save_imgs(imgs, dd); del imgs
        f = compute_fid(dd, ref_dir)
        print(f"    DDIM NFE={nfe}: FID={f:.2f}")
        all_results.append({'blend': 'ddim', 'nfe': nfe, 'fid': f})

    # Summary
    print("\n" + "=" * 55)
    print(f"  {'blend':>6s}  {'NFE':>5s}  {'FID':>8s}")
    print(f"  {'-'*25}")
    for r in sorted(all_results, key=lambda x: (x['nfe'], str(x['blend']))):
        print(f"  {str(r['blend']):>6s}  {r['nfe']:>5d}  {r['fid']:>8.2f}")

    with open(Path("results/test_blend/results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/test_blend/results.json")


if __name__ == "__main__":
    main()
