"""
Test: importance-weighted GeoSPRINT schedule vs top-K vs DDIM uniform.
Runs at NFE=10,20,30,50. Quick comparison (~2 hrs).

Usage: python scripts/test_weighted_schedule.py --device cuda --num_samples 10000
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


def importance_weighted_schedule(retention_freq, pool_timesteps, K):
    """
    Place K timesteps proportional to retention frequency.
    Like importance sampling: steps are dense where w(t) is high,
    sparse where w(t) is low, but ALWAYS cover the full range.
    """
    w = retention_freq.copy()
    # Ensure minimum weight everywhere (prevents zero-density gaps)
    w = w + 0.01 * w.max()
    # Cumulative distribution
    W = np.cumsum(w)
    W = W / W[-1]  # normalize to [0, 1]

    # Place K steps at uniform quantiles of the cumulative distribution
    quantiles = np.linspace(0, 1, K)
    indices = []
    for q in quantiles:
        idx = np.searchsorted(W, q)
        idx = min(idx, len(pool_timesteps) - 1)
        indices.append(idx)

    # Deduplicate while maintaining K steps
    indices = sorted(set(indices))
    # If we lost steps due to dedup, fill from highest-retention unused indices
    if len(indices) < K:
        all_idx = np.argsort(-w)
        for idx in all_idx:
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= K:
                break
        indices = sorted(indices)

    # Map to actual timesteps
    selected = [int(pool_timesteps[i]) for i in indices if i < len(pool_timesteps)]

    # Ensure first and last
    if int(pool_timesteps[0]) not in selected:
        selected.insert(0, int(pool_timesteps[0]))
    if int(pool_timesteps[-1]) not in selected:
        selected.append(int(pool_timesteps[-1]))

    return sorted(set(selected), reverse=True)


def compute_retention_frequency(normalized_trajs, pool_timesteps, target_K=50):
    """Compute per-timestep retention frequency across trajectories."""
    N_traj = len(pool_timesteps) + 1
    counts = np.zeros(N_traj)

    for traj in normalized_trajs:
        tau, res = find_threshold_for_K(traj, target_K, tol=5)
        for idx in res.retained_indices:
            if idx < N_traj:
                counts[idx] += 1

    # Drop index 0 (initial noise sentinel) to align with pool_timesteps
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

    # Normalize
    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    # Compute retention frequency (using target_K=50 for good resolution)
    print("Computing retention frequency...")
    w = compute_retention_frequency(normalized, dts, target_K=50)
    print(f"  w shape: {w.shape}, range: [{w.min():.1f}, {w.max():.1f}]")

    sdir = Path("results/test_weighted/samples")
    sdir.mkdir(parents=True, exist_ok=True)

    nfe_budgets = [10, 20, 30, 50]
    all_results = []

    for nfe in nfe_budgets:
        print(f"\n  === NFE = {nfe} ===")

        # (a) Importance-weighted GeoSPRINT
        weighted_ts = importance_weighted_schedule(w, dts, nfe)
        Kw = len(weighted_ts)
        print(f"    Weighted: {Kw} steps, ts={weighted_ts[:5]}...")

        wd = sdir / f"weighted_{nfe}"
        if not wd.exists():
            imgs = generate(weighted_ts, a.num_samples, a.device, a.batch_size,
                           f"Weighted K={Kw}")
            save_imgs(imgs, wd); del imgs
        w_fid = compute_fid(wd, ref_dir)
        print(f"    GeoSPRINT-weighted K={Kw}: FID={w_fid:.2f}")
        all_results.append({'method': 'GeoSPRINT-weighted', 'nfe': Kw, 'fid': w_fid})

        # (b) DDIM uniform at same K
        from diffusers import DDIMScheduler
        sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        sc.set_timesteps(Kw)
        ddim_ts = sc.timesteps.cpu().numpy().tolist()

        dd = sdir / f"ddim_{Kw}"
        if not dd.exists():
            imgs = generate(ddim_ts, a.num_samples, a.device, a.batch_size,
                           f"DDIM K={Kw}")
            save_imgs(imgs, dd); del imgs
        d_fid = compute_fid(dd, ref_dir)
        print(f"    DDIM uniform      K={Kw}: FID={d_fid:.2f}")
        all_results.append({'method': 'DDIM', 'nfe': Kw, 'fid': d_fid})

        delta = w_fid - d_fid
        print(f"    Δ = {delta:+.2f} {'(worse)' if delta > 0 else '(BETTER!)'}")

        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 55)
    print(f"  {'Method':<25s} {'NFE':>5s} {'FID':>8s} {'vs DDIM':>10s}")
    print(f"  {'-'*50}")
    by_nfe = {}
    for r in all_results:
        by_nfe.setdefault(r['nfe'], {})[r['method']] = r['fid']
    for r in sorted(all_results, key=lambda x: (x['nfe'], x['method'])):
        nfe = r['nfe']
        if r['method'] != 'DDIM' and 'DDIM' in by_nfe.get(nfe, {}):
            delta = r['fid'] - by_nfe[nfe]['DDIM']
            dstr = f"{delta:+.2f}"
        else:
            dstr = "ref"
        print(f"  {r['method']:<25s} {r['nfe']:>5d} {r['fid']:>8.2f} {dstr:>10s}")

    with open(Path("results/test_weighted/results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/test_weighted/results.json")


if __name__ == "__main__":
    main()
