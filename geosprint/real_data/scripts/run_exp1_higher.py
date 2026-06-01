"""
Add GeoSPRINT+gap + DDIM at higher NFEs (target_K=110, 120).

Fix: gap constraint uses max(T/K, 2*pool_spacing) so it never
over-fills when K is already dense relative to the pool.

Usage: python3 scripts/run_exp1_higher.py
"""

import json, numpy as np, torch
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

def find_threshold_for_K(traj, target_K, tol=3, max_iter=30):
    tau_lo, tau_hi = 0.0, np.max(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * 3
    best_tau, best_result = tau_hi, None
    for _ in range(max_iter):
        tau_mid = (tau_lo + tau_hi) / 2
        result = prune_trajectory(traj, k=2, threshold=tau_mid)
        K = len(result.retained_indices)
        if abs(K - target_K) <= tol:
            return tau_mid, result
        elif K < target_K:
            tau_hi = tau_mid
        else:
            tau_lo = tau_mid
        best_tau, best_result = tau_mid, result
    return best_tau, best_result

def enforce_min_spacing(ts, max_gap):
    """Fill gaps, but max_gap is already pool-aware."""
    ts = sorted(ts, reverse=True)
    out = [ts[0]]
    for i in range(1, len(ts)):
        gap = out[-1] - ts[i]
        if gap > max_gap:
            n = int(np.ceil(gap/max_gap)) - 1
            for j in range(1, n+1):
                v = out[-1] - int(j * gap / (n+1))
                if v > 0: out.append(v)
        out.append(ts[i])
    return sorted(set(out), reverse=True)


def main():
    rdir = Path("results/exp1")
    sdir = rdir / "samples"
    tdir = rdir / "trajectories"
    ref_dir = Path("results/fid_stats/cifar10_train_images")

    trajs = [np.load(f) for f in sorted(tdir.glob("traj_*.npy"))]
    dts = np.load(tdir / "ddim_timesteps.npy")
    print(f"Loaded {len(trajs)} trajectories, {len(dts)} pool timesteps")

    # Compute pool spacing (e.g., 5 for timesteps [995,990,985,...])
    pool_spacing = int(np.median(np.abs(np.diff(dts))))
    print(f"Pool spacing: {pool_spacing}")

    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True)
        s = t.std(0, keepdims=True) + 1e-8
        normalized.append((t - mu) / s)

    # Load existing
    final_file = rdir / "exp1_final.json"
    if final_file.exists():
        existing = json.load(open(final_file))
    elif (rdir / "exp1_results_with_gap.json").exists():
        existing = json.load(open(rdir / "exp1_results_with_gap.json"))
    else:
        existing = json.load(open(rdir / "exp1_results.json"))
    print(f"Loaded {len(existing)} existing results")

    new_results = []
    T = 1000
    target_Ks = [110, 120]

    for target_K in target_Ks:
        print(f"\n{'='*50}")
        print(f"Target K = {target_K}")
        print(f"{'='*50}")

        all_retained = []
        all_alpha = []
        all_K = []
        for traj in normalized:
            tau, res = find_threshold_for_K(traj, target_K, tol=5)
            all_retained.append(res.retained_indices)
            all_alpha.append(res.projection_score)
            all_K.append(len(res.retained_indices))

        median_K = int(np.median(all_K))
        mean_alpha = float(np.mean(all_alpha))

        N_traj = len(dts) + 1
        counts = np.zeros(N_traj)
        for ret in all_retained:
            for idx in ret:
                if idx < N_traj: counts[idx] += 1
        top_idx = np.sort(np.argsort(-counts)[:median_K])
        geo_ts = []
        for idx in top_idx:
            if idx > 0 and idx - 1 < len(dts):
                geo_ts.append(int(dts[idx - 1]))
        if int(dts[0]) not in geo_ts: geo_ts.insert(0, int(dts[0]))
        if int(dts[-1]) not in geo_ts: geo_ts.append(int(dts[-1]))
        geo_ts = sorted(set(geo_ts), reverse=True)
        K_raw = len(geo_ts)

        print(f"  GeoSPRINT raw: K={K_raw}, α_traj={mean_alpha:.2e}")

        # Gap constraint: max_gap = max(T/K, 2*pool_spacing)
        # This prevents over-filling when K is already dense
        max_gap = max(T // K_raw, pool_spacing * 2) if K_raw < len(dts) // 2 else 999999
        hybrid_ts = enforce_min_spacing(geo_ts, max_gap)
        Kh = len(hybrid_ts)

        print(f"  Gap max_gap={max_gap} (T/K={T//K_raw}, 2*pool={pool_spacing*2})")
        print(f"  GeoSPRINT+gap: K={K_raw}→{Kh}")

        # GeoSPRINT+gap
        hdir = sdir / f"geosprint_gap_higher_K{K_raw}_Kh{Kh}"
        if not hdir.exists():
            imgs = generate(hybrid_ts, 10000, "cuda", 64, f"GeoSPRINT+gap K={Kh}")
            save_imgs(imgs, hdir); del imgs
        hfid = compute_fid(hdir, ref_dir)
        print(f"  GeoSPRINT+gap K={Kh}: FID={hfid:.2f}")
        new_results.append({'method': 'GeoSPRINT+gap', 'nfe': Kh, 'fid': hfid, 'alpha': mean_alpha})

        # DDIM at same Kh
        from diffusers import DDIMScheduler
        ddir = sdir / f"ddim_higher_K{Kh}"
        if not ddir.exists():
            sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
            sc.set_timesteps(Kh)
            imgs = generate(sc.timesteps.cpu().numpy().tolist(), 10000, "cuda", 64, f"DDIM K={Kh}")
            save_imgs(imgs, ddir); del imgs
        dfid = compute_fid(ddir, ref_dir)
        print(f"  DDIM K={Kh}: FID={dfid:.2f}")
        new_results.append({'method': 'DDIM', 'nfe': Kh, 'fid': dfid, 'alpha': 0})

        # GeoSPRINT raw
        gdir = sdir / f"geosprint_higher_K{K_raw}"
        if not gdir.exists():
            imgs = generate(geo_ts, 10000, "cuda", 64, f"GeoSPRINT K={K_raw}")
            save_imgs(imgs, gdir); del imgs
        gfid = compute_fid(gdir, ref_dir)
        print(f"  GeoSPRINT raw K={K_raw}: FID={gfid:.2f}")
        new_results.append({'method': 'GeoSPRINT', 'nfe': K_raw, 'fid': gfid, 'alpha': mean_alpha})

        # DDIM at raw K
        ddir2 = sdir / f"ddim_higher_K{K_raw}"
        if not ddir2.exists():
            sc = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
            sc.set_timesteps(K_raw)
            imgs = generate(sc.timesteps.cpu().numpy().tolist(), 10000, "cuda", 64, f"DDIM K={K_raw}")
            save_imgs(imgs, ddir2); del imgs
        dfid2 = compute_fid(ddir2, ref_dir)
        print(f"  DDIM K={K_raw}: FID={dfid2:.2f}")
        new_results.append({'method': 'DDIM', 'nfe': K_raw, 'fid': dfid2, 'alpha': 0})

        torch.cuda.empty_cache()

    # Merge, deduplicate, save
    combined = existing + new_results
    seen = set()
    deduped = []
    for r in combined:
        key = (r['method'], r['nfe'])
        if key not in seen:
            deduped.append(r)
            seen.add(key)
    deduped.sort(key=lambda x: (x['nfe'], x['method']))

    with open(rdir / "exp1_final.json", 'w') as f:
        json.dump(deduped, f, indent=2)

    print("\n" + "=" * 65)
    print("ALL RESULTS")
    print("=" * 65)
    print(f"\n  {'Method':<25s} {'NFE':>5s} {'FID':>8s} {'α_traj':>12s}")
    print(f"  {'-'*55}")
    for r in deduped:
        astr = f"{r['alpha']:.1e}" if r.get('alpha', 0) > 0 else "---"
        print(f"  {r['method']:<25s} {r['nfe']:>5d} {r['fid']:>8.2f} {astr:>12s}")
    print(f"\nSaved to results/exp1/exp1_final.json")


if __name__ == "__main__":
    main()
