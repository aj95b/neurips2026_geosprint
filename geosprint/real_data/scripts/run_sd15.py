"""
GeoSPRINT on Stable Diffusion v1.5 (512×512 latent diffusion)
===============================================================
Uses GeoSPRINT+gap in the 4×64×64 latent space.
Float16 to fit on 11GB GPUs.

Usage:
    python scripts/run_sd15.py --device cuda:1 --num_samples 5000
    python scripts/run_sd15.py --device cuda:1 --num_samples 5000 --skip_record
"""

import argparse, json, shutil, time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LATENT_CHANNELS = 4
LATENT_SIZE = 64
T = 1000

PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a beautiful sunset over the ocean",
    "a mountain landscape with snow",
    "a city street at night with neon lights",
    "a garden with colorful flowers",
    "a portrait of a person smiling",
    "a bowl of fruit on a wooden table",
    "a forest path in autumn",
    "an old castle on a hilltop",
    "a dog playing in a park",
]

_pipe_cache = {}

def get_pipe(device):
    if device not in _pipe_cache:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        _pipe_cache[device] = pipe
    return _pipe_cache[device]

def enforce_min_spacing(ts, max_gap):
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


def record_trajectories(B, N, device, save_dir):
    from diffusers import DDIMScheduler
    print(f"\n[Phase 1] Recording {B} SD v1.5 trajectories at {N} steps...")
    pipe = get_pipe(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe.scheduler.set_timesteps(N)
    ts = pipe.scheduler.timesteps.cpu().numpy()
    alphas = pipe.scheduler.alphas_cumprod.to(device)
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    trajs = []

    for b in tqdm(range(B), desc="Recording"):
        prompt = PROMPTS[b % len(PROMPTS)]
        g = torch.Generator(device=device).manual_seed(b)
        text_input = pipe.tokenizer(prompt, padding="max_length",
                                     max_length=pipe.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = pipe.tokenizer("", padding="max_length",
                                       max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_emb = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        prompt_emb = torch.cat([uncond_emb, text_emb])

        latents = torch.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
                              generator=g, device=device, dtype=torch.float16)
        latents = latents * pipe.scheduler.init_noise_sigma
        path = [latents.float().cpu().numpy().flatten()]

        tl = ts.tolist()
        for i, t in enumerate(tl):
            latent_input = torch.cat([latents]*2)
            latent_input = pipe.scheduler.scale_model_input(latent_input, t)
            with torch.no_grad():
                noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + 7.5 * (noise_text - noise_uncond)

            alpha_t = alphas[int(t)]
            alpha_prev = alphas[int(tl[i+1])] if i+1<len(tl) else torch.tensor(1.0, device=device)
            pred_x0 = (latents - (1-alpha_t).sqrt() * noise_pred.half()) / alpha_t.sqrt()
            latents = alpha_prev.sqrt() * pred_x0 + (1-alpha_prev).sqrt() * noise_pred.half()
            path.append(latents.float().cpu().numpy().flatten())

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

def generate(ts_use, num, device, prompt_list=None, desc="Gen"):
    pipe = get_pipe(device)
    alphas = pipe.scheduler.alphas_cumprod.to(device)
    valid = sorted([int(t) for t in ts_use if 0<=int(t)<1000], reverse=True)
    if prompt_list is None: prompt_list = PROMPTS
    all_images = []

    for bi in tqdm(range(num), desc=desc):
        prompt = prompt_list[bi % len(prompt_list)]
        g = torch.Generator(device=device).manual_seed(20000+bi)
        text_input = pipe.tokenizer(prompt, padding="max_length",
                                     max_length=pipe.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = pipe.tokenizer("", padding="max_length",
                                       max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_emb = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        prompt_emb = torch.cat([uncond_emb, text_emb])

        latents = torch.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
                              generator=g, device=device, dtype=torch.float16)
        for i, t in enumerate(valid):
            latent_input = torch.cat([latents]*2)
            with torch.no_grad():
                noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + 7.5 * (noise_text - noise_uncond)
            alpha_t = alphas[t]
            alpha_prev = alphas[valid[i+1]] if i+1<len(valid) else torch.tensor(1.0, device=device)
            pred_x0 = (latents - (1-alpha_t).sqrt() * noise_pred.half()) / alpha_t.sqrt()
            latents = alpha_prev.sqrt() * pred_x0 + (1-alpha_prev).sqrt() * noise_pred.half()

        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        image = ((image.clamp(-1,1)+1)/2*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        all_images.append(image[0])

    return np.stack(all_images)

def save_imgs(imgs, d):
    from PIL import Image
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs): Image.fromarray(im).save(d/f"{i:05d}.png")

def compute_fid(gd, rd):
    from cleanfid import fid as cf
    return cf.compute_fid(str(gd), str(rd))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_batch", type=int, default=30)
    p.add_argument("--pool_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--skip_record", action="store_true")
    a = p.parse_args()

    rdir = Path("results/sd15"); tdir = rdir/"trajectories"; sdir = rdir/"samples"
    fdir = Path("figures")
    for d in [rdir, sdir, fdir]: d.mkdir(parents=True, exist_ok=True)

    if a.skip_record and (tdir / "ddim_timesteps.npy").exists():
        trajs, dts = load_trajectories(tdir)
    else:
        if tdir.exists(): shutil.rmtree(tdir)
        trajs, dts = record_trajectories(a.ref_batch, a.pool_steps, a.device, tdir)

    pool_spacing = int(np.median(np.abs(np.diff(dts))))
    print(f"Pool spacing: {pool_spacing}")

    print(f"\n[Phase 2] GeoSPRINT budget search...")
    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    target_Ks = list(range(10, min(len(dts)+1, 51), 5))
    budget_results = []

    for target_K in target_Ks:
        all_K = []; all_alpha = []; all_retained = []
        for traj in normalized:
            tau, res = find_threshold_for_K(traj, target_K, tol=3)
            all_K.append(len(res.retained_indices))
            all_alpha.append(res.projection_score)
            all_retained.append(res.retained_indices)
        median_K = int(np.median(all_K))
        mean_alpha = float(np.mean(all_alpha))
        N_traj = len(dts)+1
        counts = np.zeros(N_traj)
        for ret in all_retained:
            for idx in ret:
                if idx < N_traj: counts[idx] += 1
        top_idx = np.sort(np.argsort(-counts)[:median_K])
        geo_ts = [int(dts[idx-1]) for idx in top_idx if idx>0 and idx-1<len(dts)]
        if int(dts[0]) not in geo_ts: geo_ts.insert(0, int(dts[0]))
        if int(dts[-1]) not in geo_ts: geo_ts.append(int(dts[-1]))
        geo_ts = sorted(set(geo_ts), reverse=True)
        K_raw = len(geo_ts)

        # Gap constraint (skip if dense)
        if K_raw >= len(dts) // 2:
            hybrid_ts = geo_ts
        else:
            max_gap = max(T // K_raw, pool_spacing * 2)
            hybrid_ts = enforce_min_spacing(geo_ts, max_gap)
        Kh = len(hybrid_ts)

        budget_results.append({'target_K': target_K, 'K_raw': K_raw, 'Kh': Kh,
                                'alpha': mean_alpha, 'timesteps': hybrid_ts})
        print(f"  target_K={target_K:>3d}  K_raw={K_raw:>3d}  Kh={Kh:>3d}  α={mean_alpha:.2e}")
        if mean_alpha < 1e-6:
            print("  → diminishing returns, stopping"); break

    print(f"\n[Phase 3] Generating {a.num_samples} samples...")

    # Baseline
    print(f"\n  --- Baseline: DDIM-{len(dts)} ---")
    base_dir = sdir / "baseline"
    if not base_dir.exists():
        imgs = generate(dts.tolist(), a.num_samples, a.device, desc=f"DDIM-{len(dts)}")
        save_imgs(imgs, base_dir); del imgs

    all_results = []
    for br in budget_results:
        Kh = br['Kh']; alpha = br['alpha']; ts = br['timesteps']
        print(f"\n  --- GeoSPRINT+gap Kh={Kh} (α={alpha:.2e}) ---")

        gd = sdir / f"geosprint_gap_K{Kh}"
        if not gd.exists():
            imgs = generate(ts, a.num_samples, a.device, desc=f"GeoSPRINT+gap K={Kh}")
            save_imgs(imgs, gd); del imgs

        from diffusers import DDIMScheduler
        sc = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler"); sc.set_timesteps(Kh)
        dd = sdir / f"ddim_K{Kh}"
        if not dd.exists():
            imgs = generate(sc.timesteps.cpu().numpy().tolist(), a.num_samples, a.device,
                           desc=f"DDIM K={Kh}")
            save_imgs(imgs, dd); del imgs

        geo_fid = compute_fid(gd, base_dir)
        ddim_fid = compute_fid(dd, base_dir)
        print(f"    GeoSPRINT+gap K={Kh}: FID={geo_fid:.2f}")
        print(f"    DDIM          K={Kh}: FID={ddim_fid:.2f}")
        all_results.append({'method':'GeoSPRINT+gap','nfe':Kh,'fid':geo_fid,'alpha':alpha})
        all_results.append({'method':'DDIM','nfe':Kh,'fid':ddim_fid,'alpha':0})
        torch.cuda.empty_cache()

    with open(rdir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'Method':<20s} {'NFE':>5s} {'FID':>8s}")
    print("-" * 38)
    for r in sorted(all_results, key=lambda x: (x['nfe'], x['method'])):
        print(f"{r['method']:<20s} {r['nfe']:>5d} {r['fid']:>8.2f}")
    print(f"\nSaved to results/sd15/results.json")

if __name__ == "__main__":
    main()
