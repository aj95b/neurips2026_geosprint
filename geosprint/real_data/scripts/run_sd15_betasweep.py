"""
Rebuttal experiment: blend parameter (beta) sweep on Stable Diffusion v1.5.

Answers reviewer point 5 (7G9F, eXSn) and AC point on robustness:
"beta blending parameter is only tuned primarily on CIFAR-10... no strong
evidence that parameter choices generalize across architectures."

Sweeps beta in {0.3, 0.5, 0.6, 0.7, 0.9} at a representative subset of NFEs
(20, 34, 44) on SD v1.5, reusing the saved reference trajectories and the
50-step DDIM baseline. Confirms whether beta=0.6 (chosen on CIFAR-10)
transfers to a latent-space text-to-image model.

Usage:
    python scripts/run_sd15_betasweep.py --device cuda:0 --num_samples 2000 --skip_record
"""

import argparse, json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geosprint.core import prune_trajectory

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LATENT_CH = 4
LATENT_SZ = 64

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

# Representative NFEs (subset of the main SD run to keep cost down)
NFES = [20, 34, 44]
BETAS = [0.3, 0.5, 0.6, 0.7, 0.9]

_pipe_cache = {}

def get_pipe(device):
    if device not in _pipe_cache:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        _pipe_cache[device] = pipe
    return _pipe_cache[device]


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


def compute_retention_frequency(normalized_trajs, pool_timesteps, target_K=25):
    N_traj = len(pool_timesteps) + 1
    counts = np.zeros(N_traj)
    for traj in normalized_trajs:
        tau, res = find_threshold_for_K(traj, target_K, tol=3)
        for idx in res.retained_indices:
            if idx < N_traj: counts[idx] += 1
    return counts[1:len(pool_timesteps)+1]


def logsnr_curvature_schedule(alphas_np, retention_freq, pool_timesteps, K, blend):
    logsnr_density = np.ones(len(alphas_np))
    curv_density = np.zeros(len(alphas_np))
    for i, t in enumerate(pool_timesteps):
        t_int = int(t)
        if 0 <= t_int < len(curv_density):
            curv_density[t_int] = retention_freq[i]
    curv_density = gaussian_filter1d(curv_density, sigma=10)
    curv_density = curv_density + 0.01 * curv_density.max()
    combined = ((1-blend) * logsnr_density/logsnr_density.sum() +
                blend * curv_density/curv_density.sum())
    W = np.cumsum(combined); W = W / W[-1]
    timesteps = [min(np.searchsorted(W, q), len(alphas_np)-1) for q in np.linspace(0,1,K)]
    timesteps = sorted(set(timesteps), reverse=True)
    if len(timesteps) < K:
        for t in range(999, -1, -1):
            if t not in timesteps: timesteps.append(t); timesteps = sorted(set(timesteps), reverse=True)
            if len(timesteps) >= K: break
    return timesteps[:K]


def load_trajectories(d):
    d = Path(d)
    trajs = [np.load(f) for f in sorted(d.glob("traj_*.npy"))]
    ts = np.load(d / "ddim_timesteps.npy")
    return trajs, ts


def generate(ts_use, num, device, desc="Gen"):
    pipe = get_pipe(device)
    alphas = pipe.scheduler.alphas_cumprod.to(device)
    valid = sorted([int(t) for t in ts_use if 0<=int(t)<1000], reverse=True)
    emb_cache = {}
    def get_emb(prompt):
        if prompt not in emb_cache:
            ti = pipe.tokenizer(prompt, padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
            with torch.no_grad(): te = pipe.text_encoder(ti.input_ids.to(device))[0]
            ui = pipe.tokenizer("", padding="max_length",
                                max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
            with torch.no_grad(): ue = pipe.text_encoder(ui.input_ids.to(device))[0]
            emb_cache[prompt] = torch.cat([ue, te])
        return emb_cache[prompt]
    all_images = []
    for bi in tqdm(range(num), desc=desc):
        prompt = PROMPTS[bi % len(PROMPTS)]
        prompt_emb = get_emb(prompt)
        g = torch.Generator(device=device).manual_seed(20000+bi)
        latents = torch.randn(1, LATENT_CH, LATENT_SZ, LATENT_SZ,
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
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_record", action="store_true")
    a = p.parse_args()

    rdir = Path("results/sd15"); tdir = rdir/"trajectories"; sdir = rdir/"samples"
    ref_dir = sdir / "baseline"
    if not ref_dir.exists():
        print(f"ERROR: baseline {ref_dir} not found. Run run_sd15_logsnr.py first.")
        return
    if not (tdir / "ddim_timesteps.npy").exists():
        print(f"ERROR: trajectories not found in {tdir}.")
        return

    trajs, dts = load_trajectories(tdir)
    print(f"Loaded {len(trajs)} SD reference trajectories, pool={len(dts)}")

    normalized = []
    for t in trajs:
        mu = t.mean(0, keepdims=True); s = t.std(0, keepdims=True)+1e-8
        normalized.append((t-mu)/s)

    w = compute_retention_frequency(normalized, dts, target_K=25)

    from diffusers import DDIMScheduler
    alphas_np = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler").alphas_cumprod.numpy()

    print("\nBeta sweep on SD v1.5")
    print("=" * 55)

    results = []
    bdir = sdir / "betasweep"
    for nfe in NFES:
        for beta in BETAS:
            geo_ts = logsnr_curvature_schedule(alphas_np, w, dts, nfe, blend=beta)
            K = len(geo_ts)
            tag = f"b{beta:.1f}_K{nfe}"
            gd = bdir / tag
            if not gd.exists():
                imgs = generate(geo_ts, a.num_samples, a.device, f"beta={beta} NFE={nfe}")
                save_imgs(imgs, gd); del imgs
            fid = compute_fid(gd, ref_dir)
            print(f"  NFE={nfe:>3d}  beta={beta:.1f}  FID={fid:.2f}")
            results.append({"nfe": nfe, "beta": beta, "fid": fid})
            torch.cuda.empty_cache()

    with open(rdir / "betasweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print("\n" + "=" * 55)
    print("BETA SWEEP SUMMARY (FID vs 50-step DDIM baseline)")
    print("=" * 55)
    header = "  NFE  " + "  ".join(f"b={b:.1f}" for b in BETAS)
    print(header)
    print("  " + "-" * (len(header)-2))
    for nfe in NFES:
        row = f"  {nfe:>3d}  "
        best_fid = min(r['fid'] for r in results if r['nfe']==nfe)
        for beta in BETAS:
            fid = next(r['fid'] for r in results if r['nfe']==nfe and r['beta']==beta)
            mark = "*" if fid == best_fid else " "
            row += f" {fid:5.2f}{mark}"
        print(row)
    print("\n  (* = best beta at that NFE)")
    print(f"\nSaved to results/sd15/betasweep.json")


if __name__ == "__main__":
    main()
