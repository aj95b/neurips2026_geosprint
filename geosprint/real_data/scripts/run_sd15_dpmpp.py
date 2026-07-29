"""
Rebuttal experiment: DPM-Solver++ on Stable Diffusion v1.5.

Generates DPM-Solver++ samples at the SAME NFEs as the existing
GeoSPRINT / DDIM runs (8, 15, 20, 24, 29, 34, 39, 44, 51) using the
SAME prompts, seeds, and guidance scale, then computes FID against
the SAME 50-step DDIM baseline directory.

This closes the AC's critical concern #2: extending the
"schedule quality vs solver order" comparison to a high-resolution model.

Critical fix: fresh DPMSolverMultistepScheduler per image to reset
the internal step_index (same bug pattern as CIFAR-10 exp4).

Usage:
    python scripts/run_sd15_dpmpp.py --device cuda:0 --num_samples 5000
"""

import argparse, json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LATENT_CH = 4
LATENT_SZ = 64

# MUST match run_sd15_logsnr.py exactly
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

# NFEs matching the existing GeoSPRINT/DDIM SD runs
NFES = [8, 15, 20, 24, 29, 34, 39, 44, 51]

_pipe_cache = {}

def get_pipe(device):
    if device not in _pipe_cache:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        _pipe_cache[device] = pipe
    return _pipe_cache[device]


def generate_dpmpp(num_steps, num, device, desc="Gen"):
    """
    Generate with DPM-Solver++ (order 2) and its default schedule.
    Fresh scheduler per image resets step_index. Uses the same prompt
    rotation, seed offset (20000+i), and guidance scale (7.5) as the
    GeoSPRINT/DDIM SD generation path.
    """
    from diffusers import DPMSolverMultistepScheduler
    pipe = get_pipe(device)

    # Precompute text embeddings once per prompt (they don't change)
    emb_cache = {}
    def get_emb(prompt):
        if prompt not in emb_cache:
            ti = pipe.tokenizer(prompt, padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
            with torch.no_grad():
                te = pipe.text_encoder(ti.input_ids.to(device))[0]
            ui = pipe.tokenizer("", padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                return_tensors="pt")
            with torch.no_grad():
                ue = pipe.text_encoder(ui.input_ids.to(device))[0]
            emb_cache[prompt] = torch.cat([ue, te])
        return emb_cache[prompt]

    all_images = []
    for bi in tqdm(range(num), desc=desc):
        prompt = PROMPTS[bi % len(PROMPTS)]
        prompt_emb = get_emb(prompt)

        # Fresh scheduler each image -> resets internal step_index
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler",
            algorithm_type="dpmsolver++", solver_order=2,
        )
        scheduler.config.lower_order_final = True
        scheduler.set_timesteps(num_steps, device=device)

        g = torch.Generator(device=device).manual_seed(20000 + bi)
        latents = torch.randn(1, LATENT_CH, LATENT_SZ, LATENT_SZ,
                              generator=g, device=device, dtype=torch.float16)
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            latent_input = scheduler.scale_model_input(latent_input, t)
            with torch.no_grad():
                noise_pred = pipe.unet(latent_input, t,
                                       encoder_hidden_states=prompt_emb).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + 7.5 * (noise_text - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        image = ((image.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        all_images.append(image[0])

    return np.stack(all_images)


def save_imgs(imgs, d):
    from PIL import Image
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        Image.fromarray(im).save(d / f"{i:05d}.png")


def compute_fid(gd, rd):
    from cleanfid import fid as cf
    return cf.compute_fid(str(gd), str(rd))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    rdir = Path("results/sd15")
    sdir = rdir / "samples"
    ref_dir = sdir / "baseline"   # 50-step DDIM reference (same as main SD run)

    if not ref_dir.exists():
        print(f"ERROR: baseline directory {ref_dir} not found. Run run_sd15_logsnr.py first.")
        return

    print("Rebuttal: DPM-Solver++ on Stable Diffusion v1.5")
    print("=" * 55)

    results = []
    for nfe in NFES:
        print(f"\n  --- DPM-Solver++ NFE={nfe} ---")
        dpm_dir = sdir / f"dpmpp_K{nfe}"
        if not dpm_dir.exists():
            imgs = generate_dpmpp(nfe, a.num_samples, a.device, f"DPM++ K={nfe}")
            save_imgs(imgs, dpm_dir); del imgs
        fid = compute_fid(dpm_dir, ref_dir)
        print(f"    DPM-Solver++ K={nfe}: FID={fid:.2f}")
        results.append({"method": "DPM-Solver++", "nfe": nfe, "fid": fid})
        torch.cuda.empty_cache()

    # Merge with existing GeoSPRINT/DDIM results for a full comparison table
    logsnr_file = rdir / "results_logsnr.json"
    existing = json.load(open(logsnr_file)) if logsnr_file.exists() else []

    with open(rdir / "results_dpmpp.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print combined table
    print("\n" + "=" * 60)
    print("FULL COMPARISON — Stable Diffusion v1.5")
    print("=" * 60)
    print(f"\n  {'NFE':>4s}  {'GeoSPRINT':>10s}  {'DDIM':>8s}  {'DPM++':>8s}")
    print(f"  {'-'*36}")
    for nfe in NFES:
        geo = next((r['fid'] for r in existing if r['method']=='GeoSPRINT' and r['nfe']==nfe), None)
        ddim = next((r['fid'] for r in existing if r['method']=='DDIM' and r['nfe']==nfe), None)
        dpm = next((r['fid'] for r in results if r['nfe']==nfe), None)
        gs = f"{geo:.2f}" if geo is not None else "---"
        ds = f"{ddim:.2f}" if ddim is not None else "---"
        ps = f"{dpm:.2f}" if dpm is not None else "---"
        print(f"  {nfe:>4d}  {gs:>10s}  {ds:>8s}  {ps:>8s}")

    print(f"\nSaved to results/sd15/results_dpmpp.json")


if __name__ == "__main__":
    main()
