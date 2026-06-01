"""
Extend Experiment 4: DPM-Solver++ at higher NFEs (60, 70, 80, 89)
GeoSPRINT and DDIM results already exist from exp1.

Usage: python scripts/run_exp4_higher.py --device cuda --num_samples 10000
"""

import argparse, json, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

_model_cache = {}

def get_model(device):
    if device not in _model_cache:
        from diffusers import UNet2DModel
        m = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
        m.eval()
        _model_cache[device] = m
    return _model_cache[device]

def to_images(latents):
    return ((latents.clamp(-1,1)+1)/2*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

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
    from cleanfid import fid as cf
    return cf.compute_fid(str(gd), str(rd))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()

    ref_dir = Path("results/fid_stats/cifar10_train_images")
    sdir = Path("results/exp4/samples")
    sdir.mkdir(parents=True, exist_ok=True)

    # NFEs to add (already have 10, 20, 30, 50)
    nfes = [60, 70, 80, 89]
    results = []

    # Load existing exp4 results
    exp4_file = Path("results/exp4/exp4_results.json")
    if exp4_file.exists():
        existing = json.load(open(exp4_file))
        print(f"Loaded {len(existing)} existing results")
    else:
        existing = []

    for nfe in nfes:
        print(f"\n  --- DPM++ NFE={nfe} ---")
        dpm_dir = sdir / f"dpm_default_{nfe}"
        if not dpm_dir.exists():
            imgs = generate_dpmsolver(nfe, a.num_samples, a.device, a.batch_size,
                                      f"DPM++ K={nfe}")
            save_imgs(imgs, dpm_dir); del imgs
        fid = compute_fid(dpm_dir, ref_dir)
        print(f"    DPM++ K={nfe}: FID={fid:.2f}")
        results.append({'method': 'DPM-Solver++', 'nfe': nfe, 'fid': fid})
        torch.cuda.empty_cache()

    # Merge with existing
    combined = existing + results
    seen = set()
    deduped = []
    for r in combined:
        key = (r['method'], r['nfe'])
        if key not in seen:
            deduped.append(r)
            seen.add(key)
    deduped.sort(key=lambda x: (x['nfe'], x['method']))

    with open(exp4_file, 'w') as f:
        json.dump(deduped, f, indent=2)

    # Print full comparison using exp1 GeoSPRINT/DDIM results
    exp1_file = Path("results/exp1/exp1_results.json")
    exp1 = json.load(open(exp1_file)) if exp1_file.exists() else []

    print("\n" + "=" * 65)
    print("FULL COMPARISON (exp4 + exp1)")
    print("=" * 65)
    print(f"\n  {'NFE':>5s}  {'DPM++':>8s}  {'GeoSPRINT':>10s}  {'DDIM':>8s}")
    print(f"  {'-'*37}")

    all_nfes = sorted(set([r['nfe'] for r in deduped if r['method']=='DPM-Solver++']))
    for nfe in all_nfes:
        dpm = next((r['fid'] for r in deduped if r['method']=='DPM-Solver++' and r['nfe']==nfe), None)
        geo = next((r['fid'] for r in exp1 if r['method']=='GeoSPRINT' and abs(r['nfe']-nfe)<=1), None)
        ddim = next((r['fid'] for r in exp1 if r['method']=='DDIM' and abs(r['nfe']-nfe)<=1), None)
        dstr = f"{dpm:.2f}" if dpm else "---"
        gstr = f"{geo:.2f}" if geo else "---"
        ustr = f"{ddim:.2f}" if ddim else "---"
        print(f"  {nfe:>5d}  {dstr:>8s}  {gstr:>10s}  {ustr:>8s}")

    print(f"\nSaved to results/exp4/exp4_results.json")

if __name__ == "__main__":
    main()
