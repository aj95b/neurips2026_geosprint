"""
Step 0B: Download CIFAR-10 and precompute FID reference statistics.

Usage: python scripts/setup_fid_stats.py
Time:  ~10 minutes
GPU:   Recommended but not required
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    out_dir = Path("results/fid_stats")
    img_dir = out_dir / "cifar10_train_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Downloading CIFAR-10 train set...")
    import torchvision
    dataset = torchvision.datasets.CIFAR10(root="data/", train=True, download=True)

    print(f"[2/3] Saving {len(dataset)} images to {img_dir}...")
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, _) in enumerate(tqdm(dataset)):
        img.save(img_dir / f"{i:05d}.png")

    print("[3/3] Computing Inception statistics...")
    try:
        from cleanfid import fid as cleanfid
        cleanfid.make_custom_stats("cifar10_train", str(img_dir), mode="clean")
        print("  Stats cached by clean-fid")
    except ImportError:
        from pytorch_fid.fid_score import compute_statistics_of_path
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mu, sigma = compute_statistics_of_path(str(img_dir), batch_size=256, device=device, dims=2048)
        np.savez(out_dir / "cifar10_train.npz", mu=mu, sigma=sigma)
        print(f"  Stats saved to {out_dir / 'cifar10_train.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
