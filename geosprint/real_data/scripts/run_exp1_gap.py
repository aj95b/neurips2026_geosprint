# Save as scripts/run_exp1_gap.py
# Adds GeoSPRINT+gap to existing exp1 results. ~3 hours.
# Run AFTER run_exp1.py finishes.

import json, numpy as np, torch, shutil
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from geosprint.evaluate import EvalResult

rdir = Path("results/exp1")
sdir = rdir / "samples"
ref_dir = Path("results/fid_stats/cifar10_train_images")

# Load existing results
existing = json.load(open(rdir / "exp1_results.json"))
geo_results = [r for r in existing if r['method'] == 'GeoSPRINT']

# Reuse model/alphas/generate from run_exp1
exec(open("scripts/run_exp1.py").read().split("def main")[0])

T = 1000
new_results = []

for gr in geo_results:
    K = gr['nfe']
    alpha = gr['alpha']
    
    # Load the GeoSPRINT schedule that was used
    geo_dir = sdir / f"geosprint_K{K}"
    if not geo_dir.exists():
        print(f"  Skipping K={K}, no GeoSPRINT samples found")
        continue
    
    # Reconstruct timesteps from the budget search
    # They're in the JSON indirectly; re-extract from trajectories
    trajs, dts = load_trajectories(rdir / "trajectories")
    normalized = normalize_trajectories(trajs)
    tau, res = find_threshold_for_K(normalized[0], K, tol=5)
    
    # Get consensus schedule same way as main script
    N_traj = len(dts) + 1
    from geosprint.core import prune_trajectory
    counts = np.zeros(N_traj)
    for traj in normalized:
        r = prune_trajectory(traj, k=2, threshold=tau)
        for idx in r.retained_indices:
            if idx < N_traj: counts[idx] += 1
    top_idx = np.sort(np.argsort(-counts)[:K])
    geo_ts = sorted([int(dts[idx-1]) for idx in top_idx if idx > 0 and idx-1 < len(dts)], reverse=True)
    if int(dts[0]) not in geo_ts: geo_ts.insert(0, int(dts[0]))
    if int(dts[-1]) not in geo_ts: geo_ts.append(int(dts[-1]))
    geo_ts = sorted(set(geo_ts), reverse=True)
    
    # Apply gap constraint
    max_gap = T // len(geo_ts)
    filled = [geo_ts[0]]
    for i in range(1, len(geo_ts)):
        gap = filled[-1] - geo_ts[i]
        if gap > max_gap:
            n = int(np.ceil(gap / max_gap)) - 1
            for j in range(1, n + 1):
                v = filled[-1] - int(j * gap / (n + 1))
                if v > 0: filled.append(v)
        filled.append(geo_ts[i])
    hybrid_ts = sorted(set(filled), reverse=True)
    Kh = len(hybrid_ts)
    
    print(f"\n  GeoSPRINT+gap K={K}→{Kh} steps")
    
    hdir = sdir / f"geosprint_gap_K{K}_Kh{Kh}"
    if not hdir.exists():
        imgs = generate(hybrid_ts, 10000, "cuda", 64, f"GeoSPRINT+gap K={Kh}")
        save_imgs(imgs, hdir); del imgs
    
    from cleanfid import fid as cf
    h_fid = cf.compute_fid(str(hdir), str(ref_dir))
    print(f"    GeoSPRINT+gap K={Kh}: FID={h_fid:.2f}")
    new_results.append({'method': 'GeoSPRINT+gap', 'nfe': Kh, 'fid': h_fid, 'alpha': alpha})

# Merge and save
all_results = existing + new_results
with open(rdir / "exp1_results_with_gap.json", 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/exp1/exp1_results_with_gap.json")
