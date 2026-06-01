#!/bin/bash
# ──────────────────────────────────────────────────────────
# GeoSPRINT: Submit All Experiments to SLURM
# Cluster: gpu + highmem partitions, python/3.11.4, cuda/12.9
#
# Usage: bash slurm/submit_all.sh
# ──────────────────────────────────────────────────────────

set -e
mkdir -p logs figures results

# ── Job 0: Setup ──
JOB0=$(sbatch --parsable << 'SETUP'
#!/bin/bash
#SBATCH --job-name=geo_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
module load python/3.11.4
module load cuda/12.9
cd $SLURM_SUBMIT_DIR
pip install --user -e .
pip install --user diffusers accelerate transformers safetensors
pip install --user clean-fid pytorch-fid torchvision matplotlib scipy tqdm
python tests/test_core.py
python scripts/setup_fid_stats.py
SETUP
)
echo "Job 0 (setup):    $JOB0"

# ── Job 1: Exp1 — FID vs NFE ──
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 << 'EXP1'
#!/bin/bash
#SBATCH --job-name=geo_exp1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err
module load python/3.11.4
module load cuda/12.9
cd $SLURM_SUBMIT_DIR
python scripts/run_exp1.py --device cuda --ref_batch 100 --num_samples 10000
EXP1
)
echo "Job 1 (exp1/FID): $JOB1  (after $JOB0)"

# ── Job 2: CPU experiments (exp2, exp5, exp6) ──
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 << 'CPU'
#!/bin/bash
#SBATCH --job-name=geo_cpu
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_%j.out
#SBATCH --error=logs/cpu_%j.err
module load python/3.11.4
cd $SLURM_SUBMIT_DIR
python scripts/run_exp2.py
python scripts/run_exp5.py
python scripts/run_exp6.py
CPU
)
echo "Job 2 (exp2/5/6): $JOB2  (after $JOB1)"

# ── Job 3: Exp3 — α_traj (parallel with Job 1) ──
JOB3=$(sbatch --parsable --dependency=afterok:$JOB0 << 'EXP3'
#!/bin/bash
#SBATCH --job-name=geo_exp3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err
module load python/3.11.4
module load cuda/12.9
cd $SLURM_SUBMIT_DIR
python scripts/run_exp3.py --device cuda
EXP3
)
echo "Job 3 (exp3):     $JOB3  (after $JOB0, parallel with $JOB1)"

# ── Job 4: Collect ──
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 << 'COLLECT'
#!/bin/bash
#SBATCH --job-name=geo_collect
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
module load python/3.11.4
cd $SLURM_SUBMIT_DIR
python scripts/collect.py
COLLECT
)
echo "Job 4 (collect):  $JOB4  (after $JOB2 and $JOB3)"

echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    tail -f logs/exp1_${JOB1}.out"
