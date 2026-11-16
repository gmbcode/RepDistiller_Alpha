#!/bin/bash
#SBATCH --job-name=crd_sweep
#SBATCH --output=/scratch/%u/crd/logs/sweep_%j.out
#SBATCH --error=/scratch/%u/crd/logs/sweep_%j.err
#SBATCH --time=12:00:00          # 25 runs × ~20 min each = ~8-9h; 12h is safe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # check with: sinfo -s

# ── Activate environment ──────────────────────────────────────────────────────
module purge
module load anaconda3            # Sharanga has anaconda3 (multiple variants)
# If the above fails, try: module load miniconda3

conda activate crd_env           # environment created during setup (see README.md)

# ── Paths ────────────────────────────────────────────────────────────────────
export REPO=/scratch/$USER/crd/RepDistiller
export RESULTS=/scratch/$USER/crd/sweep_results

mkdir -p $RESULTS
mkdir -p /scratch/$USER/crd/logs

cd $REPO

# ── Run sweep ────────────────────────────────────────────────────────────────
python sweep_tau_beta.py \
    --epochs 30 \
    --subset_frac 1.0 \
    --teacher_path ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
    --gpu 0

# Copy results to scratch for persistence
cp -r sweep_results/ $RESULTS/

echo "Sweep done. Results at $RESULTS"
