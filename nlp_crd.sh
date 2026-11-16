#!/bin/bash
#SBATCH --job-name=nlp_distill
#SBATCH --output=/scratch/%u/logs/nlp_%j.out
#SBATCH --error=/scratch/%u/logs/nlp_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module purge
module load anaconda3
conda activate crd_env

cd /scratch/$USER/RepDistiller
python train_nlp_crd.py --student_type transformer --task sst2 --beta 0.8