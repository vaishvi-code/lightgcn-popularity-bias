#!/bin/bash
#SBATCH --job-name=lightgcn-k3
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

# Set PROJECT_ROOT to the absolute path of this repository before submission.
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository path before submitting}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$PROJECT_ROOT"
"/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python" \
  -u scripts/train_validation.py \
  --dataset gowalla \
  --model lgn \
  --layer 3 \
  --recdim 64 \
  --bpr_batch 1024 \
  --lr 0.001 \
  --decay 0.0001 \
  --epochs 200 \
  --seed 2020 \
  --topks "[20]" \
  --testbatch 256 \
  --tensorboard 0 \
  --multicore 0
