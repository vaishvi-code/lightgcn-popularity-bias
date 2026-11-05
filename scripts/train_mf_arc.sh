#!/bin/bash
#SBATCH --job-name=mf-k0
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository path before submitting}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PROJECT_ROOT"

PYTHON="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
RUN_DIR="$PROJECT_ROOT/results/gowalla/mf-k0-seed2020-job${SLURM_JOB_ID}"

"$PYTHON" -u "${TRAINING_SCRIPT:-scripts/train_validation.py}" \
  --run-dir "$RUN_DIR" \
  --dataset gowalla --model lgn --layer 0 --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 \
  --epochs "${MAX_EPOCHS:-500}" --early-stopping-patience 20 \
  --seed 2020 --topks "[20]" --testbatch 256 \
  --tensorboard 0 --multicore 0

"$PYTHON" -u "${EVALUATION_SCRIPT:-scripts/evaluate_lightgcn.py}" --run-dir "$RUN_DIR"
