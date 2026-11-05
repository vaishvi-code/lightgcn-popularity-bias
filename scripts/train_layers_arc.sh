#!/bin/bash
#SBATCH --job-name=lightgcn-depth
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository path before submitting}"
: "${LAYERS:?Set LAYERS to 1, 2, 3, or 4}"
case "$LAYERS" in
  1|2|3|4) ;;
  *) echo "LAYERS must be 1, 2, 3, or 4" >&2; exit 1 ;;
esac
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PROJECT_ROOT"

PYTHON="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
SEED="${TRAINING_SEED:-2020}"
RUN_DIR="$PROJECT_ROOT/results/gowalla/lightgcn-k${LAYERS}-seed${SEED}-job${SLURM_JOB_ID}"

"$PYTHON" -u "${TRAINING_SCRIPT:-scripts/train_validation.py}" \
  --run-dir "$RUN_DIR" \
  --dataset gowalla --model lgn --layer "$LAYERS" --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 \
  --epochs 500 --early-stopping-patience 20 \
  --seed "$SEED" --topks "[20]" --testbatch 256 \
  --tensorboard 0 --multicore 0

"$PYTHON" -u "${EVALUATION_SCRIPT:-scripts/evaluate_lightgcn.py}" --run-dir "$RUN_DIR"
