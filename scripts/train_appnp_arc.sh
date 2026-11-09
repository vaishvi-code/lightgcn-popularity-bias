#!/bin/bash
#SBATCH --job-name=appnp-b05
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
set -euo pipefail
project_root="/mnt/beegfs/$USER/lightgcn-appnp-20261105"
for script in train_validation.py evaluate_lightgcn.py weighted_lightgcn.py; do
  if ! cmp -s "$HOME/lightgcn-appnp-20261105/scripts/$script" "$project_root/scripts/$script"; then
    echo "Deploy matching script copies before submission: $script" >&2
    exit 1
  fi
done
cd "$project_root"
test "$(git -C LightGCN-PyTorch rev-parse HEAD)" = "$(cat provenance/upstream-commit.txt)"
cp provenance/appnp-environment-freeze.txt "$HOME/lightgcn-appnp-20261105/provenance/"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="/mnt/beegfs/$USER/conda/envs/lightgcn-appnp/bin/python"
beta="${APPNP_BETA:-0.5}"
seed="${TRAINING_SEED:-2020}"
variant_args=()
prefix="lightgcn-k3-fresh"
if [[ "$beta" != "uniform" ]]; then
  variant_args=(--appnp-beta "$beta")
  prefix="lightgcn-appnp-beta${beta}-k3"
fi
run="$project_root/results/gowalla/${prefix}-seed${seed}-job${SLURM_JOB_ID}"
echo "Variant=$beta seed=$seed max_epochs=${MAX_EPOCHS:-500} patience=20"
"$python" -u scripts/train_validation.py --run-dir "$run" \
  "${variant_args[@]}" --dataset gowalla --model lgn --layer 3 --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 \
  --epochs "${MAX_EPOCHS:-500}" --early-stopping-patience 20 \
  --seed "$seed" --topks '[20]' --testbatch 256 --tensorboard 0 --multicore 0
"$python" -u scripts/evaluate_lightgcn.py --run-dir "$run"
export_dir="$HOME/lightgcn-appnp-20261105/results/gowalla/${run##*/}"
mkdir -p "$export_dir"
cp "$run"/{config.json,history.csv,run-summary.json,best-validation-metrics.json,best-validation-exposure.csv} "$export_dir/"
echo "ARTIFACTS_EXPORTED=$export_dir"
