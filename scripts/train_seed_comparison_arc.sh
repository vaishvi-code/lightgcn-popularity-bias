#!/bin/bash
#SBATCH --job-name=gowalla-s2021
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
set -euo pipefail
source_root="$HOME/lightgcn-normalization-20261109"
project_root="/mnt/beegfs/$USER/lightgcn-normalization-20261109"
python="/mnt/beegfs/$USER/conda/envs/lightgcn-appnp/bin/python"
seed="${TRAINING_SEED:-2021}"
case "$seed" in 2021|2022) ;; *) echo 'TRAINING_SEED must be 2021 or 2022' >&2; exit 1 ;; esac
variant_args=()
case "${SLURM_ARRAY_TASK_ID:?Submit as an array}" in
  0) layers=0; prefix=mf-k0 ;;
  1|2|3|4) layers="$SLURM_ARRAY_TASK_ID"; prefix="lightgcn-k${layers}" ;;
  5) layers=3; prefix=lightgcn-appnp-beta0.5-k3; variant_args=(--appnp-beta 0.5) ;;
  *) echo 'Invalid task index' >&2; exit 1 ;;
esac
# Reuse the verified environment, data and model code without changing old runs.
for script in train_validation.py evaluate_lightgcn.py weighted_lightgcn.py normalized_graph.py; do
  cmp "$source_root/scripts/$script" "$project_root/scripts/$script"
done
cd "$project_root"
test "$(git -C LightGCN-PyTorch rev-parse HEAD)" = "$(cat provenance/upstream-commit.txt)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python" - <<'PY'
import hashlib,json
from pathlib import Path
reference=json.loads(Path('results/gowalla/lightgcn-k3-fresh-seed2020-job329225/config.json').read_text())
for filename,key in [('train.txt','fitting_file_sha256'),('val.txt','validation_file_sha256')]:
    assert hashlib.sha256((Path('prepared/gowalla')/filename).read_bytes()).hexdigest()==reference[key]
print('Fixed split hashes match seed-2020 control')
PY
run="$project_root/results/gowalla/${prefix}-seed${seed}-job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Variant=$prefix training_seed=$seed layers=$layers max_epochs=500 patience=20"
"$python" -u scripts/train_validation.py --run-dir "$run" \
  "${variant_args[@]}" --norm-a 0.5 --norm-b 0.5 \
  --dataset gowalla --model lgn --layer "$layers" --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 \
  --epochs 500 --early-stopping-patience 20 --seed "$seed" \
  --topks '[20]' --testbatch 256 --tensorboard 0 --multicore 0
"$python" -u scripts/evaluate_lightgcn.py --run-dir "$run"
export_dir="$source_root/results/gowalla/${run##*/}"
mkdir -p "$export_dir"
cp "$run"/{config.json,history.csv,run-summary.json,best-validation-metrics.json,best-validation-exposure.csv} "$export_dir/"
echo "ARTIFACTS_EXPORTED=$export_dir"
