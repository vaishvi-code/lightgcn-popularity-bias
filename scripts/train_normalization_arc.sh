#!/bin/bash
#SBATCH --job-name=norm-b075
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
set -euo pipefail
source_root="$HOME/lightgcn-normalization-20261109"
project_root="/mnt/beegfs/$USER/lightgcn-normalization-20261109"
reference_root="/mnt/beegfs/$USER/lightgcn-appnp-20261105"
python="/mnt/beegfs/$USER/conda/envs/lightgcn-appnp/bin/python"
# Initialize this isolated experiment workspace once, before parallel submissions.
if [[ ! -d "$project_root" ]]; then
  mkdir -p "$project_root"
  cp -a "$source_root/scripts" "$source_root/tests" "$source_root/provenance" "$source_root/results" "$project_root/"
  cp -a "$reference_root/LightGCN-PyTorch" "$reference_root/prepared" "$project_root/"
fi
for script in train_validation.py evaluate_lightgcn.py weighted_lightgcn.py normalized_graph.py; do
  cmp "$source_root/scripts/$script" "$project_root/scripts/$script"
done
cd "$project_root"
test "$(git -C LightGCN-PyTorch rev-parse HEAD)" = "$(cat provenance/upstream-commit.txt)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python" tests/check_normalization.py LightGCN-PyTorch/code
"$python" - <<'PY'
import hashlib,json
from pathlib import Path
ref=json.loads(Path('results/gowalla/lightgcn-k3-fresh-seed2020-job329225/config.json').read_text())
for filename,key in [('train.txt','fitting_file_sha256'),('val.txt','validation_file_sha256')]:
    assert hashlib.sha256((Path('prepared/gowalla')/filename).read_bytes()).hexdigest()==ref[key]
print('Split hashes match fresh K=3 control')
PY
a="${NORM_A:-0.5}"
b="${NORM_B:-0.75}"
seed="${TRAINING_SEED:-2020}"
run="$project_root/results/gowalla/lightgcn-norm-a${a}-b${b}-k3-seed${seed}-job${SLURM_JOB_ID}"
echo "Normalization a=$a b=$b seed=$seed max_epochs=${MAX_EPOCHS:-500} patience=20"
"$python" -u scripts/train_validation.py --run-dir "$run" \
  --norm-a "$a" --norm-b "$b" --dataset gowalla --model lgn --layer 3 --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 \
  --epochs "${MAX_EPOCHS:-500}" --early-stopping-patience 20 \
  --seed "$seed" --topks '[20]' --testbatch 256 --tensorboard 0 --multicore 0
"$python" -u scripts/evaluate_lightgcn.py --run-dir "$run"
export_dir="$source_root/results/gowalla/${run##*/}"
mkdir -p "$export_dir"
cp "$run"/{config.json,history.csv,run-summary.json,best-validation-metrics.json,best-validation-exposure.csv} "$export_dir/"
echo "ARTIFACTS_EXPORTED=$export_dir"
