#!/bin/bash
#SBATCH --job-name=supplemental-check
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail
source_root="$HOME/lightgcn-supplemental-20261111"
project_root="/mnt/beegfs/$USER/lightgcn-supplemental-20261111"
original_root="/mnt/beegfs/$USER/lightgcn-work"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$project_root/scripts" "$project_root/prepared/gowalla" "$project_root/results/gowalla"
cp "$source_root"/scripts/*.py "$project_root/scripts/"
if [[ ! -e "$project_root/LightGCN-PyTorch" ]]; then
    ln -s "$original_root/LightGCN-PyTorch" "$project_root/LightGCN-PyTorch"
fi
cp "$original_root/prepared/gowalla/"{train.txt,val.txt} "$project_root/prepared/gowalla/"
cp "$source_root/results/gowalla/"{item-popularity.csv,popularity-summary.json} "$project_root/results/gowalla/"

"$python" "$source_root/tests/check_embedding_metrics.py"
"$python" "$source_root/tests/check_single_lightgcn.py" "$project_root/LightGCN-PyTorch/code"

run="$project_root/results/gowalla/lightgcn-single-k1-seed2020-check-job${SLURM_JOB_ID}"
cd "$project_root"
"$python" -u scripts/train_validation.py --run-dir "$run" \
  --layer-combination single --dataset gowalla --model lgn --layer 1 --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 --epochs 2 \
  --early-stopping-patience 20 --seed 2020 --topks '[20]' \
  --testbatch 256 --tensorboard 0 --multicore 0
"$python" -u scripts/evaluate_lightgcn.py --run-dir "$run"

"$python" -u scripts/analyze_embeddings.py \
  --project-root "$original_root" --repository-root "$source_root" \
  --checkpoint-root "$original_root/results/gowalla" \
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-appnp-20261105/results/gowalla \
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-normalization-20261109/results/gowalla \
  --output "$project_root/results/gowalla/embedding-diagnostics-job${SLURM_JOB_ID}"
echo 'SUPPLEMENTAL_CHECK_PASSED'
