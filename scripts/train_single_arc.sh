#!/bin/bash
#SBATCH --job-name=lightgcn-single
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=1-4%2

set -euo pipefail
source_root="$HOME/lightgcn-supplemental-20261111"
project_root="/mnt/beegfs/$USER/lightgcn-supplemental-20261111"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
layers="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
case "$layers" in 1|2|3|4) ;; *) echo 'Invalid layer count' >&2; exit 1 ;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for script in train_validation.py evaluate_lightgcn.py weighted_lightgcn.py normalized_graph.py single_lightgcn.py; do
    cmp "$source_root/scripts/$script" "$project_root/scripts/$script"
done
cd "$project_root"
run="$project_root/results/gowalla/lightgcn-single-k${layers}-seed2020-job${SLURM_ARRAY_JOB_ID}_${layers}"
"$python" -u scripts/train_validation.py --run-dir "$run" \
  --layer-combination single --dataset gowalla --model lgn --layer "$layers" --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 --epochs 500 \
  --early-stopping-patience 20 --seed 2020 --topks '[20]' \
  --testbatch 256 --tensorboard 0 --multicore 0
"$python" -u scripts/evaluate_lightgcn.py --run-dir "$run"
