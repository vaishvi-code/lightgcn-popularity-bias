#!/bin/bash
#SBATCH --job-name=single-recover
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=3-4%2

set -euo pipefail
source_root="$HOME/lightgcn-supplemental-20261111"
project_root="/mnt/beegfs/$USER/lightgcn-supplemental-20261111"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
layers="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
case "$layers" in 3|4) ;; *) echo 'Only interrupted depths 3 and 4 need recovery' >&2; exit 1 ;; esac
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$project_root"
original="$project_root/results/gowalla/lightgcn-single-k${layers}-seed2020-job340255_${layers}"
inputs="$project_root/recovery-inputs/job${SLURM_ARRAY_JOB_ID}_${layers}"
"$python" "$source_root/scripts/prepare_interrupted_resume.py" --source-run "$original" --output "$inputs"
run="$project_root/results/gowalla/lightgcn-single-k${layers}-seed2020-resumed-job${SLURM_ARRAY_JOB_ID}_${layers}"
"$python" -u scripts/train_validation.py --run-dir "$run" --resume "$inputs/last.pt" \
  --layer-combination single --dataset gowalla --model lgn --layer "$layers" --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 --epochs 500 \
  --early-stopping-patience 20 --seed 2020 --topks '[20]' \
  --testbatch 256 --tensorboard 0 --multicore 0
"$python" -u scripts/evaluate_lightgcn.py --run-dir "$run"
