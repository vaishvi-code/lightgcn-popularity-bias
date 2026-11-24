#!/bin/bash
#SBATCH --job-name=extended-test
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
set -euo pipefail
source_root="$HOME/lightgcn-extended-test-20261123"
project_root="/mnt/beegfs/$USER/lightgcn-work"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
output="$source_root/results/extended-test-job${SLURM_JOB_ID}"
common=(--repository "$source_root" --project-root "$project_root" --output "$output"
  --checkpoint-root "$project_root/results/gowalla"
  --checkpoint-root "/mnt/beegfs/$USER/lightgcn-supplemental-20261111/results/gowalla"
  --checkpoint-root "/mnt/beegfs/$USER/lightgcn-proposal-completion-20261112/results/gowalla"
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-appnp-20261105/results/gowalla
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-normalization-20261109/results/gowalla)
"$python" "$source_root/tests/check_extended_test_gate.py"
"$python" -u "$source_root/scripts/evaluate_extended_test.py" --phase verify "${common[@]}"
"$python" -u "$source_root/scripts/evaluate_extended_test.py" --phase test "${common[@]}"
