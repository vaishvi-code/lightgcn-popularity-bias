#!/bin/bash
#SBATCH --job-name=single-user-item
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
set -euo pipefail
source_root="${SOURCE_ROOT:-$HOME/lightgcn-single-users-items-20261122}"
project_root="/mnt/beegfs/$USER/lightgcn-work"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
output="$source_root/results/single-users-items-job${SLURM_JOB_ID}"
"$python" -u "$source_root/scripts/analyze_single_users_items.py" \
  --repository "$source_root" --project-root "$project_root" \
  --checkpoint-root "$project_root/results/gowalla" \
  --checkpoint-root "/mnt/beegfs/$USER/lightgcn-supplemental-20261111/results/gowalla" \
  --checkpoint-root "/mnt/beegfs/$USER/lightgcn-proposal-completion-20261112/results/gowalla" \
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-appnp-20261105/results/gowalla \
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-normalization-20261109/results/gowalla \
  --output "$output"
"$python" "$source_root/scripts/audit_single_users_items.py" "$output"
"$python" "$source_root/scripts/plot_single_users_items.py" "$output"
