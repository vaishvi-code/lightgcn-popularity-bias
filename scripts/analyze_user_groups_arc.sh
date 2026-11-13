#!/bin/bash
#SBATCH --job-name=user-groups
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
set -euo pipefail
source_root="$HOME/lightgcn-user-groups-20261112"
project_root="/mnt/beegfs/$USER/lightgcn-work"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python" "$source_root/tests/check_user_groups.py"
"$python" -u "$source_root/scripts/analyze_user_groups.py" \
  --repository "$source_root" --project-root "$project_root" \
  --checkpoint-root "$project_root/results/gowalla" \
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-appnp-20261105/results/gowalla \
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-normalization-20261109/results/gowalla \
  --output "$source_root/results/user-groups-job${SLURM_JOB_ID}"
