#!/bin/bash
#SBATCH --job-name=item-magnitude
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
set -euo pipefail
root="$HOME/lightgcn-magnitude-20261124"
project="/mnt/beegfs/$USER/lightgcn-work"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
output="$root/results/magnitude-job${SLURM_JOB_ID}"
common=(--repository "$root" --project-root "$project" --output "$output"
  --checkpoint-root "$project/results/gowalla"
  --checkpoint-root "/mnt/beegfs/$USER/lightgcn-supplemental-20261111/results/gowalla"
  --checkpoint-root "/mnt/beegfs/$USER/lightgcn-proposal-completion-20261112/results/gowalla"
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-appnp-20261105/results/gowalla
  --checkpoint-root /mnt/beegfs/stpatel4/lightgcn-normalization-20261109/results/gowalla)
"$python" "$root/tests/check_magnitude_scoring.py"
"$python" -u "$root/scripts/analyze_magnitude.py" --phase verify "${common[@]}"
"$python" -u "$root/scripts/analyze_magnitude.py" --phase analyze "${common[@]}"
