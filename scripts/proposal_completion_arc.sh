#!/bin/bash
#SBATCH --job-name=proposal-followup
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
set -euo pipefail
source_root="$HOME/lightgcn-proposal-completion-20261112"
project_root="/mnt/beegfs/$USER/lightgcn-proposal-completion-20261112"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python" -u "$source_root/scripts/proposal_completion.py" run \
  --source "$source_root" --project "$project_root" \
  --group "${FOLLOWUP_GROUP:?Specify grid, replication or selected}" --index "${SLURM_ARRAY_TASK_ID:?Array required}"
