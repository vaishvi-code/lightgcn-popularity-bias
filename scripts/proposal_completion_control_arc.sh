#!/bin/bash
#SBATCH --job-name=proposal-control
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
set -euo pipefail
source_root="$HOME/lightgcn-proposal-completion-20261112"
project_root="/mnt/beegfs/$USER/lightgcn-proposal-completion-20261112"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
args=()
if [[ "${FOLLOWUP_PHASE:?}" == select ]]; then
  args=(--replication-job "${REPLICATION_JOB:?}")
fi
"/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python" -u "$source_root/scripts/proposal_completion.py" "$FOLLOWUP_PHASE" \
  --source "$source_root" --project "$project_root" "${args[@]}"
