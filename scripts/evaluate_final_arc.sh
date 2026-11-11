#!/bin/bash
#SBATCH --job-name=final-verify
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail
source_root="$HOME/lightgcn-final-test-20261111"
project_root="/mnt/beegfs/$USER/lightgcn-normalization-20261109"
python="/mnt/beegfs/$USER/conda/envs/lightgcn-appnp/bin/python"
output="$source_root/results/final-evaluation-job${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python" "$source_root/tests/check_final_metrics.py"
"$python" "$source_root/tests/check_final_summary.py"
common=(--plan "$source_root/provenance/final-test-plan.json" --project-root "$project_root" --output "$output" --checkpoint-store "$source_root/incoming")
# Missing checkpoints or a mismatch prevents the following test command.
"$python" -u "$source_root/scripts/evaluate_final.py" --phase verify "${common[@]}"
if [[ "${RUN_FINAL_TEST:-0}" == "1" ]]; then
  "$python" -u "$source_root/scripts/evaluate_final.py" --phase test "${common[@]}"
  "$python" "$source_root/scripts/summarize_final.py" --plan "$source_root/provenance/final-test-plan.json" --output "$output"
fi
echo "EVALUATION_OUTPUT=$output"
