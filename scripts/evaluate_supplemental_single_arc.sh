#!/bin/bash
#SBATCH --job-name=single-test
#SBATCH --partition=rtx4060ti16g
#SBATCH --exclude=c0,c23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

set -euo pipefail
source_root="$HOME/lightgcn-single-test-20261112"
project_root="/mnt/beegfs/$USER/lightgcn-supplemental-20261111"
python="/mnt/beegfs/$USER/conda/envs/lightgcn/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
output="$source_root/results/supplemental-single-test-job${SLURM_JOB_ID}"
common=(--plan "$source_root/provenance/supplemental-single-test-plan.json" --project-root "$project_root" --output "$output")
"$python" "$source_root/tests/check_supplemental_gate.py"
"$python" -u "$source_root/scripts/evaluate_supplemental_single.py" --phase verify "${common[@]}"
"$python" -u "$source_root/scripts/evaluate_supplemental_single.py" --phase test "${common[@]}"
echo "SUPPLEMENTAL_TEST_OUTPUT=$output"
