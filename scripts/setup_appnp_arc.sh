#!/bin/bash
#SBATCH --job-name=appnp-setup
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
set -euo pipefail
source_root="$HOME/lightgcn-appnp-20261105"
project_root="/mnt/beegfs/$USER/lightgcn-appnp-20261105"
mkdir -p "$project_root"
tar --exclude=.git -C "$source_root" -cf - . | tar -C "$project_root" -xf -
cd "$project_root"
export CONDA_PKGS_DIRS="/mnt/beegfs/$USER/conda/pkgs"
env_root="/mnt/beegfs/$USER/conda/envs/lightgcn-appnp"
/usr/bin/conda create --yes --override-channels --channel conda-forge --prefix "$env_root" python=3.10.22
python="$env_root/bin/python"
"$python" -m pip install --no-cache-dir --upgrade pip
"$python" -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$python" -m pip install --no-cache-dir -r requirements.txt
if [ ! -d LightGCN-PyTorch ]; then
  git clone https://github.com/gusye1234/LightGCN-PyTorch.git LightGCN-PyTorch
fi
git -C LightGCN-PyTorch checkout --detach "$(cat provenance/upstream-commit.txt)"
"$python" scripts/prepare_gowalla.py
"$python" - <<'PY'
import hashlib,json
from pathlib import Path
reference = json.loads(Path("results/gowalla/lightgcn-k3-seed2020-20261105-001600-153197/config.json").read_text())
for filename,key in [("train.txt","fitting_file_sha256"),("val.txt","validation_file_sha256")]:
    assert hashlib.sha256((Path("prepared/gowalla")/filename).read_bytes()).hexdigest() == reference[key]
print("Prepared split hashes match the team baseline")
PY
"$python" -m pip freeze > provenance/appnp-environment-freeze.txt
"$python" -c 'import ctypes, torch, numpy; print(torch.__version__, numpy.__version__)'
echo "SETUP_COMPLETE"
