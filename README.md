# Does LightGCN Favor Popular Items?

CSC 591/791-038: Machine Learning for Graphs — Group 5, Fall 2026.

Team: Sachi Vyas, Sakhi Patel, and Vaishvi Patel.

We study how LightGCN's propagation depth, layer weights, and degree normalization affect recommendation accuracy and exposure for less popular items. The data pipeline, MostPopular baseline, and a 200-epoch three-layer LightGCN run have completed on ARC. The selected LightGCN checkpoint was evaluated for accuracy and popularity exposure on validation data. Results use one training seed and remain preliminary research findings.

## Research questions

1. How does increasing the number of propagation layers affect popularity exposure?
2. Does combining layers behave differently from using only the final layer?
3. How do APPNP-style layer weights change accuracy and tail exposure?
4. What trade-offs arise from stronger degree-based neighbor down-weighting?
5. Do item embedding norms and recommendation frequencies correlate with item degree?

This is a controlled reproduction and ablation study. Popularity amplification in graph recommenders is already studied in prior work; we do not claim to discover it for the first time.

## Start here

- [Completed work and measured results](#results)
- [Setup and ARC commands](#getting-started)
- [Experiment protocol and work split](docs/experiments.md)
- [Result files and provenance](results/README.md)

## Getting started

Use Python 3.10. The observed ARC environment used Python 3.10.22, PyTorch 2.5.1+cu121, and an RTX 4060 Ti with 16 GB GPU memory.

From the repository root, install the CUDA-enabled wheel, followed by the recorded direct dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Download the author-linked upstream code and processed datasets into the ignored checkout directory. Follow its applicable licensing and data-use terms.

```bash
git clone https://github.com/gusye1234/LightGCN-PyTorch.git LightGCN-PyTorch
git -C LightGCN-PyTorch checkout "$(cat provenance/upstream-commit.txt)"
python scripts/prepare_gowalla.py
python scripts/analyze_popularity.py
python scripts/evaluate_mostpopular.py
```

The recorded [upstream commit](provenance/upstream-commit.txt) is `947ca2b3b1d2d3545b114145710cb06c4e57b3d2`. The actual [ARC environment export](provenance/environment-freeze.txt) is included for provenance. Its `packaging` entry references a local Conda build, so use the installation commands above rather than installing that raw export directly. `requirements.txt` records observed direct dependencies rather than a full lockfile.

On an allocated GPU node, run a two-epoch check:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/train_validation.py \
  --dataset gowalla --model lgn --layer 3 --recdim 64 \
  --bpr_batch 1024 --lr 0.001 --decay 0.0001 --epochs 2 \
  --seed 2020 --topks "[20]" --testbatch 256 --tensorboard 0 --multicore 0
```

For ARC batch training, set `PROJECT_ROOT` to this repository's absolute cluster path. Submit from your home filesystem:

```bash
export PROJECT_ROOT="/absolute/path/to/lightgcn-popularity-bias"
cd "$HOME"
sbatch --output="$HOME/lightgcn-k3-%j.out" "$PROJECT_ROOT/scripts/train_arc.sh"
```

The batch script assumes the already-created environment at `/mnt/beegfs/$USER/conda/envs/lightgcn`. Adapt that path for another environment. It starts a fresh run with a 200-epoch maximum. The recorded baseline run has completed; submitting this script starts a new run.

### Optional early stopping

Add `--early-stopping-patience 20` to the Python training command to stop after 20 consecutive validation checks without a new highest Recall@20. Validation runs once per epoch. Any strictly higher recall resets the counter. A value of `0`, the default, runs the full epoch budget.

For the ARC batch script, set the stopping rule before submitting the job.

```bash
export EARLY_STOPPING_PATIENCE=20
sbatch --output="$HOME/lightgcn-k3-%j.out" "$PROJECT_ROOT/scripts/train_arc.sh"
```

The stopping settings are recorded in `config.json`. A completed run also writes `run-summary.json` with its actual epoch count, stopping reason, and best epoch. `best.pt` retains the highest-validation-recall checkpoint and `last.pt` records the final training epoch. Use the same epoch budget and stopping rule for compared model variants and baselines. Re-run the standard baseline when evaluating a new stopping protocol.

### Continue an existing run

Add `--resume /absolute/path/to/last.pt` to the training command. Set `--epochs` to the total epoch limit, such as `500` when continuing after epoch 200. Keep the seed and model configuration identical to the saved run. The wrapper checks the split hashes, restores model and optimizer states, and reconstructs the best checkpoint and stopping counter from the parent history.

The continuation writes a new run directory containing the parent history and additional epochs. It copies the previous best checkpoint into the new directory before training. New checkpoints also save Python, NumPy, and PyTorch random-generator states. The original 200-epoch checkpoint lacks these states, so its continuation restarts sampling from seed 2020. Its future trajectory can differ from uninterrupted training.

The ARC continuation script requests the original node c3, a total limit of 500 epochs, and patience 20. It queues when that node is occupied.

```bash
export PROJECT_ROOT="/absolute/path/to/lightgcn-popularity-bias"
cd "$HOME"
sbatch --output="$HOME/lightgcn-k3-resume-%j.out" "$PROJECT_ROOT/scripts/resume_arc.sh"
```

The training wrapper supplies fitting interactions and validation interactions to the upstream loader in a separate hash-keyed directory. Its `test.txt` is validation data; the supplied final test set is not evaluated. Graph caches are isolated by split hashes. Each run saves `config.json`, `history.csv`, `best.pt` (selected by validation Recall@20), and `last.pt`.

To evaluate a trusted checkpoint on validation data, set `RUN_DIR` to its directory and run on an allocated GPU node:

```bash
python scripts/evaluate_lightgcn.py --run-dir "$RUN_DIR"
```

This writes `best-validation-metrics.json` and `best-validation-exposure.csv` inside that run. Checkpoints remain on ARC; they are not stored on GitHub. The evaluator verifies the split hashes and uses the saved popular/tail item groups.


The repository scripts were recovered from the team's shared terminal logs and adapted to live under `scripts/`. ARC's existing root-level scripts and checkpoints were not changed by this GitHub sync.

## Repository layout

| Path | Contents |
| --- | --- |
| [docs/proposal.pdf](docs/proposal.pdf) | Submitted project proposal |
| [docs/experiments.md](docs/experiments.md) | Shared protocol, planned variants, and independent work tracks |
| [scripts/](scripts/) | Data preparation, analysis, training, evaluation, and ARC submission |
| [configs/](configs/) | Reference experiment specifications |
| [results/README.md](results/README.md) | Completed results and artifact index |
| [results/gowalla/archive/](results/gowalla/archive/) | Earlier snapshots and the two-epoch check |
| [provenance/](provenance/) | Original ARC environment export and upstream commit |
| [data/README.md](data/README.md) | Local data instructions |

Local datasets, environments, graph caches, checkpoints, and presentation exports are excluded from Git.

## Script guide

Run these commands from the repository root. Training and checkpoint evaluation require an allocated GPU node.

| Script | Purpose |
| --- | --- |
| `scripts/prepare_gowalla.py` | Create the fixed fitting/validation split under `prepared/gowalla/` |
| `scripts/analyze_popularity.py` | Save fitting-degree groups and the popularity figure |
| `scripts/evaluate_mostpopular.py` | Evaluate the popularity-count baseline on validation |
| `scripts/train_validation.py` | Train standard LightGCN and save histories and checkpoints |
| `scripts/evaluate_lightgcn.py --run-dir "$RUN_DIR"` | Evaluate a saved standard LightGCN checkpoint for relevance and exposure |
| `scripts/train_arc.sh` | Submit the standard 200-epoch K=3 run through Slurm |
| `scripts/resume_arc.sh` | Continue the K=3 checkpoint on c3, up to epoch 500 with patience 20 |

The ablation implementations are planned in [the experiment guide](docs/experiments.md). The JSON configuration in `configs/` is a reference specification. The scripts currently receive training settings through CLI arguments.

## Results

Results below use the fitting/validation split with seed 2020 and 29,858 validation users. The final test set remains untouched.

| Model | Validation Recall@20 | Validation NDCG@20 | Tail share@20 | Coverage@20 | Exposure Gini |
| --- | ---: | ---: | ---: | ---: | ---: |
| MostPopular (completed) | 0.02715 | 0.01298 | 0% | 0.0903% | 0.99950 |
| LightGCN K=3 (best epoch 192 of 200, one seed) | 0.18968 | 0.09853 | 10.59% | 35.50% | 0.93371 |

The most popular 8,197 of 40,981 items account for 53.09% of the 737,091 fitting interactions. MostPopular recommends only 37 distinct items and has tail Recall@20 of zero among 18,987 users with validation tail positives.

LightGCN recommends 14,549 distinct items and has tail Recall@20 of 0.04615 among the same 18,987 users with validation tail positives. Popular items occupy 89.41% of its recommendation slots. It improves accuracy and catalog coverage over MostPopular, while exposure remains concentrated on popular items. This comparison alone does not isolate the effect of graph propagation; MF and layer-depth comparisons are still needed.

The run completed 200 epochs, with epoch 192 selected by validation Recall@20. These are single-seed validation results, not final test scores or evidence of convergence. See [result files and provenance](results/README.md).

The actual ARC artifact export is now included: [the full 200-epoch history](results/gowalla/lightgcn-k3-seed2020-20261102-164455/history.csv), run configurations, item-level popularity and recommendation counts, environment export, and upstream commit. Exported metrics match the earlier terminal snapshots. The complete history confirms epoch 192 had the highest validation Recall@20. [The metrics template](results/metrics-template.csv) remains header-only. Published scores are not presented as team measurements.

![Gowalla fitting-data popularity distribution](results/gowalla/popularity-distribution.png)

## References

- He et al. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126), 2020.
- Wu et al. [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153), 2019.
- Gasteiger et al. [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997), 2019.
- Chen et al. [How Graph Convolutions Amplify Popularity Bias for Recommendation?](https://arxiv.org/abs/2305.14886), 2023 preprint.

External code must retain its original attribution and license. No license for redistribution of this team's original work has been selected yet.
