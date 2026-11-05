import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--early-stopping-patience", type=int, default=0)
parser.add_argument("--resume", type=Path)
stopping_args, upstream_args = parser.parse_known_args()
if stopping_args.early_stopping_patience < 0:
    parser.error("--early-stopping-patience must be nonnegative; 0 disables stopping")
# The upstream world module parses sys.argv during import.
sys.argv = [sys.argv[0], *upstream_args]

base = Path(__file__).resolve().parent.parent
upstream = base / "LightGCN-PyTorch"
prepared = base / "prepared/gowalla"

fit_hash = hashlib.sha256((prepared / "train.txt").read_bytes()).hexdigest()
val_hash = hashlib.sha256((prepared / "val.txt").read_bytes()).hexdigest()

# Different splits get different graph-cache directories.
loader_data = base / "prepared" / (
    "gowalla-validation-" + fit_hash[:12] + "-" + val_hash[:12]
)
loader_data.mkdir(parents=True, exist_ok=True)
shutil.copyfile(prepared / "train.txt", loader_data / "train.txt")
shutil.copyfile(prepared / "val.txt", loader_data / "test.txt")

sys.path.insert(0, str(upstream / "code"))
import world

if world.TRAIN_epochs < 1:
    raise ValueError("--epochs must be at least 1")

# This wrapper records CSV history instead of TensorBoard events.
world.tensorboard = 0
torch.set_num_threads(4)

import utils
import model
import Procedure
from dataloader import Loader

assert torch.cuda.is_available(), "Run this inside a GPU allocation"
utils.set_seed(world.seed)

dataset = Loader(config=world.config, path=str(loader_data))
network = model.LightGCN(world.config, dataset).to(world.device)
loss = utils.BPRLoss(network, world.config)

start_epoch = 0
best_recall = -1.0
best_epoch = 0
epochs_without_improvement = 0
parent_history = []
resume_path = None
resume_rng = "fresh_seed"

if stopping_args.resume:
    resume_path = stopping_args.resume.resolve()
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    saved = checkpoint["metadata"]
    if (saved["fitting_file_sha256"], saved["validation_file_sha256"]) != (fit_hash, val_hash):
        raise ValueError("Resume checkpoint uses different fitting or validation data")
    if saved["config"] != world.config or saved["training_seed"] != world.seed:
        raise ValueError("Resume settings must match the checkpoint configuration and seed")
    start_epoch = int(checkpoint["epoch"])
    if world.TRAIN_epochs <= start_epoch:
        raise ValueError("--epochs must exceed the saved epoch; it is the total training limit")
    with (resume_path.parent / "history.csv").open(newline="") as file:
        parent_history = list(csv.DictReader(file))
    if [int(row["epoch"]) for row in parent_history] != list(range(1, start_epoch + 1)):
        raise ValueError("Resume requires the last checkpoint and complete matching history")
    for row in parent_history:
        recall = float(row["validation_recall_at_20"])
        if recall > best_recall:
            best_recall = recall
            best_epoch = int(row["epoch"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    if float(parent_history[-1]["validation_recall_at_20"]) != checkpoint["validation_recall_at_20"]:
        raise ValueError("Last checkpoint recall differs from its recorded history")
    previous_best = torch.load(resume_path.parent / "best.pt", map_location="cpu", weights_only=False)
    if previous_best["epoch"] != best_epoch or previous_best["validation_recall_at_20"] != best_recall:
        raise ValueError("Best checkpoint differs from the best recorded validation result")
    del previous_best
    patience = stopping_args.early_stopping_patience
    if patience > 0 and epochs_without_improvement >= patience:
        raise ValueError("Saved history already meets the requested early-stopping rule")
    network.load_state_dict(checkpoint["model_state_dict"])
    loss.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    if "rng_state" in checkpoint:
        state = checkpoint["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])
        resume_rng = "restored"
    else:
        random.seed(world.seed)
        utils.set_seed(world.seed)
        resume_rng = "reseeded_legacy_checkpoint"
        print("Legacy checkpoint has no random-generator states; sampling restarts from the saved seed.", flush=True)
    del checkpoint

stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
run = base / "results/gowalla" / (
    f"lightgcn-k{world.config['lightGCN_n_layers']}"
    f"-seed{world.seed}-{stamp}"
)
run.mkdir(parents=True, exist_ok=False)

metadata = {
    "model": "LightGCN",
    "evaluation_split": "validation",
    "upstream_test_filename_contains": "validation interactions",
    "fitting_file_sha256": fit_hash,
    "validation_file_sha256": val_hash,
    "training_seed": world.seed,
    "config": world.config,
    "requested_epochs": world.TRAIN_epochs,
    "resumed_from": str(resume_path) if resume_path else None,
    "starting_epoch": start_epoch,
    "resume_rng": resume_rng,
    "early_stopping": {
        "metric": "validation_recall_at_20",
        "mode": "max",
        "patience": stopping_args.early_stopping_patience,
        "enabled": stopping_args.early_stopping_patience > 0,
        "improvement_rule": "strictly greater than the best observed recall",
    },
    "torch_version": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
(run / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

print("Evaluation split: VALIDATION, not final test", flush=True)
print("Run directory:", run, flush=True)
if resume_path:
    shutil.copyfile(resume_path.parent / "best.pt", run / "best.pt")
    shutil.copyfile(resume_path, run / "last.pt")
    print(f"Resuming after epoch {start_epoch}; next epoch={start_epoch + 1}; best epoch={best_epoch}; patience counter={epochs_without_improvement}.", flush=True)

stop_reason = "max_epochs"

with (run / "history.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", "training_seconds", "validation_seconds",
        "validation_recall_at_20", "validation_ndcg_at_20",
        "training_log"
    ])
    for row in parent_history:
        writer.writerow([row[key] for key in (
            "epoch", "training_seconds", "validation_seconds",
            "validation_recall_at_20", "validation_ndcg_at_20", "training_log"
        )])
    file.flush()

    for epoch in range(start_epoch, world.TRAIN_epochs):
        start = time.perf_counter()
        training_log = Procedure.BPR_train_original(
            dataset, network, loss, epoch
        )
        training_seconds = time.perf_counter() - start

        print(
            f"Epoch {epoch + 1}: {training_log}; "
            f"training time={training_seconds:.1f}s",
            flush=True
        )

        start = time.perf_counter()
        metrics = Procedure.Test(dataset, network, epoch, multicore=0)
        validation_seconds = time.perf_counter() - start

        recall = float(metrics["recall"][0])
        ndcg = float(metrics["ndcg"][0])
        if not math.isfinite(recall) or not math.isfinite(ndcg):
            raise ValueError(f"Non-finite validation metrics at epoch {epoch + 1}")

        writer.writerow([
            epoch + 1, training_seconds, validation_seconds,
            recall, ndcg, training_log
        ])
        file.flush()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": loss.opt.state_dict(),
            "metadata": metadata,
            "validation_recall_at_20": recall,
            "validation_ndcg_at_20": ndcg,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }
        torch.save(checkpoint, run / "last.pt")

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(checkpoint, run / "best.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"VALIDATION epoch={epoch + 1} "
            f"Recall@20={recall:.6f} NDCG@20={ndcg:.6f}",
            flush=True
        )

        patience = stopping_args.early_stopping_patience
        if patience > 0 and epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            print(
                f"Early stopping after {patience} epochs without improvement. "
                f"Best validation Recall@20={best_recall:.6f} at epoch {best_epoch}.",
                flush=True,
            )
            break

summary = {
    "completed_epochs": epoch + 1,
    "epochs_trained_this_run": epoch + 1 - start_epoch,
    "requested_epochs": world.TRAIN_epochs,
    "stop_reason": stop_reason,
    "best_epoch": best_epoch,
    "best_validation_recall_at_20": best_recall,
    "early_stopping_patience": stopping_args.early_stopping_patience,
}
(run / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Training finished. Results:", run, flush=True)
