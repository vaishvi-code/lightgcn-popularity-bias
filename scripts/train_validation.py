import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

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

# Disable TensorBoard for this initial check.
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

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
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
    "torch_version": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
(run / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

print("Evaluation split: VALIDATION, not final test", flush=True)
print("Run directory:", run, flush=True)

best_recall = -1.0

with (run / "history.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", "training_seconds", "validation_seconds",
        "validation_recall_at_20", "validation_ndcg_at_20",
        "training_log"
    ])
    file.flush()

    for epoch in range(world.TRAIN_epochs):
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
        }
        torch.save(checkpoint, run / "last.pt")

        if recall > best_recall:
            best_recall = recall
            torch.save(checkpoint, run / "best.pt")

        print(
            f"VALIDATION epoch={epoch + 1} "
            f"Recall@20={recall:.6f} NDCG@20={ndcg:.6f}",
            flush=True
        )

print("Training check finished. Results:", run, flush=True)
