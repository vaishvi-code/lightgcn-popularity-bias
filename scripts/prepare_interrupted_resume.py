"""Copy an interrupted run into fresh resume inputs aligned to its durable checkpoints."""

import argparse
import csv
import hashlib
import json
import pickletools
import shutil
import zipfile
from pathlib import Path


def checkpoint_epoch(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Checkpoint ZIP checksum failed")
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError("Expected one checkpoint metadata entry")
        operations = list(pickletools.genops(archive.read(names[0])))
    position = next(index for index, (_, value, _) in enumerate(operations) if value == "epoch")
    for operation, value, _ in operations[position + 1:]:
        if operation.name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:
            continue
        if operation.name not in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT"} or not isinstance(value, int):
            raise ValueError("Invalid checkpoint epoch")
        return value
    raise ValueError("Missing checkpoint epoch value")


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Resume inputs must use a separate directory")
    last_epoch = checkpoint_epoch(source / "last.pt")
    best_epoch = checkpoint_epoch(source / "best.pt")
    with (source / "history.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        original = list(reader)
    history = [row for row in original if int(row["epoch"]) <= last_epoch]
    if [int(row["epoch"]) for row in history] != list(range(1, last_epoch + 1)):
        raise ValueError("Saved checkpoint has incomplete history")
    selected = max(history, key=lambda row: float(row["validation_recall_at_20"]))
    if int(selected["epoch"]) != best_epoch:
        raise ValueError("Best checkpoint does not match the durable history")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ["last.pt", "best.pt", "config.json"]:
        shutil.copyfile(source / name, output / name)
        digest = hashlib.sha256()
        with (output / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    receipt = {
        "source_run": str(source), "resume_inputs": str(output),
        "last_checkpoint_epoch": last_epoch, "best_checkpoint_epoch": best_epoch,
        "discarded_uncheckpointed_epochs": [int(row["epoch"]) for row in original if int(row["epoch"]) > last_epoch],
        "copied_sha256": hashes, "source_artifacts_modified": False,
    }
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_run, args.output), indent=2))
