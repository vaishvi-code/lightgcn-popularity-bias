"""Verify that recovery discards only unsaved epochs and leaves original artifacts intact."""

import csv
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_interrupted_resume import checkpoint_epoch, prepare

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "original"
    source.mkdir()
    for name, epoch in [("last.pt", 3), ("best.pt", 2)]:
        with zipfile.ZipFile(source / name, "w") as archive:
            archive.writestr("archive/data.pkl", pickle.dumps({"epoch": epoch}, protocol=2))
        assert checkpoint_epoch(source / name) == epoch
    (source / "config.json").write_text("{}\n")
    with (source / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "validation_recall_at_20"])
        writer.writerows([(1, .1), (2, .2), (3, .19), (4, .3)])
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    output = root / "recovery"
    receipt = prepare(source, output)
    assert receipt["discarded_uncheckpointed_epochs"] == [4]
    assert receipt["last_checkpoint_epoch"] == 3
    assert receipt["best_checkpoint_epoch"] == 2
    with (output / "history.csv").open() as stream:
        assert [int(row["epoch"]) for row in csv.DictReader(stream)] == [1, 2, 3]
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}
    for name in ["last.pt", "best.pt", "config.json"]:
        assert (output / name).read_bytes() == original[name]
    try:
        prepare(source, source / "nested")
    except ValueError:
        pass
    else:
        raise AssertionError("Nested recovery directory accepted")
    with zipfile.ZipFile(source / "best.pt", "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"epoch": 1}, protocol=2))
    try:
        prepare(source, root / "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatching best checkpoint accepted")
print("PASS: durable-history recovery, byte-identical checkpoint copies, source preservation, and mismatch rejection.")
