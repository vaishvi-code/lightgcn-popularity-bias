"""Select one illustrative history without looking at outcomes; keep output private."""
import argparse
import csv
import gc
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
from final_metrics import load_interactions, exclusion_sets
from user_groups import assign_group


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, base = args.repository, args.project_root
    plan = json.loads((repo / "provenance/final-test-plan.json").read_text())
    fit_path = base / "prepared/gowalla/train.txt"
    val_path = base / "prepared/gowalla/val.txt"
    assert digest(fit_path) == plan["fitting_file_sha256"]
    assert digest(val_path) == plan["validation_file_sha256"]
    pop_path = repo / "results/gowalla/item-popularity.csv"
    assert digest(pop_path) == plan["popularity_file_sha256"]
    with pop_path.open() as stream:
        popularity = {int(row["item_id"]): row for row in csv.DictReader(stream)}
    tail = {item for item, row in popularity.items() if row["group"] == "tail"}
    fitting = load_interactions(fit_path)
    candidates = [(user, len(items), len(items & tail) / len(items)) for user, items in fitting.items()
                  if items and assign_group(len(items & tail), len(items)) == "tail_oriented"]
    median_length = statistics.median(row[1] for row in candidates)
    median_tail = statistics.median(row[2] for row in candidates)
    user, history_count, history_tail = min(candidates, key=lambda row: (abs(row[1] - median_length), abs(row[2] - median_tail), row[0]))
    args.output.mkdir(parents=True, exist_ok=False)
    selection = dict(selection_rule="Among fitting-history tail-oriented users, minimize distance to group median fitting-history length, then distance to group median tail fraction, then user ID. No outcomes used.",
                     user_id=user, fitting_interactions=history_count, fitting_tail_fraction=history_tail,
                     candidate_count=len(candidates), group_median_length=median_length, group_median_tail_fraction=median_tail,
                     fitting_file_sha256=plan["fitting_file_sha256"], study_phase="post_hoc_illustrative_case", publish_per_user_data=False)
    (args.output / "selection-private.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Only now read held-out labels. Do not try another user based on the results.
    frozen = repo / "results/gowalla/final-test-job337725"
    test_path = base / "LightGCN-PyTorch/data/gowalla/test.txt"
    reference_path = next((frozen / "test").glob("*/metrics.json"))
    assert digest(test_path) == json.loads(reference_path.read_text())["test_file_sha256"]
    relevant = load_interactions(test_path)
    assert relevant[user], "Selected user must not be replaced if ineligible"
    validation = load_interactions(val_path)
    seen = exclusion_sets(fitting, validation, "test")
    users = sorted(u for u, items in relevant.items() if items)
    index = users.index(user)
    batch = users[index // 256 * 256:index // 256 * 256 + 256]
    selected_row = batch.index(user)
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(base / "LightGCN-PyTorch/code"))
    import world
    from dataloader import Loader
    from weighted_lightgcn import build_model
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    world.tensorboard = 0
    world.device = torch.device("cuda")
    entries = [entry for entry in plan["evaluations"] if entry["comparison_group"] == "main_three_seed" and entry["training_seed"] == 2020 and entry["model"] in ["LightGCN K=2", "LightGCN K=4", "APPNP beta=0.5"]]
    assert len(entries) == 3
    loader = base / "prepared" / ("gowalla-validation-" + plan["fitting_file_sha256"][:12] + "-" + plan["validation_file_sha256"][:12])
    assert digest(loader / "train.txt") == plan["fitting_file_sha256"]
    dataset = Loader(config=entries[0]["config"], path=str(loader))
    output = []
    aliases = {}
    for entry in entries:
        run = entry["run_id"]
        checkpoint_path = next(root / run / "best.pt" for root in args.checkpoint_root if (root / run / "best.pt").is_file())
        expected = json.loads((frozen / "test" / run / "metrics.json").read_text())
        checkpoint_hash = digest(checkpoint_path)
        assert checkpoint_hash == expected["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == entry["checkpoint_epoch"]
        assert checkpoint["metadata"]["config"] == entry["config"]
        world.config.update(entry["config"])
        network = build_model(world.config, dataset, entry["appnp_beta"]).to(world.device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        with torch.no_grad():
            scores = network.getUsersRating(torch.tensor(batch, device=world.device))
            assert torch.isfinite(scores).all()
            for row, identity in enumerate(batch):
                scores[row, list(seen[identity])] = -torch.inf
            recs = torch.topk(scores, k=20, dim=1).indices[selected_row].cpu().tolist()
        assert len(set(recs)) == 20 and not set(recs) & seen[user]
        rankings = []
        for rank, item in enumerate(recs, 1):
            aliases.setdefault(item, "Place %02d" % (len(aliases) + 1))
            rankings.append(dict(rank=rank, item_id=item, display_name=aliases[item], tail=item in tail,
                                 fitting_degree=int(popularity[item]["fitting_degree"]), test_hit=item in relevant[user]))
        relevant_tail = relevant[user] & tail
        output.append(dict(model=entry["model"], run_id=run, checkpoint_sha256=checkpoint_hash, rankings=rankings,
                           hits=sum(row["test_hit"] for row in rankings), tail_slots=sum(row["tail"] for row in rankings),
                           tail_hits=sum(row["tail"] and row["test_hit"] for row in rankings),
                           recall_at_20=len(set(recs) & relevant[user]) / len(relevant[user]),
                           tail_recall_at_20=len(set(recs) & relevant_tail) / len(relevant_tail) if relevant_tail else None))
        del network, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "example-private.json").write_text(json.dumps(dict(selection=selection, test_positives=len(relevant[user]), test_tail_positives=len(relevant[user] & tail), models=output), indent=2) + "\n")
    print("Example selected without outcomes; private artifacts written. No user ID printed.")


if __name__ == "__main__":
    main()
