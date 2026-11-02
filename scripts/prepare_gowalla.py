import hashlib
import json
import random
from pathlib import Path

base = Path(__file__).resolve().parent.parent
source = base / "LightGCN-PyTorch/data/gowalla"
output = base / "prepared/gowalla"
output.mkdir(parents=True, exist_ok=True)

seed = 2020
rng = random.Random(seed)

original = {}
for line in (source / "train.txt").read_text().splitlines():
    values = list(map(int, line.split()))
    if values:
        original[values[0]] = sorted(set(values[1:]))

fit, validation = {}, {}

for user in sorted(original):
    items = original[user].copy()
    rng.shuffle(items)

    # Hold out approximately 10%, but retain at least one fitting item.
    count = min(len(items) - 1, max(1, len(items) // 10)) if items else 0
    count = max(0, count)

    validation[user] = sorted(items[:count])
    fit[user] = sorted(items[count:])

    assert not set(fit[user]) & set(validation[user])
    assert set(fit[user]) | set(validation[user]) == set(original[user])
    assert not original[user] or fit[user]

def write_split(path, interactions):
    with path.open("w") as file:
        for user in sorted(interactions):
            items = interactions[user]
            if items:
                file.write(f"{user} " + " ".join(map(str, items)) + "\n")

write_split(output / "train.txt", fit)
write_split(output / "val.txt", validation)

fit_items = {item for items in fit.values() for item in items}
original_items = {item for items in original.values() for item in items}
val_count = sum(map(len, validation.values()))

metadata = {
    "split_seed": seed,
    "validation_rule": "floor(10% per user), minimum 1 if user has >=2 items",
    "source_train_sha256": hashlib.sha256(
        (source / "train.txt").read_bytes()
    ).hexdigest(),
    "fit_interactions": sum(map(len, fit.values())),
    "validation_interactions": val_count,
    "users_with_validation": sum(bool(items) for items in validation.values()),
    "original_train_items": len(original_items),
    "items_with_fitting_interactions": len(fit_items),
    "items_without_fitting_interactions": len(original_items - fit_items),
    "validation_interactions_on_unseen_items": sum(
        item not in fit_items
        for items in validation.values()
        for item in items
    ),
}

(output / "split-metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)

print(json.dumps(metadata, indent=2))
print("Saved prepared splits to:", output)
print("Original training and test files were left unchanged.")
