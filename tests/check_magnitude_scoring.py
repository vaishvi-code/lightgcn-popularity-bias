"""Hand-built vector checks for the magnitude intervention and score decomposition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from magnitude_scoring import equal_length_items, score

users = torch.tensor([[1., 0.], [0., 2.], [0., 0.]])
items = torch.tensor([[2., 2.], [1., 0.], [0., 0.]])
original = items.clone()
unit, norms = equal_length_items(items)
torch.testing.assert_close(items, original, atol=0, rtol=0)
torch.testing.assert_close(torch.linalg.vector_norm(unit, dim=1), torch.tensor([1., 1., 0.]))
torch.testing.assert_close(unit * norms[:, None], items)
raw, intervention = score(users, items, "raw_dot"), score(users, unit, "unit_item")
assert raw[0, 0] > raw[0, 1] and intervention[0, 0] < intervention[0, 1]
torch.testing.assert_close(score(users, items, "legacy_sigmoid"), torch.sigmoid(raw))
rescaled = items * torch.tensor([[10.], [.5], [1.]])
unit_rescaled, _ = equal_length_items(rescaled)
torch.testing.assert_close(unit_rescaled, unit)
unit_users = users / torch.linalg.vector_norm(users, dim=1, keepdim=True).clamp_min(1e-12)
cosine = unit_users @ unit.T
torch.testing.assert_close(raw, torch.linalg.vector_norm(users, dim=1)[:, None] * norms[None, :] * cosine)
torch.testing.assert_close(intervention, torch.linalg.vector_norm(users, dim=1)[:, None] * cosine)
assert torch.isfinite(intervention).all()
tiny, _ = equal_length_items(torch.tensor([[1e-14, 0.]]))
torch.testing.assert_close(tiny, torch.tensor([[.01, 0.]]))
for mode in ["unknown", "normalized_training"]:
    try: score(users, items, mode)
    except ValueError: pass
    else: raise AssertionError("Unknown scoring mode accepted")
print("PASS: score decomposition, fixed directions, item-scale invariance, zero/tiny vectors and a hand-checked rank reversal")
