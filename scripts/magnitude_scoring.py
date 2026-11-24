"""Fixed inference-only item-magnitude intervention; no model parameters are modified."""
import torch


def equal_length_items(items, epsilon=1e-12):
    if items.ndim != 2 or not torch.isfinite(items).all():
        raise ValueError("Finite item-vector matrix required")
    if not 0 < epsilon < 1:
        raise ValueError("A small positive epsilon is required")
    norms = torch.linalg.vector_norm(items, dim=1, keepdim=True)
    # Exact zeros stay zero. Vectors smaller than epsilon are scaled by epsilon.
    return items / norms.clamp_min(epsilon), norms.squeeze(1)


def score(users, items, mode):
    if mode not in ("legacy_sigmoid", "raw_dot", "unit_item"):
        raise ValueError("Unknown fixed scoring mode")
    values = users @ items.T
    return torch.sigmoid(values) if mode == "legacy_sigmoid" else values
