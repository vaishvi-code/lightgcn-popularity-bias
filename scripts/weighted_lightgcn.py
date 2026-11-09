"""APPNP-style layer averaging for the pinned LightGCN implementation."""
import math

import torch
from model import LightGCN


class APPNPLightGCN(LightGCN):
    def __init__(self, config, dataset, beta):
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("APPNP beta must be finite and in (0, 1]")
        if config["lightGCN_n_layers"] < 0:
            raise ValueError("The layer count must be nonnegative")
        if config["dropout"] or config["A_split"]:
            raise ValueError("This APPNP experiment requires dropout=0 and A_split=False")
        super().__init__(config, dataset)
        weights = [beta * (1 - beta) ** k for k in range(self.n_layers)]
        weights.append((1 - beta) ** self.n_layers)
        self.register_buffer("layer_weights", torch.tensor(weights, dtype=torch.float32))

    def computer(self):
        all_embeddings = torch.cat([
            self.embedding_user.weight, self.embedding_item.weight
        ])
        layers = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.Graph, all_embeddings)
            layers.append(all_embeddings)
        stacked = torch.stack(layers, dim=1)
        combined = (stacked * self.layer_weights.view(1, -1, 1)).sum(dim=1)
        return torch.split(combined, [self.num_users, self.num_items])


def build_model(config, dataset, beta=None):
    if beta is None:
        return LightGCN(config, dataset)
    return APPNPLightGCN(config, dataset, beta)
