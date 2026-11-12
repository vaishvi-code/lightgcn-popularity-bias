"""Final-layer-only LightGCN for the supplemental validation ablation."""

import torch
from model import LightGCN


class SingleLayerOutputLightGCN(LightGCN):
    def __init__(self, config, dataset):
        if config["dropout"] or config["A_split"]:
            raise ValueError("The single-output ablation requires dropout=0 and A_split=False")
        if config["lightGCN_n_layers"] < 0:
            raise ValueError("Layer count must be nonnegative")
        super().__init__(config, dataset)

    def computer(self):
        embeddings = torch.cat([
            self.embedding_user.weight, self.embedding_item.weight
        ])
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(self.Graph, embeddings)
        return torch.split(embeddings, [self.num_users, self.num_items])
