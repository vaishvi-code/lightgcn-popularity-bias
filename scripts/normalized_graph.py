"""Degree normalization ablation; rows receive messages from column neighbors."""
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import scipy.sparse as sp

import world
from dataloader import Loader


def normalization_spec(a=0.5, b=0.5):
    a, b = float(a), float(b)
    if not all(math.isfinite(x) and x >= 0 for x in (a, b)):
        raise ValueError("Normalization exponents must be finite and nonnegative")
    return {"a": a, "b": b, "formula": "D^-a A D^-b", "degree_source": "fitting_only"}


class NormalizedLoader(Loader):
    def __init__(self, config, path, a, b):
        self.normalization = normalization_spec(a, b)
        if config["A_split"] or config["dropout"]:
            raise ValueError("Normalization experiments require A_split=False and dropout=0")
        super().__init__(config=config, path=path)

    @property
    def graph_cache_path(self):
        # The containing directory is keyed by fitting and validation hashes.
        # Exact float encodings distinguish exponents without rounded-key collisions.
        a, b = self.normalization['a'], self.normalization['b']
        return Path(self.path) / f"degree-norm-v1-a{a.hex()}-b{b.hex()}.npz"

    def getSparseGraph(self):
        if self.Graph is not None:
            return self.Graph
        cache = self.graph_cache_path
        if cache.exists():
            norm = sp.load_npz(cache)
        else:
            interactions = self.UserItemNet.astype(np.float32)
            adjacency = sp.bmat([[None, interactions], [interactions.T, None]], format="csr")
            degrees = np.asarray(adjacency.sum(axis=1)).ravel()
            def scale(exponent):
                result = np.zeros_like(degrees)
                present = degrees > 0
                result[present] = degrees[present] ** -exponent
                return sp.diags(result)
            norm = (scale(self.normalization['a']) @ adjacency @ scale(self.normalization['b'])).tocsr()
            temporary = cache.with_name(f".{cache.stem}.{uuid4().hex}.npz")
            sp.save_npz(temporary, norm)
            temporary.replace(cache)
        expected = self.n_users + self.m_items
        if norm.shape != (expected, expected) or not np.isfinite(norm.data).all():
            raise ValueError("Invalid normalized graph cache")
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm).coalesce().to(world.device)
        print(f"Normalization {self.normalization}; graph cache={cache}", flush=True)
        return self.Graph


def build_dataset(config, path, a=0.5, b=0.5):
    spec = normalization_spec(a, b)
    if spec['a'] == spec['b'] == 0.5:
        return Loader(config=config, path=str(path))
    return NormalizedLoader(config=config, path=str(path), a=a, b=b)
