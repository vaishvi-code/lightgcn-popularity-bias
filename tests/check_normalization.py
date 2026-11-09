"""Run: python tests/check_normalization.py /path/to/LightGCN-PyTorch/code"""
import sys
from pathlib import Path
import tempfile

upstream = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]  # upstream world parses argv at import
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'), str(upstream)]
import numpy as np
import torch
import world
from dataloader import Loader
from normalized_graph import NormalizedLoader, build_dataset, normalization_spec
from weighted_lightgcn import build_model

world.device = torch.device('cpu')
config = dict(world.config, lightGCN_n_layers=3, latent_dim_rec=4, dropout=0, A_split=False)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root/'train.txt').write_text('0 0 1\n1 1 2\n2 2\n')
    (root/'test.txt').write_text('0 3\n1 0\n2 1\n3 2\n')
    standard = build_dataset(config, root)
    assert type(standard) is Loader
    graph = standard.getSparseGraph().to_dense()
    equivalent = NormalizedLoader(config, tmp, .5, .5)
    torch.testing.assert_close(equivalent.getSparseGraph().to_dense(), graph)
    adjacency = torch.zeros(8,8)
    for u,i in [(0,0),(0,1),(1,1),(1,2),(2,2)]:
        adjacency[u,4+i] = adjacency[4+i,u] = 1
    degree = adjacency.sum(1)
    paths = set()
    for b in (.75,1.0):
        dataset = build_dataset(config, root, .5, b)
        paths.add(dataset.graph_cache_path)
        expected = torch.zeros_like(adjacency)
        for row in range(8):
            for col in range(8):
                if adjacency[row,col]:
                    expected[row,col] = 1/(degree[row]**.5 * degree[col]**b)
        observed = dataset.getSparseGraph().to_dense()
        torch.testing.assert_close(observed, expected)
        assert not torch.allclose(observed, observed.T), 'Asymmetric exponents must keep receiver/neighbor orientation'
        assert not observed[3].any() and not observed[7].any()
        reloaded = build_dataset(config, root, .5, b)
        torch.testing.assert_close(reloaded.getSparseGraph().to_dense(), observed)
        torch.manual_seed(2020); baseline = build_model(config, standard)
        torch.manual_seed(2020); network = build_model(config, dataset)
        for a,c in zip(baseline.parameters(), network.parameters()): torch.testing.assert_close(a,c,rtol=0,atol=0)
        initial = torch.cat([network.embedding_user.weight, network.embedding_item.weight])
        layers = [initial]
        for _ in range(3): layers.append(expected @ layers[-1])
        dense_output = torch.stack(layers,dim=1).mean(dim=1)
        torch.testing.assert_close(torch.cat(network.computer()), dense_output)
        users,pos,neg = torch.tensor([0,1]),torch.tensor([0,2]),torch.tensor([3,0])
        loss,reg = network.bpr_loss(users,pos,neg)
        out_u,out_i = dense_output[:4],dense_output[4:]
        expected_loss = torch.nn.functional.softplus((out_u[users]*(out_i[neg]-out_i[pos])).sum(1)).mean()
        torch.testing.assert_close(loss,expected_loss)
        torch.testing.assert_close(reg,baseline.bpr_loss(users,pos,neg)[1])
        (loss+.0001*reg).backward()
        assert all(torch.isfinite(p.grad).all() for p in network.parameters())
        torch.save(network.state_dict(),root/'checkpoint.pt')
        restored = build_model(config,reloaded)
        restored.load_state_dict(torch.load(root/'checkpoint.pt',weights_only=True))
        torch.testing.assert_close(torch.cat(restored.computer()),torch.cat(network.computer()))
    assert len(paths)==2 and all(p.exists() for p in paths)
    for bad in (-1,float('nan'),float('inf')):
        try: normalization_spec(.5,bad)
        except ValueError: pass
        else: raise AssertionError('Invalid exponent accepted')
print('PASS: default equivalence, orientation, isolated nodes, cache separation/reload, initialization, propagation, BPR, gradients, checkpoint reload')
