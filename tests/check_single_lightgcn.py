"""Check final-layer outputs, matched initialization, gradients, and reloads on a tiny graph."""

import sys
import tempfile
from pathlib import Path

upstream = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"), str(upstream)]

import torch
import world
from dataloader import Loader
from model import LightGCN
from single_lightgcn import SingleLayerOutputLightGCN

world.device = torch.device("cpu")
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path / "train.txt").write_text("0 0 1\n1 1 2\n2 2\n")
    (path / "test.txt").write_text("0 3\n1 0\n2 1\n3 2\n")
    config = dict(world.config, latent_dim_rec=4, dropout=0, A_split=False)
    dataset = Loader(config=config, path=str(path))
    graph = dataset.getSparseGraph().to_dense()
    for depth in range(5):
        config["lightGCN_n_layers"] = depth
        torch.manual_seed(2020)
        standard = LightGCN(config, dataset)
        torch.manual_seed(2020)
        single = SingleLayerOutputLightGCN(config, dataset)
        assert list(standard.state_dict()) == list(single.state_dict())
        for left, right in zip(standard.parameters(), single.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

        initial = torch.cat([single.embedding_user.weight, single.embedding_item.weight])
        layers = [initial]
        for _ in range(depth):
            layers.append(graph @ layers[-1])
        observed = torch.cat(single.computer())
        torch.testing.assert_close(observed, layers[-1])
        torch.testing.assert_close(torch.cat(standard.computer()), torch.stack(layers, dim=1).mean(dim=1))
        if depth:
            assert not torch.allclose(observed, torch.cat(standard.computer()))

        users = torch.tensor([0, 1])
        positive, negative = torch.tensor([0, 2]), torch.tensor([3, 0])
        loss, regularization = single.bpr_loss(users, positive, negative)
        expected_users, expected_items = layers[-1][:dataset.n_users], layers[-1][dataset.n_users:]
        expected_loss = torch.nn.functional.softplus(
            (expected_users[users] * (expected_items[negative] - expected_items[positive])).sum(dim=1)
        ).mean()
        torch.testing.assert_close(loss, expected_loss)
        parameters = list(single.parameters())
        actual_gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected_loss, parameters)
        for left, right in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(regularization, standard.bpr_loss(users, positive, negative)[1])

        checkpoint = path / "single.pt"
        torch.save(single.state_dict(), checkpoint)
        reloaded = SingleLayerOutputLightGCN(config, dataset)
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(torch.cat(reloaded.computer()), observed)

print("PASS: depths 0–4, identical initialization and parameter sets, dense-reference outputs and gradients, regularization, and checkpoint reload.")
