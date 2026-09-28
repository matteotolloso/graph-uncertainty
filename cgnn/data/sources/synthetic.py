"""Tiny deterministic stochastic-block-model graph for tests and smoke runs.

Not a benchmark: it exists so that every method can be run end-to-end in a few
seconds on CPU (``cgnn run -m <method> -d synthetic --wandb disabled``).
Generation uses its own ``torch.Generator`` and never touches the global RNG.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset

NUM_NODES = 240
NUM_CLASSES = 5
NUM_FEATURES = 16


def make_sbm_graph(
    num_nodes: int = NUM_NODES,
    num_classes: int = NUM_CLASSES,
    num_features: int = NUM_FEATURES,
    p_in: float = 0.08,
    p_out: float = 0.02,
    seed: int = 0,
) -> RawGraph:
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(num_nodes) % num_classes
    y = y[torch.randperm(num_nodes, generator=g)]

    centers = torch.randn(num_classes, num_features, generator=g)
    x = centers[y] + 0.8 * torch.randn(num_nodes, num_features, generator=g)

    same = y.unsqueeze(0) == y.unsqueeze(1)
    prob = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    upper = torch.triu(torch.rand(num_nodes, num_nodes, generator=g) < prob, diagonal=1)
    src, dst = upper.nonzero(as_tuple=True)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    return RawGraph(x=x.float(), edge_index=edge_index.long(), y=y.long())


def _load_synthetic(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    return make_sbm_graph(seed=spec.extra.get("seed", 0))


register_dataset(
    DatasetSpec(
        name="synthetic",
        paper_name="Synthetic-SBM",
        load_raw=_load_synthetic,
        id_classes=(2, 3, 4),
        ood_classes=(0, 1),
        num_features=NUM_FEATURES,
        homophily="synthetic",
        split="random",
        in_paper=False,
        notes="Generated in memory; for tests and smoke runs only.",
        extra={"seed": 0},
    )
)
