"""Contextual SBM graphs with controlled homophily (experiment: OOD detection vs homophily).

Not in the paper's Table 1. ``csbm_h<10*h>`` (h = 0.1 ... 0.9) share the *same* nodes, labels and
features; only the edges change: each of the ``n * degree / 2`` edges joins a uniformly random node to a
node of its own class with probability ``h`` and to a node of another class otherwise, so the edge
homophily is ~h. Features are Gaussian around class centres (``feature_signal`` sets their separation),
weak enough that message passing matters. Generated in memory with private ``torch.Generator``s.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset

NUM_NODES = 3000
NUM_CLASSES = 6
NUM_FEATURES = 32
DEGREE = 10
FEATURE_SIGNAL = 0.25
HOMOPHILY_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def make_csbm_graph(
    homophily: float,
    num_nodes: int = NUM_NODES,
    num_classes: int = NUM_CLASSES,
    num_features: int = NUM_FEATURES,
    degree: int = DEGREE,
    feature_signal: float = FEATURE_SIGNAL,
    seed: int = 0,
) -> RawGraph:
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(num_nodes) % num_classes
    y = y[torch.randperm(num_nodes, generator=g)]
    centers = feature_signal * torch.randn(num_classes, num_features, generator=g)
    x = centers[y] + torch.randn(num_nodes, num_features, generator=g)

    ge = torch.Generator().manual_seed(seed + 1)  # edges: separate stream, so x/y do not depend on h
    members = [torch.nonzero(y == c).view(-1) for c in range(num_classes)]
    num_edges = num_nodes * degree // 2
    src = torch.randint(0, num_nodes, (num_edges,), generator=ge)
    same = torch.rand(num_edges, generator=ge) < homophily
    offset = torch.randint(1, num_classes, (num_edges,), generator=ge)
    dst_class = torch.where(same, y[src], (y[src] + offset) % num_classes)
    dst = torch.empty_like(src)
    for c in range(num_classes):
        sel = dst_class == c
        pick = torch.randint(0, members[c].numel(), (int(sel.sum()),), generator=ge)
        dst[sel] = members[c][pick]
    keep = src != dst
    edges = torch.stack([src[keep], dst[keep]])
    edge_index = torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)
    return RawGraph(x=x.float(), edge_index=edge_index.long(), y=y.long())


def _load_csbm(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    return make_csbm_graph(spec.extra["homophily"], seed=spec.extra.get("seed", 0))


for _h in HOMOPHILY_LEVELS:
    register_dataset(
        DatasetSpec(
            name=f"csbm_h{round(10 * _h)}",
            paper_name=f"CSBM (h={_h:.1f})",
            load_raw=_load_csbm,
            id_classes=(2, 3, 4, 5),
            ood_classes=(0, 1),
            num_features=NUM_FEATURES,
            homophily="synthetic",
            split="random",
            in_paper=False,
            notes="Contextual SBM generated in memory; homophily study (not in the paper's Table 1).",
            extra={"homophily": _h, "seed": 0},
        )
    )
