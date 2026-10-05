"""Method specification and registry.

A *method* is an experiment recipe: which LightningModule to build from a run
config, and how the generic runner (``cgnn.runner``) drives it.

- ``kind="trainable"``: seed -> build -> fit(train, val) -> test.
- ``kind="posthoc"``: select VanillaGNN backbone(s) -> build -> ``prepare`` on
  the training graph -> validate/test, or fit on ``fit_on`` loader -> test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cgnn.registry import Registry

if TYPE_CHECKING:
    import lightning as L

    from cgnn.checkpoints import CheckpointRecord
    from cgnn.data.spec import DatasetSpec
    from cgnn.paths import Paths


@dataclass
class RunContext:
    method: MethodSpec
    dataset: DatasetSpec
    cfg: dict[str, Any]
    paths: Paths
    run_id: str
    backbones: list[CheckpointRecord] = field(default_factory=list)

    @property
    def backbone_path(self) -> str:
        return str(self.backbones[0].path)


@dataclass(frozen=True)
class MethodSpec:
    name: str  # CLI / W&B key
    kind: Literal["trainable", "posthoc"]
    build: Callable[[RunContext], L.LightningModule]
    paper_name: str | None = None  # row label in the paper tables, None if not in the paper
    description: str = ""
    # --- post-hoc only ---
    num_backbones: Callable[[Mapping[str, Any]], int] | None = None  # default: 1 backbone at `backbone_rank`
    fit_on: Literal["train", "val"] | None = None  # post-hoc methods with a training stage
    seed_is_backbone_rank: bool = False
    # --- metadata ---
    requires: tuple[str, ...] = ()  # optional python modules (reported by `cgnn doctor`)
    score_keys: tuple[str, ...] = ("",)  # uncertainty scores logged as `{split}_auroc{_key}`

    @property
    def in_paper(self) -> bool:
        return self.paper_name is not None


METHODS: Registry[MethodSpec] = Registry("method")


def register_method(spec: MethodSpec) -> MethodSpec:
    return METHODS.register(spec.name, spec)
