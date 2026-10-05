"""Trainable methods: VanillaGNN and the credal family (CGNN and its ablations)."""

from __future__ import annotations

from typing import Any

from cgnn.methods.spec import MethodSpec, RunContext, register_method

CREDAL_SCORES = ("EU", "AU", "TU")
# Architecture switches of CredalGNN that may be overridden from the run config
# (e.g. `cgnn run -m credal_LJ_dual_head_detached --set credal_mid_activation=identity`).
CREDAL_TUNABLE = (
    "lambda_cls",
    "joint_include_input",
    "joint_act_on_last",
    "credal_mid_activation",
    "credal_half_activation",
)


def _backbone_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "gnn_type": cfg["gnn_type"],
        "in_channels": cfg["in_channels"],
        "out_channels": cfg["out_channels"],
        "hidden_channels": cfg["hidden_channels"],
        "num_layers": cfg["num_layers"],
        "lr": cfg["lr"],
    }


def build_vanilla(ctx: RunContext):
    from cgnn.models.vanilla import VanillaGNN

    return VanillaGNN(**_backbone_kwargs(ctx.cfg), weight_decay=ctx.cfg["weight_decay"])


def credal_builder(*, latent: str, heads: str, detach_credal: bool | None = None, **fixed):
    """Builder for one CredalGNN configuration (see the table in cgnn/models/credal.py)."""

    def build(ctx: RunContext):
        from cgnn.models.credal import CredalGNN

        cfg = ctx.cfg
        kwargs = {k: cfg[k] for k in CREDAL_TUNABLE if k in cfg}
        kwargs.update(fixed)
        return CredalGNN(
            **_backbone_kwargs(cfg),
            weight_decay=cfg.get("weight_decay", 0.0),
            delta=cfg["delta"],
            latent=latent,
            heads=heads,
            detach_credal=detach_credal,
            **kwargs,
        )

    return build


register_method(
    MethodSpec(
        name="vanilla",
        kind="trainable",
        build=build_vanilla,
        description="GNN classifier, MSP score. Backbone for all post-hoc methods.",
    )
)

_CREDAL_METHODS = [
    # name, paper name, builder kwargs, description
    (
        "credal_LJ_dual_head_detached",
        "CGNN",
        dict(latent="joint", heads="dual", detach_credal=True),
        "Proposed CGNN: joint latent, classifier head + detached credal head (Eq. 7, 11).",
    ),
    (
        "credal",
        "CGNN last layer",
        dict(latent="last", heads="credal"),
        "Single credal head on the last-layer embedding, end-to-end credal loss.",
    ),
    (
        "credal_LJ",
        "CGNN only credal",
        dict(latent="joint", heads="credal"),
        "Single credal head on the joint latent, end-to-end credal loss.",
    ),
]

for _name, _paper, _kw, _desc in _CREDAL_METHODS:
    register_method(
        MethodSpec(
            name=_name,
            kind="trainable",
            build=credal_builder(**_kw),
            paper_name=_paper,
            description=_desc,
            score_keys=CREDAL_SCORES,
        )
    )
