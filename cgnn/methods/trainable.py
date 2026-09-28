"""Trainable methods: VanillaGNN, the credal family, GraphESN."""

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


def credal_builder(
    *, latent: str, heads: str, detach_credal: bool | None = None, constrained: bool = False, **fixed
):
    """Builder for one CredalGNN configuration (see the table in cgnn/models/credal.py)."""

    def build(ctx: RunContext):
        from cgnn.models.credal import CredalGNN

        cfg = ctx.cfg
        kwargs = {k: cfg[k] for k in CREDAL_TUNABLE if k in cfg}
        kwargs.update(fixed)
        if constrained:
            kwargs["lambda_cons"] = cfg.get("lambda_cons", 0.1)
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


def build_graph_esn(ctx: RunContext):
    from cgnn.models.graph_esn import GraphEchoStateNetwork

    cfg = ctx.cfg
    return GraphEchoStateNetwork(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        lr=cfg.get("lr", 0.0),
        weight_decay=cfg.get("weight_decay", 0.0),
        ood_in_val=cfg.get("ood_in_val", True),
        spectral_radius=cfg.get("spectral_radius", 0.9),
        input_scaling=cfg.get("input_scaling", 1.0),
        leakage=cfg.get("leakage", 1.0),
        num_reservoirs=cfg.get("num_reservoirs", 1),
        readout_regularization=cfg.get("readout_regularization", 1e-3),
        bias=cfg.get("bias", False),
        pooling=cfg.get("pooling", None),
        fully=cfg.get("fully", False),
        max_iterations=cfg.get("max_iterations", None),
        epsilon=cfg.get("epsilon", 1e-6),
    )


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
    (
        "credal_LJ_dual_head",
        None,
        dict(latent="joint", heads="dual", detach_credal=False),
        "Dual head, credal head NOT detached.",
    ),
    (
        "credal_LJ_dual_head_constrained",
        None,
        dict(latent="joint", heads="dual", detach_credal=False, constrained=True),
        "Dual head + interval-consistency penalty.",
    ),
    (
        "credal_LJ_dual_head_constrained_detached",
        None,
        dict(latent="joint", heads="dual", detach_credal=True, constrained=True),
        "Detached dual head + interval-consistency penalty.",
    ),
    (
        "credal_last_dual_head_detached",
        None,
        dict(latent="last", heads="dual", detach_credal=True),
        "NEW: clean 'last layer' ablation of CGNN (only the latent changes).",
    ),
    (
        "credal_LJ0_dual_head_detached",
        None,
        dict(latent="joint", heads="dual", detach_credal=True, joint_include_input=True),
        "NEW: CGNN with z^0 in the joint latent, exactly as in paper Eq. 14.",
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

register_method(
    MethodSpec(
        name="graph_esn",
        kind="trainable",
        build=build_graph_esn,
        description="Graph Echo State Network ensemble (exploratory).",
        requires=("graphesn",),
        score_keys=tuple(f"{k}_{f}" for f in ("credal", "classic") for k in CREDAL_SCORES),
    )
)
