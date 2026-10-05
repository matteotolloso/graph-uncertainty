"""Post-hoc methods on pre-trained VanillaGNN backbones."""

from __future__ import annotations

from cgnn.methods.spec import MethodSpec, RunContext, register_method

ENSEMBLE_SCORES = tuple(f"{k}_{f}" for f in ("credal", "classic") for k in ("EU", "AU", "TU"))


def _detector(class_name: str, *param_names: str, **defaults):
    """Builder for ``cgnn.models.detectors.<class_name>(backbone, **params)``."""

    def build(ctx: RunContext):
        from cgnn.models import detectors

        cls = getattr(detectors, class_name)
        kwargs = {p: ctx.cfg.get(p, defaults.get(p)) for p in param_names}
        return cls(backbone_ckpt_path=ctx.backbone_path, **{k: v for k, v in kwargs.items() if v is not None})

    return build


def build_ensemble(ctx: RunContext):
    from cgnn.models.detectors import CredalEnsemble

    return CredalEnsemble(checkpoint_paths=[str(r.path) for r in ctx.backbones])


def build_frozen(ctx: RunContext):
    from cgnn.models.frozen import CredalFrozenJoint

    cfg = ctx.cfg
    return CredalFrozenJoint(
        checkpoint_path=ctx.backbone_path,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
        delta=cfg.get("delta", 0.5),
        ood_in_val=True,
    )


def build_cagcn(ctx: RunContext):
    from cgnn.models.cagcn import CaGCNModule

    cfg = ctx.cfg
    return CaGCNModule(
        checkpoint_path=ctx.backbone_path,
        calib_hidden=cfg.get("calib_hidden", 16),
        calib_layers=cfg.get("calib_layers", 2),
        lr=cfg.get("lr", 1e-2),
        weight_decay=cfg.get("weight_decay", 5e-3),
        softplus_eps=cfg.get("softplus_eps", 1.1),
        ood_in_val=True,
    )


_SPECS = [
    MethodSpec(
        "energy",
        "posthoc",
        _detector("EnergyDetector", "temperature"),
        paper_name="Energy",
        description="Energy score of the backbone logits.",
        seed_is_backbone_rank=True,
    ),
    MethodSpec(
        "odin",
        "posthoc",
        _detector("ODINDetector", "temperature", "noise_magnitude"),
        paper_name="ODIN",
        description="ODIN: temperature scaling + input perturbation.",
    ),
    MethodSpec(
        "mahalanobis",
        "posthoc",
        _detector("MahalanobisDetector", "noise_magnitude"),
        paper_name="Mahalanobis",
        description="Min class-conditional Mahalanobis distance on logits.",
    ),
    MethodSpec(
        "knn",
        "posthoc",
        _detector("KNNDetector", "k"),
        paper_name="KNN",
        description="Deep kNN distance on the penultimate embedding.",
        requires=("faiss",),
    ),
    MethodSpec(
        "knn_LJ",
        "posthoc",
        _detector("KNNJointDetector", "k"),
        paper_name="JLDE",
        description="kNN density on the joint latent (JLDE ablation).",
        requires=("faiss",),
    ),
    MethodSpec(
        "gnnsafe",
        "posthoc",
        _detector("GNNSafeDetector", "K", "alpha"),
        paper_name="GNNSafe",
        description="Energy belief propagation (no regularisation).",
        requires=("torch_sparse",),
    ),
    MethodSpec(
        "gebm",
        "posthoc",
        _detector("GEBMDetector"),
        paper_name="GEBM",
        description="Graph energy-based model (graph-ebm submodule).",
        requires=("graph_uq",),
        seed_is_backbone_rank=True,
    ),
    MethodSpec(
        "ensemble",
        "posthoc",
        build_ensemble,
        paper_name="Classical ensemble / CGNN by ensemble",
        description="Top-M vanilla backbones: *_classic = Classical Ensemble, *_credal = CGNN by Ensemble.",
        num_backbones=lambda cfg: int(cfg.get("M", 5)),
        score_keys=ENSEMBLE_SCORES,
    ),
    MethodSpec(
        "frozen",
        "posthoc",
        build_frozen,
        paper_name="CGNN post train",
        description="Credal head trained on the joint latent of a frozen backbone.",
        fit_on="train",
        seed_is_backbone_rank=True,
        score_keys=("EU", "AU", "TU"),
    ),
    MethodSpec(
        "cagcn",
        "posthoc",
        build_cagcn,
        paper_name="CaGCN",
        description="Calibration GCN trained on ID validation nodes; -MSP score.",
        fit_on="val",
        seed_is_backbone_rank=True,
    ),
]

for _spec in _SPECS:
    register_method(_spec)
