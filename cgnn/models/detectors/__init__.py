"""Post-hoc detectors on frozen VanillaGNN backbones.

Optional dependencies are imported lazily: ``faiss`` (kNN), ``torch_sparse``
(GNNSafe), ``graph_uq`` (GEBM).
"""

from cgnn.models.detectors.base import PostHocDetector, load_frozen_backbone
from cgnn.models.detectors.ensemble import CredalEnsemble
from cgnn.models.detectors.feature_based import KNNDetector, KNNJointDetector, MahalanobisDetector
from cgnn.models.detectors.gebm import GEBMDetector
from cgnn.models.detectors.logit_based import EnergyDetector, GNNSafeDetector, ODINDetector

__all__ = [
    "CredalEnsemble",
    "EnergyDetector",
    "GEBMDetector",
    "GNNSafeDetector",
    "KNNDetector",
    "KNNJointDetector",
    "MahalanobisDetector",
    "ODINDetector",
    "PostHocDetector",
    "load_frozen_backbone",
]
