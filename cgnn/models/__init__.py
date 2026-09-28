"""LightningModules. All derive from :class:`cgnn.models.base.NodeUQModule`."""

from cgnn.models.base import NodeUQModule, TrainableUQModule
from cgnn.models.cagcn import CaGCNModule
from cgnn.models.credal import CredalGNN
from cgnn.models.frozen import CredalFrozenJoint
from cgnn.models.heads import CredalLayer
from cgnn.models.vanilla import VanillaGNN

__all__ = [
    "CaGCNModule",
    "CredalFrozenJoint",
    "CredalGNN",
    "CredalLayer",
    "NodeUQModule",
    "TrainableUQModule",
    "VanillaGNN",
]
