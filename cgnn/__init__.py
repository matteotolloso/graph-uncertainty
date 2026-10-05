"""Credal Graph Neural Networks (CGNN) and node-level uncertainty baselines.

Package map:

- ``cgnn.uncertainty`` - credal-set math: interval softmax, reachable bounds,
  min/max entropy (AU/TU/EU), credal loss, ensemble decomposition.
- ``cgnn.data``        - dataset registry, leave-out-class OOD splits, loaders.
- ``cgnn.models``      - LightningModules (backbones, credal models, detectors).
- ``cgnn.methods``     - method registry: how each experiment is built and run.
- ``cgnn.runner``      - the single generic experiment runner.
- ``cgnn.cli``         - ``cgnn`` command line (list/describe/run/sweep/results/doctor).
"""

__version__ = "1.0.0"
