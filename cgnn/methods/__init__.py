"""Method registry. Importing this package registers every method.

To add a method: register a ``MethodSpec`` in ``trainable.py`` or ``posthoc.py`` and add
``configs/methods/<name>.yaml``.
"""

from cgnn.methods import posthoc, trainable  # noqa: F401  (registration side effects)
from cgnn.methods.spec import METHODS, MethodSpec, RunContext, register_method


def get_method(name: str) -> MethodSpec:
    return METHODS.get(name)


__all__ = ["METHODS", "MethodSpec", "RunContext", "get_method", "register_method"]
