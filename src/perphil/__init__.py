"""perphil package init.

Expose a feature flag indicating whether Firedrake is available. Avoid raising at
import time so that submodules not requiring Firedrake (e.g., parameter dicts) can
still be imported in lightweight environments and CI.
"""

HAS_FIREDRAKE = False
try:  # pragma: no cover - trivial import guard
    import firedrake as _fd  # noqa: F401

    HAS_FIREDRAKE = True
except Exception:  # pragma: no cover
    HAS_FIREDRAKE = False

__all__ = ["HAS_FIREDRAKE"]
