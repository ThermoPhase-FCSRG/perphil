from __future__ import annotations

import os
from pathlib import Path

# Keep Firedrake/Loopy caches in a writable local path during tests.
_XDG_CACHE_HOME = Path(__file__).resolve().parents[1] / ".cache" / "perphil-runtime" / "xdg"
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

# Disk cache writes are a common source of flaky/read-only failures in CI/sandboxes.
os.environ.setdefault("LOOPY_NO_CACHE", "1")
