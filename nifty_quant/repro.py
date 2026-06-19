"""Reproducibility metadata.

Captures the environment needed to reproduce an experiment exactly: Python
version, platform, key dependency versions, and a stable hash of the run
configuration. Stored alongside each experiment so a result months from now can
be traced to the exact code/config/environment that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata

# Packages whose versions materially affect numerical results.
_DEFAULT_PACKAGES = ("numpy", "pandas", "pyarrow", "matplotlib", "pytest")


def package_versions(packages: tuple[str, ...] = _DEFAULT_PACKAGES) -> dict[str, str]:
    """Return {package: version} for installed packages (missing -> 'not installed')."""
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def capture_environment(
    packages: tuple[str, ...] = _DEFAULT_PACKAGES,
) -> dict:
    """Snapshot the runtime environment for reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(packages),
    }


def config_hash(config: dict, *, length: int = 16) -> str:
    """Stable short SHA-256 of a configuration dict (order-independent)."""
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
