"""Minimal, dependency-free .env loader.

Parses KEY=VALUE lines into os.environ. Handles the things that trip people up:
- full-line comments (``# ...``) and blank lines are ignored,
- inline comments (`` # ...`` after a value) are stripped,
- surrounding single/double quotes are removed,
- surrounding whitespace is trimmed.

Used by the runnable scripts so credentials load consistently regardless of
shell quirks. Secrets are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", *, override: bool = True) -> list[str]:
    """Load KEY=VALUE pairs from ``path`` into os.environ.

    Returns the list of keys that were set (names only, never values).
    """
    p = Path(path)
    if not p.exists():
        return []

    keys: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        keys.append(key)
        if override or key not in os.environ:
            os.environ[key] = _clean_value(value)
    return keys


def _clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    # Strip an inline comment that begins with whitespace + '#'.
    for marker in (" #", "\t#"):
        pos = value.find(marker)
        if pos != -1:
            value = value[:pos]
    return value.strip()
