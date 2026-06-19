"""Structured alerts emitted by the validation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class AlertLevel(IntEnum):
    """Severity, ordered so comparisons/sorting are meaningful."""

    INFO = 10
    WARNING = 20
    CRITICAL = 30


@dataclass(frozen=True)
class Alert:
    level: AlertLevel
    code: str                 # machine-readable, e.g. "sharpe_drift"
    message: str
    metric: str | None = None
    observed: float | None = None
    expected: float | None = None
    context: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "level": self.level.name,
            "code": self.code,
            "message": self.message,
            "metric": self.metric,
            "observed": self.observed,
            "expected": self.expected,
            "context": dict(self.context),
        }
