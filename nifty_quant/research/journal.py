"""Research journal: a knowledge base of hypotheses and their verdicts.

Distinct from the backtest event-sourcing journal. This records the *research*
itself -- each hypothesis, its status, confidence, and the reason for the
verdict -- so that after dozens of experiments you have an auditable record of
what was tried and why it was accepted or rejected (and don't re-test dead
ideas). Stored as append-friendly JSON Lines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

VALID_STATUSES = {"open", "testing", "supported", "rejected", "inconclusive"}


@dataclass
class Hypothesis:
    id: int
    timestamp: str
    hypothesis: str
    status: str = "open"
    confidence: float | None = None
    reason: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchJournal:
    def __init__(self, path: str | Path = "reports/research_journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[Hypothesis]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Hypothesis(**json.loads(line)))
        return out

    def _next_id(self) -> int:
        items = self._read_all()
        return (max((h.id for h in items), default=0)) + 1

    def add(
        self,
        hypothesis: str,
        *,
        status: str = "open",
        confidence: float | None = None,
        reason: str = "",
        tags: list[str] | None = None,
    ) -> Hypothesis:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        h = Hypothesis(
            id=self._next_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            hypothesis=hypothesis,
            status=status,
            confidence=confidence,
            reason=reason,
            tags=tags or [],
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(h.to_dict(), default=str) + "\n")
        return h

    def update(self, hypothesis_id: int, **fields) -> Hypothesis:
        items = self._read_all()
        found = None
        for h in items:
            if h.id == hypothesis_id:
                for k, v in fields.items():
                    if not hasattr(h, k):
                        raise AttributeError(f"unknown field {k!r}")
                    if k == "status" and v not in VALID_STATUSES:
                        raise ValueError(f"invalid status {v!r}")
                    setattr(h, k, v)
                found = h
        if found is None:
            raise KeyError(f"no hypothesis with id {hypothesis_id}")
        # Rewrite the file with the updated set.
        with self.path.open("w", encoding="utf-8") as fh:
            for h in items:
                fh.write(json.dumps(h.to_dict(), default=str) + "\n")
        return found

    def list(self, status: str | None = None) -> list[Hypothesis]:
        items = self._read_all()
        if status is not None:
            items = [h for h in items if h.status == status]
        return items

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([h.to_dict() for h in self._read_all()])
