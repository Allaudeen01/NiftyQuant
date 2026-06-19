"""Event sourcing for the backtest/execution pipeline.

Every meaningful step is recorded as a typed, timestamped :class:`JournalRecord`.
The full sequence (candle -> signal -> intent -> risk -> order -> fill ->
position) can be replayed or exported to JSON Lines to understand exactly what
happened and why. The journal also mirrors records to the structured logger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from nifty_quant.log import get_logger

_log = get_logger("backtest.journal")


class EventType(str, Enum):
    MARKET_EVENT = "market_event"
    SIGNAL_GENERATED = "signal_generated"
    INTENT_CREATED = "intent_created"
    RISK_APPROVED = "risk_approved"
    RISK_REJECTED = "risk_rejected"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_UNFILLED = "order_unfilled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"


@dataclass(frozen=True)
class JournalRecord:
    timestamp: datetime
    event_type: EventType
    payload: dict

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "event_type": self.event_type.value,
                **self.payload,
            },
            default=str,
            separators=(",", ":"),
        )


@dataclass
class Journal:
    """An append-only log of pipeline events."""

    records: list[JournalRecord] = field(default_factory=list)
    log_records: bool = True

    def record(
        self, timestamp: datetime, event_type: EventType, **payload
    ) -> JournalRecord:
        rec = JournalRecord(timestamp, event_type, payload)
        self.records.append(rec)
        if self.log_records:
            _log.event(event_type.value, **payload)
        return rec

    def filter(self, event_type: EventType) -> list[JournalRecord]:
        return [r for r in self.records if r.event_type is event_type]

    def to_jsonl(self) -> str:
        return "\n".join(r.to_json() for r in self.records)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")
