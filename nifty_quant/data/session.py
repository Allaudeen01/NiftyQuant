"""Session and contextual metadata for market snapshots.

Computes the "where in the trading day / expiry cycle / event calendar" fields
that make later option-chain analysis far easier, without changing the core
data model (these merge into the snapshot's generic ``context`` dict).
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def session_metadata(
    timestamp: datetime,
    expiry: date,
    *,
    session_open: time = SESSION_OPEN,
    session_close: time = SESSION_CLOSE,
    monthly_expiries: set[date] | None = None,
    events: dict[date, list[str]] | None = None,
) -> dict:
    """Contextual fields for one snapshot at ``timestamp`` for ``expiry``."""
    d = timestamp.date()
    open_dt = datetime.combine(d, session_open)
    close_dt = datetime.combine(d, session_close)
    mins_since_open = max(0, int((timestamp - open_dt).total_seconds() // 60))
    mins_until_close = max(0, int((close_dt - timestamp).total_seconds() // 60))
    day_events = list((events or {}).get(d, []))

    return {
        "session_id": d.isoformat(),
        "minutes_since_open": mins_since_open,
        "minutes_until_close": mins_until_close,
        "days_to_expiry": (expiry - d).days,
        "is_expiry_day": d == expiry,
        "is_monthly_expiry": (
            expiry in monthly_expiries if monthly_expiries is not None else None
        ),
        "expiry_iso_week": expiry.isocalendar()[1],
        "events": day_events,
        "is_event_day": bool(day_events),
    }


def monthly_expiry_set(expiries: list[date]) -> set[date]:
    """The last (monthly) expiry within each calendar month."""
    by_month: dict[tuple[int, int], date] = {}
    for e in expiries:
        key = (e.year, e.month)
        if key not in by_month or e > by_month[key]:
            by_month[key] = e
    return set(by_month.values())


def load_event_calendar(path: str | Path) -> dict[date, list[str]]:
    """Load an optional event calendar JSON: {"YYYY-MM-DD": ["Budget", ...]}."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[date, list[str]] = {}
    for k, v in raw.items():
        try:
            out[date.fromisoformat(k)] = list(v)
        except ValueError:
            continue
    return out
