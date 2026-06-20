"""Data-quality validation for OHLCV history.

Runs a battery of checks over an :class:`OHLCVSeries` and produces a structured
:class:`DataQualityReport`. The goal: never benchmark or backtest on a dataset
until it has demonstrably passed validation.

Checks (severity):
- non_empty (CRITICAL)
- timestamps_sorted_unique (CRITICAL): strictly ascending, no duplicates
- ohlc_integrity (CRITICAL): high>=low, high>=max(o,c), low<=min(o,c), positive
- no_weekend_bars (WARNING): NSE doesn't trade weekends
- within_session (WARNING): intraday bars fall inside the trading session
- intraday_completeness (WARNING): each day has ~expected bars
- trading_day_coverage (INFO/WARNING): distinct sessions vs business days; big gaps
- outliers (WARNING): implausible bar-to-bar returns (likely bad ticks)
- date_range_covered (WARNING): data spans the requested window

"Passed" means no CRITICAL check failed. WARNING/INFO are reported for review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from nifty_quant.data.models import OHLCVSeries

CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

# NSE regular session 09:15–15:30 == 375 minutes.
_DEFAULT_SESSION_MINUTES = 375
_INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10,
    "15m": 15, "30m": 30, "1h": 60, "60m": 60,
}


@dataclass
class QualityCheck:
    name: str
    severity: str
    passed: bool
    detail: str
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    start: str
    end: str
    num_bars: int
    checks: list[QualityCheck] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(
            (c.severity == CRITICAL and not c.passed) for c in self.checks
        )

    def failures(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "num_bars": self.num_bars,
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "summary": self.summary,
        }

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            f"DATA QUALITY REPORT - {self.symbol} {self.timeframe}",
            f"Window: {self.start} -> {self.end}",
            f"Bars: {self.num_bars}",
            f"Overall: {'PASS' if self.passed else 'FAIL'}",
            "=" * 60,
        ]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"[{mark}] ({c.severity}) {c.name}: {c.detail}")
        return "\n".join(lines)


def validate_ohlcv(
    series: OHLCVSeries,
    *,
    expected_start: date | None = None,
    expected_end: date | None = None,
    session_minutes: int = _DEFAULT_SESSION_MINUTES,
    completeness_ratio: float = 0.9,
    outlier_return: float = 0.20,
) -> DataQualityReport:
    """Validate an OHLCV series and return a structured quality report."""
    df = series.to_frame()
    checks: list[QualityCheck] = []
    report = DataQualityReport(
        symbol=series.symbol,
        timeframe=series.timeframe,
        start=expected_start.isoformat() if expected_start else "",
        end=expected_end.isoformat() if expected_end else "",
        num_bars=len(df),
    )

    # 1. non-empty
    if df.empty:
        checks.append(QualityCheck(
            "non_empty", CRITICAL, False, "series contains no candles"))
        report.checks = checks
        return report
    checks.append(QualityCheck(
        "non_empty", CRITICAL, True, f"{len(df)} bars present"))

    idx = df.index

    # 2. sorted + unique timestamps
    n_dupes = int(idx.duplicated().sum())
    is_sorted = bool(idx.is_monotonic_increasing)
    checks.append(QualityCheck(
        "timestamps_sorted_unique", CRITICAL,
        is_sorted and n_dupes == 0,
        f"sorted={is_sorted}, duplicates={n_dupes}",
        {"duplicates": n_dupes, "sorted": is_sorted},
    ))

    # 3. OHLC integrity
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad = (
        (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
        | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    )
    n_bad = int(bad.sum())
    checks.append(QualityCheck(
        "ohlc_integrity", CRITICAL, n_bad == 0,
        f"{n_bad} bars violate OHLC bounds / positivity",
        {"violations": n_bad},
    ))

    # 4. weekend bars
    weekend = idx.weekday >= 5
    n_weekend = int(weekend.sum())
    checks.append(QualityCheck(
        "no_weekend_bars", WARNING, n_weekend == 0,
        f"{n_weekend} bars fall on a weekend",
        {"weekend_bars": n_weekend},
    ))

    interval_min = _INTERVAL_MINUTES.get(series.timeframe)
    is_intraday = interval_min is not None

    # 5. within session (intraday only)
    if is_intraday:
        minutes = idx.hour * 60 + idx.minute
        start_min = 9 * 60 + 15
        end_min = start_min + session_minutes
        out_of_session = (minutes < start_min) | (minutes >= end_min)
        n_oos = int(out_of_session.sum())
        checks.append(QualityCheck(
            "within_session", WARNING, n_oos == 0,
            f"{n_oos} bars outside 09:15-15:30",
            {"out_of_session": n_oos},
        ))

    # 6. completeness
    day_counts = df.groupby(idx.date).size()
    if is_intraday:
        expected = max(int(round(session_minutes / interval_min)), 1)
        partial = day_counts[day_counts < completeness_ratio * expected]
        worst = partial.sort_values().head(10)
        checks.append(QualityCheck(
            "intraday_completeness", WARNING, len(partial) == 0,
            f"{len(partial)} of {len(day_counts)} sessions have "
            f"< {completeness_ratio:.0%} of {expected} expected bars",
            {
                "expected_per_day": expected,
                "incomplete_sessions": int(len(partial)),
                "worst": {str(k): int(v) for k, v in worst.items()},
            },
        ))
    else:
        # daily: exactly one bar per trading day
        multi = day_counts[day_counts > 1]
        checks.append(QualityCheck(
            "daily_one_bar_per_day", WARNING, len(multi) == 0,
            f"{len(multi)} days have more than one bar",
            {"multi_bar_days": int(len(multi))},
        ))

    # 7. trading-day coverage + gaps
    distinct_days = sorted(set(idx.date))
    first, last = distinct_days[0], distinct_days[-1]
    business_days = pd.bdate_range(first, last)
    coverage = len(distinct_days) / max(len(business_days), 1)
    day_series = pd.Series(pd.to_datetime(distinct_days))
    gaps = day_series.diff().dt.days.dropna()
    # Gap in *business* days between consecutive present sessions.
    max_gap_bdays = _max_business_gap(distinct_days)
    big_gap = max_gap_bdays > 5
    checks.append(QualityCheck(
        "trading_day_coverage", WARNING, not big_gap,
        f"{len(distinct_days)} sessions over {len(business_days)} business days "
        f"(coverage {coverage:.1%}); largest gap {max_gap_bdays} business days",
        {
            "sessions": len(distinct_days),
            "business_days": int(len(business_days)),
            "coverage": coverage,
            "max_gap_business_days": int(max_gap_bdays),
        },
    ))

    # 8. outliers (bar-to-bar close returns)
    returns = c.pct_change().abs().dropna()
    outliers = returns[returns > outlier_return]
    checks.append(QualityCheck(
        "outliers", WARNING, len(outliers) == 0,
        f"{len(outliers)} bar-to-bar moves exceed {outlier_return:.0%}",
        {
            "count": int(len(outliers)),
            "max_return": float(returns.max()) if len(returns) else 0.0,
        },
    ))

    # 9. requested range covered
    if expected_start and expected_end:
        covers_start = first <= expected_start or (
            (first - expected_start).days <= 5
        )
        covers_end = last >= expected_end or ((expected_end - last).days <= 5)
        checks.append(QualityCheck(
            "date_range_covered", WARNING, covers_start and covers_end,
            f"data spans {first} -> {last}; requested "
            f"{expected_start} -> {expected_end}",
            {"first": str(first), "last": str(last)},
        ))

    report.checks = checks
    report.summary = {
        "first_bar": str(idx.min()),
        "last_bar": str(idx.max()),
        "distinct_sessions": len(distinct_days),
        "is_intraday": is_intraday,
        "failed_checks": [c.name for c in checks if not c.passed],
    }
    return report


def _max_business_gap(days: list[date]) -> int:
    """Largest gap, in business days, between consecutive present sessions."""
    max_gap = 0
    for prev, cur in zip(days, days[1:]):
        # business days strictly between prev and cur
        gap = len(pd.bdate_range(prev, cur)) - 1  # excludes prev
        # subtract the 'cur' day itself to count missing days between
        missing = max(gap - 1, 0)
        max_gap = max(max_gap, missing)
    return max_gap
