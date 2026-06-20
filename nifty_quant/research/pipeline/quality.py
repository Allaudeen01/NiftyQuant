"""Data Quality Gate (Stage 0) -- pure data-integrity checks.

This module is the pre-flight quality gate that runs *before* the read stage
(Req 19). It is deliberately separated from all I/O so the individual
``Quality_Check`` functions are fully unit-testable with synthetic snapshots
(Req 17.3): every check takes the already-read Target_Session inputs
(``OptionChain`` snapshots + ``Candle`` rows, both sourced through the existing
``Storage`` interface -- no new backend, Req 16.1) and returns a pure
``QualityCheckResult``.

Five checks mirror the design's Quality_Check table:

==========================  ===================================================
``check_api_outage``        expected-vs-actual candle count + long candle gaps
``check_duplicate_ts``      repeated chain ``timestamp`` or candle ``timestamp``
``check_expiry_mismatch``   chain expiry vs session date vs ``context`` metadata
``check_holiday``           weekend / non-trading day / zero weekday snapshots
``check_timezone_anomaly``  timestamps outside the exchange session window / tz
==========================  ===================================================

``run_quality_checks`` runs the fixed battery and returns the results in a
stable order. ``score_quality`` aggregates them into a ``Quality_Score`` +
PASS/FAIL verdict (Req 19.3).

Dataclasses mirror the existing ``nifty_quant`` style: ``from __future__ import
annotations`` plus frozen dataclasses holding no behaviour beyond simple typed
value objects (``QualityReport.failing_checks`` is a trivial derived view).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Sequence

from nifty_quant.data.models import Candle, OptionChain
from nifty_quant.data.session import SESSION_CLOSE, SESSION_OPEN

try:  # zoneinfo is stdlib on 3.9+, but guard so import never breaks the gate
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - defensive only
    ZoneInfo = None  # type: ignore[assignment]


# --- Check names (stable identifiers shared with QualityConfig weights) ------

API_OUTAGE = "api_outage"
DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
EXPIRY_MISMATCH = "expiry_mismatch"
HOLIDAY = "holiday"
TIMEZONE_ANOMALY = "timezone_anomaly"


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds and weights for the Data Quality Gate (Req 19)."""

    pass_threshold: int = 70          # PASS iff score >= threshold and no blocking check fails
    expected_candles: int | None = None   # None => count not enforced (gaps/emptiness only)
    max_gap_minutes: int = 15         # gap above this flags API outage / missing candles
    exchange_tz: str = "Asia/Kolkata"
    # per-check penalty weights (sum need not be 100; score is normalised in score_quality)
    weights: dict[str, int] = field(
        default_factory=lambda: {
            API_OUTAGE: 35,
            DUPLICATE_TIMESTAMPS: 15,
            EXPIRY_MISMATCH: 20,
            HOLIDAY: 20,
            TIMEZONE_ANOMALY: 10,
        }
    )
    blocking: frozenset[str] = frozenset({API_OUTAGE, HOLIDAY})


@dataclass(frozen=True)
class QualityCheckResult:
    """Outcome of one individual Quality_Check."""

    name: str                 # e.g. "duplicate_timestamps"
    passed: bool
    penalty: int              # 0 when passed; weighted penalty when failed
    detail: str               # human-readable explanation of what was detected
    blocking: bool            # if True, a failure forces an overall FAIL verdict


@dataclass(frozen=True)
class QualityReport:
    """Aggregated Quality_Score + verdict for the Target_Session (Req 19.3/19.5)."""

    session_id: str
    score: int                # Quality_Score, clamped to 0..100
    verdict: str              # "PASS" | "FAIL"
    checks: list[QualityCheckResult]

    @property
    def failing_checks(self) -> list[QualityCheckResult]:
        return [c for c in self.checks if not c.passed]


# --- Internal helpers --------------------------------------------------------


def _weight(config: QualityConfig, name: str) -> int:
    """Configured penalty weight for a check, defaulting to 0 if unset."""
    return int(config.weights.get(name, 0))


def _passed(config: QualityConfig, name: str, detail: str) -> QualityCheckResult:
    """Build a passing result (zero penalty) for ``name``."""
    return QualityCheckResult(
        name=name,
        passed=True,
        penalty=0,
        detail=detail,
        blocking=name in config.blocking,
    )


def _failed(config: QualityConfig, name: str, detail: str) -> QualityCheckResult:
    """Build a failing result carrying the configured weighted penalty."""
    return QualityCheckResult(
        name=name,
        passed=False,
        penalty=_weight(config, name),
        detail=detail,
        blocking=name in config.blocking,
    )


def _expected_offset(config: QualityConfig, on: date) -> timedelta | None:
    """Expected UTC offset for the configured exchange timezone on ``on``."""
    if ZoneInfo is None:
        return None
    try:
        tz = ZoneInfo(config.exchange_tz)
    except Exception:
        return None
    # Use a mid-session moment so DST (irrelevant for IST, but general) resolves.
    probe = datetime.combine(on, SESSION_OPEN, tzinfo=tz)
    return probe.utcoffset()


# --- The five Quality_Checks -------------------------------------------------


def check_api_outage(
    chains: Sequence[OptionChain],
    candles: Sequence[Candle],
    session: date,
    config: QualityConfig,
) -> QualityCheckResult:
    """Detect API-outage / missing-candle conditions.

    Fails when there are no candles at all, when the candle count falls short of
    ``config.expected_candles`` (when configured), or when any gap between
    consecutive candle timestamps exceeds ``config.max_gap_minutes``.
    """
    if not candles:
        return _failed(
            config,
            API_OUTAGE,
            f"no candles for session {session.isoformat()} (possible API outage)",
        )

    ordered = sorted(candles, key=lambda c: c.timestamp)

    if config.expected_candles is not None and len(ordered) < config.expected_candles:
        return _failed(
            config,
            API_OUTAGE,
            f"only {len(ordered)} candles, expected {config.expected_candles} "
            f"(missing candles / possible API outage)",
        )

    largest_gap = 0.0
    gap_at: datetime | None = None
    for prev, curr in zip(ordered, ordered[1:]):
        gap_minutes = (curr.timestamp - prev.timestamp).total_seconds() / 60.0
        if gap_minutes > largest_gap:
            largest_gap = gap_minutes
            gap_at = prev.timestamp

    if largest_gap > config.max_gap_minutes:
        return _failed(
            config,
            API_OUTAGE,
            f"candle gap of {largest_gap:.0f}m after {gap_at} exceeds "
            f"max_gap_minutes={config.max_gap_minutes} (possible API outage)",
        )

    return _passed(
        config,
        API_OUTAGE,
        f"{len(ordered)} candles, largest gap {largest_gap:.0f}m "
        f"<= {config.max_gap_minutes}m",
    )


def check_duplicate_timestamps(
    chains: Sequence[OptionChain],
    candles: Sequence[Candle],
    session: date,
    config: QualityConfig,
) -> QualityCheckResult:
    """Detect duplicate timestamps in chain snapshots or candles.

    A healthy session has one snapshot/candle per instant; repeated timestamps
    indicate a double-write or a collection bug.
    """
    chain_ts = [c.timestamp for c in chains]
    candle_ts = [c.timestamp for c in candles]

    dup_chain = _duplicates(chain_ts)
    dup_candle = _duplicates(candle_ts)

    if dup_chain or dup_candle:
        parts: list[str] = []
        if dup_chain:
            parts.append(f"{len(dup_chain)} duplicate chain timestamp(s): {dup_chain[0]}")
        if dup_candle:
            parts.append(f"{len(dup_candle)} duplicate candle timestamp(s): {dup_candle[0]}")
        return _failed(config, DUPLICATE_TIMESTAMPS, "; ".join(parts))

    return _passed(
        config,
        DUPLICATE_TIMESTAMPS,
        f"no duplicate timestamps across {len(chain_ts)} chains and "
        f"{len(candle_ts)} candles",
    )


def check_expiry_mismatch(
    chains: Sequence[OptionChain],
    candles: Sequence[Candle],
    session: date,
    config: QualityConfig,
) -> QualityCheckResult:
    """Detect expiry mismatches in the Target_Session chains.

    Cross-checks each chain's ``expiry`` against the session date and against
    the session metadata carried in ``OptionChain.context``
    (``days_to_expiry`` / ``is_expiry_day``). Also flags chains whose expiry has
    already passed relative to the session.
    """
    for chain in chains:
        expected_dte = (chain.expiry - session).days

        if chain.expiry < session:
            return _failed(
                config,
                EXPIRY_MISMATCH,
                f"chain expiry {chain.expiry.isoformat()} is before session "
                f"{session.isoformat()} (expired contract)",
            )

        ctx = chain.context or {}
        ctx_dte = ctx.get("days_to_expiry")
        if ctx_dte is not None and int(ctx_dte) != expected_dte:
            return _failed(
                config,
                EXPIRY_MISMATCH,
                f"context days_to_expiry={ctx_dte} != expected {expected_dte} "
                f"for expiry {chain.expiry.isoformat()} on {session.isoformat()}",
            )

        ctx_is_expiry = ctx.get("is_expiry_day")
        expected_is_expiry = chain.expiry == session
        if ctx_is_expiry is not None and bool(ctx_is_expiry) != expected_is_expiry:
            return _failed(
                config,
                EXPIRY_MISMATCH,
                f"context is_expiry_day={ctx_is_expiry} != expected "
                f"{expected_is_expiry} for expiry {chain.expiry.isoformat()}",
            )

    return _passed(
        config,
        EXPIRY_MISMATCH,
        f"expiry metadata consistent across {len(chains)} chain snapshot(s)",
    )


def check_holiday(
    chains: Sequence[OptionChain],
    candles: Sequence[Candle],
    session: date,
    config: QualityConfig,
) -> QualityCheckResult:
    """Detect holiday / non-trading-day conditions.

    Fails when the session date falls on a weekend, or when it is a weekday but
    no chain snapshots were collected (a non-trading day or a full-day outage).
    """
    # Monday=0 .. Sunday=6; Saturday/Sunday are non-trading days.
    if session.weekday() >= 5:
        weekday_name = session.strftime("%A")
        return _failed(
            config,
            HOLIDAY,
            f"session {session.isoformat()} falls on a {weekday_name} "
            f"(non-trading day)",
        )

    if not chains:
        return _failed(
            config,
            HOLIDAY,
            f"no option-chain snapshots on weekday {session.isoformat()} "
            f"(possible holiday / non-trading day)",
        )

    return _passed(
        config,
        HOLIDAY,
        f"session {session.isoformat()} is a weekday with "
        f"{len(chains)} chain snapshot(s)",
    )


def check_timezone_anomaly(
    chains: Sequence[OptionChain],
    candles: Sequence[Candle],
    session: date,
    config: QualityConfig,
) -> QualityCheckResult:
    """Detect timezone anomalies in chain/candle timestamps.

    Flags timestamps that fall outside the expected exchange session window
    ``[SESSION_OPEN, SESSION_CLOSE]`` or that carry a UTC offset different from
    the configured exchange timezone (``config.exchange_tz``).
    """
    expected_offset = _expected_offset(config, session)

    timestamps: list[datetime] = [c.timestamp for c in chains]
    timestamps.extend(c.timestamp for c in candles)

    for ts in timestamps:
        # Offset check (only when both the timestamp and the tz resolve an offset).
        if expected_offset is not None and ts.utcoffset() is not None:
            if ts.utcoffset() != expected_offset:
                return _failed(
                    config,
                    TIMEZONE_ANOMALY,
                    f"timestamp {ts} has offset {ts.utcoffset()} != expected "
                    f"{expected_offset} for {config.exchange_tz}",
                )

        # Window check against the exchange session hours.
        if ts.time() < SESSION_OPEN or ts.time() > SESSION_CLOSE:
            return _failed(
                config,
                TIMEZONE_ANOMALY,
                f"timestamp {ts} is outside the exchange session window "
                f"[{SESSION_OPEN.isoformat()}, {SESSION_CLOSE.isoformat()}]",
            )

    return _passed(
        config,
        TIMEZONE_ANOMALY,
        f"all {len(timestamps)} timestamp(s) within the exchange session window",
    )


def run_quality_checks(
    chains: Sequence[OptionChain],
    candles: Sequence[Candle],
    session: date,
    config: QualityConfig,
) -> list[QualityCheckResult]:
    """Run the fixed battery of Quality_Checks for the Target_Session (Req 19.2).

    Operates only on the already-read ``Storage``-sourced inputs and introduces
    no new backend (Req 16.1). Results are returned in a stable order so that
    downstream scoring/reporting is deterministic.
    """
    return [
        check_api_outage(chains, candles, session, config),
        check_duplicate_timestamps(chains, candles, session, config),
        check_expiry_mismatch(chains, candles, session, config),
        check_holiday(chains, candles, session, config),
        check_timezone_anomaly(chains, candles, session, config),
    ]


def score_quality(
    checks: Sequence[QualityCheckResult],
    config: QualityConfig,
    session_id: str = "",
) -> QualityReport:
    """Aggregate Quality_Checks into a ``Quality_Score`` + PASS/FAIL verdict (Req 19.3).

    Pure function -- no I/O, no globals -- so it is deterministic across identical
    inputs and fully unit-testable:

    * ``score = clamp(100 - sum(penalties), 0, 100)`` where the penalties are the
      ``penalty`` values of the supplied checks (passing checks carry zero).
    * ``verdict`` is ``"PASS"`` if and only if ``score >= config.pass_threshold``
      **and** no blocking check failed; otherwise ``"FAIL"``.

    A check is "blocking" when its ``blocking`` flag is set; a single failed
    blocking check forces a ``FAIL`` regardless of the score (Req 19.4).

    ``session_id`` is threaded straight onto the returned ``QualityReport`` for
    report/orchestration use; it does not affect the score or verdict.
    """
    checks = list(checks)

    total_penalty = sum(c.penalty for c in checks)
    score = max(0, min(100, 100 - total_penalty))

    blocking_failed = any((not c.passed) and c.blocking for c in checks)
    verdict = "PASS" if score >= config.pass_threshold and not blocking_failed else "FAIL"

    return QualityReport(
        session_id=session_id,
        score=score,
        verdict=verdict,
        checks=checks,
    )


def _duplicates(values: Sequence[datetime]) -> list[datetime]:
    """Return the timestamps that appear more than once, in first-seen order."""
    seen: set[datetime] = set()
    dups: list[datetime] = []
    reported: set[datetime] = set()
    for v in values:
        if v in seen and v not in reported:
            dups.append(v)
            reported.add(v)
        seen.add(v)
    return dups
