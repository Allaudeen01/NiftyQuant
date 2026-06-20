"""Pipeline stage logic for the daily-research-pipeline.

Each stage is a small, pure-ish function operating on explicit inputs so it is
independently unit-testable with synthetic data (Req 17.3) -- there is no hidden
global state. The orchestrator (a later task) wires the stages together in the
fixed order ``quality-gate -> read -> features -> validation -> comparison ->
evidence -> ideas -> report`` and routes each call through its cost-capturing
wrapper.

This module currently implements:

* ``StageError`` -- the shared, minimal exception every stage raises to signal an
  unrecoverable condition. The orchestrator catches it, records the failing
  stage name + detail, and returns a non-zero exit code.
* ``read_stage`` (Stage 1, Requirements 2, 16.1) -- reads the Target_Session
  option-chain snapshots and candles through the **existing** ``Storage``
  interface (no new backend) and packages them into a ``ReadResult``.

It mirrors the existing ``nifty_quant`` style: ``from __future__ import
annotations`` plus frozen dataclasses that hold no behaviour beyond being typed
value objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

from nifty_quant.analytics import options as analytics
from nifty_quant.data.models import OHLCVSeries, OptionChain
from nifty_quant.feed.events import CandleEvent, OptionChainEvent
from nifty_quant.research.pipeline.experiment import (
    decode_experiment,
    merge_experiment_tags,
)
from nifty_quant.research.pipeline.models import (
    ComparisonResult,
    EvidenceChange,
    ExperimentRecord,
    MarketSummary,
    ResearchIdea,
    SessionMetrics,
    UnusualEvent,
)
from nifty_quant.validation.alerts import AlertLevel
from nifty_quant.validation.engine import Baseline, ValidationEngine, ValidationThresholds

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from nifty_quant.data.storage.base import Storage
    from nifty_quant.features.engine import FeatureEngine
    from nifty_quant.features.store import FeatureStore
    from nifty_quant.features.vector import FeatureVector
    from nifty_quant.research.journal import Hypothesis, ResearchJournal
    from nifty_quant.research.pipeline.evidence import EvidenceEngine, Verdict


class StageError(Exception):
    """Unrecoverable error raised by a pipeline stage.

    Carries the ``stage_name`` that failed and a human-readable ``detail`` so the
    orchestrator can record the failing stage and surface a non-zero exit code
    without re-parsing the message (Req 1.5). Kept deliberately minimal and
    reusable so every later stage (``quality_gate``, ``read``, ...) raises the
    same type.
    """

    def __init__(self, stage_name: str, detail: str) -> None:
        self.stage_name = stage_name
        self.detail = detail
        super().__init__(f"[{stage_name}] {detail}")


@dataclass(frozen=True)
class ReadResult:
    """Output of the read stage (Stage 1) for one Target_Session.

    Holds both the full ordered snapshot list (for intraday metrics) and the
    selected representative end-of-day snapshot (for single-snapshot analytics),
    alongside the session's candles. ``OptionChain.context`` -- carrying
    ``india_vix``, ``days_to_expiry``, ``minutes_since_open``, ``is_expiry_day``
    -- is preserved unchanged on every snapshot for downstream stages (Req 2.4).
    """

    target_session: date
    chains: list[OptionChain]   # full Target_Session snapshots, ascending by timestamp
    eod_chain: OptionChain      # representative end-of-day (latest timestamp) snapshot
    candles: OHLCVSeries        # Target_Session candles, ascending by timestamp


# Day window bounds for a Target_Session read: ``[00:00:00, 23:59:59]``.
# ``read_option_chains`` / ``read_candles`` read ``[start, end]`` inclusive, so
# this captures every snapshot/candle collected on the session date.
_DAY_START = time(0, 0, 0)
_DAY_END = time(23, 59, 59)

# Replay ordering for events sharing a timestamp: apply the option-chain snapshot
# before the candle so the candle's emitted vector reflects the synchronized
# option-derived features.
_OPTION_CHAIN_PRIORITY = 0
_CANDLE_PRIORITY = 1


def read_stage(
    storage: "Storage",
    underlying: str,
    timeframe: str,
    target: date,
    prior_window: object | None = None,
) -> ReadResult:
    """Read the Target_Session option chains and candles from the Warehouse.

    Uses only the existing ``Storage`` interface -- ``read_option_chains`` and
    ``read_candles`` -- over the Target_Session day window ``[00:00, 23:59:59]``
    and introduces no new backend (Req 2.1, 2.2, 2.5, 16.1).

    The full ordered snapshot list is retained for intraday metrics while the
    latest-``timestamp`` snapshot is selected as the representative end-of-day
    snapshot for single-snapshot analytics. Every snapshot's ``context`` is
    passed through untouched so the synchronized market/session metadata is
    available downstream (Req 2.4).

    ``prior_window`` is accepted to match the orchestration stage contract and
    is reserved for later history needs; the read itself is scoped to the
    Target_Session day window.

    Raises:
        StageError: tagged ``"read"`` when the Target_Session has zero
            option-chain snapshots, so the orchestrator exits non-zero (Req 2.3).
    """
    start = datetime.combine(target, _DAY_START)
    end = datetime.combine(target, _DAY_END)

    # Option chains: ascending by snapshot time per the Storage contract.
    chains: list[OptionChain] = list(
        storage.read_option_chains(underlying, start, end)
    )

    if not chains:
        raise StageError(
            "read",
            f"no option-chain data for {target.isoformat()}",
        )

    # Candles for the same day window through the existing interface (Req 2.2).
    candles = storage.read_candles(underlying, timeframe, start, end)

    eod_chain = _select_eod_snapshot(chains)

    return ReadResult(
        target_session=target,
        chains=chains,
        eod_chain=eod_chain,
        candles=candles,
    )


def _select_eod_snapshot(chains: Sequence[OptionChain]) -> OptionChain:
    """Return the representative end-of-day snapshot (latest ``timestamp``).

    ``read_option_chains`` already returns snapshots ascending by snapshot time,
    but selecting via ``max`` keeps the choice correct regardless of input order
    and preserves the chosen snapshot's ``context`` unchanged.
    """
    return max(chains, key=lambda c: c.timestamp)


@dataclass(frozen=True)
class FeatureResult:
    """Output of the feature stage (Stage 2) for one Target_Session.

    Holds the ordered list of emitted :class:`FeatureVector` objects (one per
    candle, with the latest option-derived features folded in) and the
    ``feature_version`` that produced them (from ``FeatureEngine.version``), so
    incompatible feature sets are never mixed downstream (Req 3.3).
    """

    vectors: list["FeatureVector"]   # emitted vectors, ascending by timestamp
    feature_version: str             # FeatureEngine.version that produced them


def feature_stage(
    feature_engine: "FeatureEngine",
    feature_store: "FeatureStore",
    read_result: ReadResult,
) -> FeatureResult:
    """Replay the Target_Session events through the engine and persist features.

    Interleaves the session's option-chain snapshots and candles in strict
    timestamp order and feeds them to the **frozen** ``FeatureEngine`` (Req 3.1,
    16.2): each ``OptionChain`` is replayed via ``on_option_chain`` (updating the
    engine's latest option-derived features) and each ``Candle`` via
    ``on_candle``, which emits a ``FeatureVector`` with those option features
    folded in. All option analytics come exclusively from ``analytics.options``
    through the engine -- none are reimplemented here (Req 3.5, 16.3).

    When an option-chain snapshot and a candle share a timestamp the snapshot is
    applied first, so the candle's emitted vector reflects the synchronized
    option features. Each emitted vector is persisted via ``feature_store.put``
    carrying ``FeatureEngine.version`` (Req 3.2, 3.3). Because the engine is
    deterministic and ``ParquetFeatureStore.put`` de-duplicates on key keeping
    last, re-running the stage for the same session yields identical stored
    values (Req 3.4).
    """
    symbol = read_result.candles.symbol
    timeframe = read_result.candles.timeframe

    # Build a single time-ordered event stream. ``_OPTION_CHAIN_PRIORITY`` <
    # ``_CANDLE_PRIORITY`` so a snapshot sharing a timestamp with a candle is
    # folded into that candle's emitted vector. Python's sort is stable, so
    # events with equal (timestamp, priority) keep their warehouse order.
    ordered: list[tuple[datetime, int, OptionChain | None, object]] = []
    for chain in read_result.chains:
        ordered.append((chain.timestamp, _OPTION_CHAIN_PRIORITY, chain, None))
    for candle in read_result.candles.candles:
        ordered.append((candle.timestamp, _CANDLE_PRIORITY, None, candle))
    ordered.sort(key=lambda item: (item[0], item[1]))

    vectors: list["FeatureVector"] = []
    for timestamp, priority, chain, candle in ordered:
        if priority == _OPTION_CHAIN_PRIORITY:
            feature_engine.on_option_chain(
                OptionChainEvent(timestamp=timestamp, chain=chain)
            )
        else:
            fv = feature_engine.on_candle(
                CandleEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    timeframe=timeframe,
                    candle=candle,
                )
            )
            feature_store.put(fv)
            vectors.append(fv)

    return FeatureResult(vectors=vectors, feature_version=feature_engine.version)


@dataclass(frozen=True)
class CapturedAlert:
    """One :class:`~nifty_quant.validation.alerts.Alert` captured for the report.

    A flattened, render-ready value object holding the alert's severity ``level``
    name (e.g. ``"INFO"``/``"WARNING"``/``"CRITICAL"``), its machine-readable
    ``code``, and its human-readable ``message`` so the report can list every
    alert the Validation_Engine produced (Req 4.2) without re-importing the
    validation types.
    """

    level: str   # AlertLevel name, e.g. "INFO" | "WARNING" | "CRITICAL"
    code: str    # machine-readable, e.g. "insufficient_data" | "feature_drift"
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Output of the validation stage (Stage 3) for one Target_Session.

    Captures every emitted alert's ``(level, code, message)`` for the report
    (Req 4.2) and surfaces insufficient-data / informational conditions as
    ``notes`` rather than failures (Req 4.3). ``passed`` mirrors the engine's
    own verdict (nothing at WARNING level or above fired); the stage itself
    never raises for an insufficient-data condition.
    """

    alerts: tuple[CapturedAlert, ...] = ()
    notes: tuple[str, ...] = ()
    passed: bool = True


def validation_stage(
    feature_result: FeatureResult,
    prior_features: Sequence[FeatureResult] = (),
    *,
    thresholds: ValidationThresholds | None = None,
) -> ValidationResult:
    """Run drift/health checks for the Target_Session via the Validation_Engine.

    Builds a :class:`~nifty_quant.validation.engine.Baseline` whose
    ``feature_distributions`` are drawn from the prior-session feature values
    (``prior_features``) and calls
    :meth:`~nifty_quant.validation.engine.ValidationEngine.validate` with the
    Target_Session feature arrays (from ``feature_result``) as
    ``current_features`` so the **existing** drift detection runs (Req 4.1).
    No drift or health-check logic is reimplemented here -- the stage only wires
    the inputs into the frozen engine (Req 4.4, 16.4).

    Because the daily pipeline has no live equity/trades, an empty
    ``pd.Series`` and an empty trade list are passed; the engine then defers the
    trade-based metric checks with an ``insufficient_data`` INFO alert. Every
    emitted alert's level/code/message is captured for the report (Req 4.2), and
    informational (INFO-level) conditions -- including the insufficient-data
    deferral -- are surfaced as ``notes`` rather than treated as a failure
    (Req 4.3).

    Args:
        feature_result: the Target_Session features (``current_features`` source).
        prior_features: prior-session ``FeatureResult``s whose values seed the
            baseline distributions; may be empty (e.g. the first collected
            session), in which case drift detection has no baseline to compare
            against and simply contributes no drift alerts.
        thresholds: optional override for the engine's
            :class:`~nifty_quant.validation.engine.ValidationThresholds`.

    Returns:
        ValidationResult: captured alerts, informational notes, and the engine's
        pass/fail verdict.
    """
    baseline_distributions = _feature_distributions(
        fv for fr in prior_features for fv in fr.vectors
    )
    current_features = _feature_distributions(feature_result.vectors)

    # Performance metrics are neutral/NaN-safe defaults: the daily pipeline has
    # no trades, so the engine never reaches the metric checks (it defers with
    # an insufficient_data INFO alert). Only feature_distributions matter here.
    baseline = Baseline(
        sharpe=0.0,
        win_rate=0.0,
        max_drawdown=0.0,
        expectancy=0.0,
        feature_distributions=baseline_distributions,
        regime_trend=None,
    )
    engine = ValidationEngine(baseline, thresholds=thresholds)

    report = engine.validate(
        pd.Series(dtype=float),   # no live equity curve in the daily pipeline
        [],                       # no live trades
        current_features=current_features,
        current_regime_trend=None,
    )

    alerts = tuple(
        CapturedAlert(level=a.level.name, code=a.code, message=a.message)
        for a in report.alerts
    )
    # INFO-level alerts (e.g. the insufficient_data deferral) are informational
    # conditions, recorded as notes rather than failures (Req 4.3).
    notes = tuple(
        a.message for a in report.alerts if a.level == AlertLevel.INFO
    )

    return ValidationResult(alerts=alerts, notes=notes, passed=report.passed)


def _feature_distributions(
    vectors: Iterable["FeatureVector"],
) -> dict[str, np.ndarray]:
    """Collect per-feature value arrays from a stream of feature vectors.

    Aggregates every numeric feature value keyed by feature name into a
    ``dict[name -> np.ndarray]`` suitable for the engine's baseline/current
    distributions. Non-numeric values are skipped; NaN handling is left to the
    engine's own drift routines so no drift logic is duplicated here.
    """
    accumulated: dict[str, list[float]] = {}
    for fv in vectors:
        for name, value in fv.values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            accumulated.setdefault(name, []).append(numeric)
    return {
        name: np.asarray(values, dtype=float)
        for name, values in accumulated.items()
    }


def market_summary_stage(
    read_result: ReadResult,
    prior_eod_chain: OptionChain | None = None,
    prior_close: float | None = None,
) -> MarketSummary:
    """Compute the Target_Session headline metrics (Stage of Requirement 6).

    Produces a :class:`SessionMetrics` for the report's "What happened today"
    section -- NIFTY % change, India VIX % change, PCR, max pain, gamma sign
    (and total GEX), OI change, and ATM IV -- and wraps it in a
    :class:`MarketSummary`. Every metric is computed per the design's
    metric-source rules (Requirement 6), reusing ``analytics.options`` for all
    option-derived numbers so put/call ratio, max pain, ATM IV, and gamma
    exposure are never reimplemented here (Req 3.5, 16.3).

    Any metric that cannot be computed from the available data is recorded by
    name in ``MarketSummary.unavailable`` and carried as ``None`` on the
    metrics object -- it is never silently dropped (Req 6.5).

    Args:
        read_result: the Target_Session read output (full snapshots + the
            representative end-of-day snapshot + candles).
        prior_eod_chain: the prior Session's end-of-day chain, used only to
            compute the India VIX percentage change relative to the prior
            session; ``None`` when no prior session was collected.
        prior_close: the prior Session's closing NIFTY price, used as the
            reference for the session-relative NIFTY percentage change; ``None``
            falls back to the intraday ``(close - open) / open`` change.

    Returns:
        MarketSummary: the session metrics plus the list of metric names that
        could not be computed.
    """
    eod_chain = read_result.eod_chain
    unavailable: list[str] = []

    nifty_pct_change = _compute_nifty_pct_change(read_result.candles, prior_close)
    if nifty_pct_change is None:
        unavailable.append("nifty_pct_change")

    india_vix = _context_india_vix(eod_chain)
    if india_vix is None:
        unavailable.append("india_vix")

    india_vix_pct_change = _compute_india_vix_pct_change(india_vix, prior_eod_chain)
    if india_vix_pct_change is None:
        unavailable.append("india_vix_pct_change")

    pcr = _compute_pcr(eod_chain)
    if pcr is None:
        unavailable.append("pcr")

    max_pain = _compute_max_pain(eod_chain)
    if max_pain is None:
        unavailable.append("max_pain")

    gamma_sign, total_gex = _compute_gamma(eod_chain)
    if gamma_sign is None:
        unavailable.append("gamma_sign")

    oi_change = _compute_oi_change(eod_chain)
    if oi_change is None:
        unavailable.append("oi_change")

    atm_iv = _compute_atm_iv(eod_chain)
    if atm_iv is None:
        unavailable.append("atm_iv")

    metrics = SessionMetrics(
        session_id=read_result.target_session.isoformat(),
        nifty_pct_change=nifty_pct_change,
        india_vix=india_vix,
        india_vix_pct_change=india_vix_pct_change,
        pcr=pcr,
        max_pain=max_pain,
        gamma_sign=gamma_sign,
        total_gex=total_gex,
        oi_change=oi_change,
        atm_iv=atm_iv,
    )
    return MarketSummary(
        session_id=read_result.target_session.isoformat(),
        metrics=metrics,
        unavailable=unavailable,
    )


def _compute_nifty_pct_change(
    candles: OHLCVSeries,
    prior_close: float | None,
) -> float | None:
    """NIFTY % change for the session, or ``None`` when there are no candles.

    Session-relative ``(last_close - prior_session_close) / prior_session_close``
    when a ``prior_close`` is supplied; otherwise the intraday fallback
    ``(close - open) / open`` using the session's first open and last close.
    Unavailable only when the session has no candles (Req 6 metric table).
    """
    session_candles = candles.candles
    if not session_candles:
        return None

    last_close = session_candles[-1].close
    if prior_close is not None and prior_close != 0:
        return (last_close - prior_close) / prior_close

    first_open = session_candles[0].open
    if first_open == 0:
        return None
    return (last_close - first_open) / first_open


def _context_india_vix(chain: OptionChain) -> float | None:
    """India VIX from the chain ``context``, coerced to float, else ``None``."""
    value = chain.context.get("india_vix")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_india_vix_pct_change(
    india_vix: float | None,
    prior_eod_chain: OptionChain | None,
) -> float | None:
    """India VIX % change vs the prior session, or ``None`` when uncomputable.

    Requires the Target_Session India VIX (from context) and a prior-session
    India VIX reference; unavailable when ``india_vix`` is absent in either
    context or no prior session is available (Req 6 metric table).
    """
    if india_vix is None or prior_eod_chain is None:
        return None
    prior_vix = _context_india_vix(prior_eod_chain)
    if prior_vix is None or prior_vix == 0:
        return None
    return (india_vix - prior_vix) / prior_vix


def _compute_pcr(chain: OptionChain) -> float | None:
    """Put/call ratio by OI via ``analytics``; ``None`` on NaN / no quotes."""
    pcr = analytics.put_call_ratio(chain, by="oi")
    if pcr is None or np.isnan(pcr):
        return None
    return pcr


def _compute_max_pain(chain: OptionChain) -> float | None:
    """Max-pain strike via ``analytics``; ``None`` when the chain has no strikes."""
    try:
        return analytics.max_pain(chain)
    except ValueError:
        return None


def _compute_gamma(chain: OptionChain) -> tuple[int | None, float | None]:
    """Gamma sign and total GEX via ``analytics``; ``(None, None)`` on failure.

    Returns ``(sign(total_gex), total_gex)`` where the sign is ``-1``, ``0``, or
    ``+1``. Unavailable when the analytics computation raises (Req 6 metric
    table).
    """
    try:
        gex = analytics.gamma_exposure(chain)
    except (ValueError, ZeroDivisionError):
        return None, None
    total_gex = gex.total_gex
    return int(np.sign(total_gex)), total_gex


def _compute_oi_change(chain: OptionChain) -> float | None:
    """Aggregate OI change across the chain; ``None`` when there are no quotes."""
    if not chain.quotes:
        return None
    return float(sum(q.oi_change for q in chain.quotes))


def _compute_atm_iv(chain: OptionChain) -> float | None:
    """ATM implied vol via ``analytics``; ``None`` when it cannot be computed."""
    try:
        return analytics.atm_iv(chain)
    except ValueError:
        return None


# Numeric SessionMetrics fields compared for extremity unusual events, paired
# with the human-readable label used in the event description. ``gamma_sign`` is
# excluded here -- it is categorical and handled by the dedicated gamma-flip
# detector below. The order is fixed so emitted events are deterministic.
_EXTREMITY_METRICS: tuple[tuple[str, str], ...] = (
    ("nifty_pct_change", "NIFTY move"),
    ("india_vix", "India VIX level"),
    ("india_vix_pct_change", "India VIX change"),
    ("pcr", "put/call ratio"),
    ("max_pain", "max pain strike"),
    ("total_gex", "gamma exposure"),
    ("oi_change", "open-interest change"),
    ("atm_iv", "IV expansion"),
)


def comparison_stage(
    market_summary: MarketSummary | SessionMetrics,
    history: Sequence[SessionMetrics] = (),
    collected_session_count: int | None = None,
    *,
    min_sample_size: int = 20,
) -> ComparisonResult:
    """Compare the Target_Session against the Prior_Sessions actually collected.

    Stage 4 (Requirements 5, 9, 14). Builds a per-session series for each numeric
    headline metric over the collected Prior_Sessions plus the Target_Session,
    detects unusual events, and packages the result with the honest session
    counts so downstream rendering never overclaims.

    Two families of unusual events are detected:

    * **Extremity** -- for each metric whose Target_Session value is the strict
      maximum or strict minimum among the collected sessions that have a value
      for that metric, an :class:`UnusualEvent` is recorded naming the metric and
      the number of sessions spanned (e.g. "largest IV expansion in 18
      sessions"). ``sessions_spanned`` equals the count of sessions considered
      for that metric (Property 9). Detection requires at least one Prior_Session
      value to compare against, so the single-session case yields no extremity
      events (there is no historical comparison, Req 5.4).
    * **Gamma flip** -- a gamma-flip event is recorded iff the Target_Session
      ``gamma_sign`` differs from the immediately prior Session's ``gamma_sign``
      (Property 10). It is skipped when either sign is unavailable.

    Honest small-sample behaviour (Req 5.3, 9.4, 14.2, 14.3): every event's
    description is annotated with the actual number of sessions used, and while
    ``collected_session_count`` is below ``min_sample_size`` every event is
    marked ``significant=False`` so no statistical significance is ever claimed.

    The single-session case (no Prior_Sessions) completes gracefully without
    raising and sets ``history_available=False`` (Req 5.4, 14.1, 14.4).

    Args:
        market_summary: the Target_Session :class:`MarketSummary` (or its
            :class:`SessionMetrics` directly) whose values are compared.
        history: the collected Prior_Sessions as :class:`SessionMetrics`,
            ascending by ``session_id`` (the last element is the immediately
            prior Session). Empty for the single-session case.
        collected_session_count: ``Collected_Session_Count`` from the
            ``SessionCatalog`` (sessions at or before the target). Defaults to
            ``len(history) + 1`` when not supplied by the orchestrator.
        min_sample_size: minimum sessions before significance may be claimed.

    Returns:
        ComparisonResult: the session counts, detected unusual events,
        ``min_sample_size``, and the ``history_available`` flag.
    """
    target = (
        market_summary.metrics
        if isinstance(market_summary, MarketSummary)
        else market_summary
    )

    prior = list(history)
    prior_session_count = len(prior)
    if collected_session_count is None:
        collected_session_count = prior_session_count + 1
    history_available = prior_session_count > 0

    # Below the configured sample size no event may claim significance, while the
    # per-event session count keeps every statement annotated (Req 9.4, 14.3).
    significant = collected_session_count >= min_sample_size

    unusual_events: list[UnusualEvent] = []

    # --- Extremity events (Req 9.2 / Property 9) -----------------------------
    # Requires history to compare against; with no Prior_Sessions there is no
    # historical comparison to make (Req 5.4).
    if history_available:
        for attr, label in _EXTREMITY_METRICS:
            event = _detect_extremity(attr, label, target, prior, significant)
            if event is not None:
                unusual_events.append(event)

    # --- Gamma flip (Req 9.3 / Property 10) ----------------------------------
    gamma_flip = _detect_gamma_flip(target, prior, significant)
    if gamma_flip is not None:
        unusual_events.append(gamma_flip)

    return ComparisonResult(
        collected_session_count=collected_session_count,
        prior_session_count=prior_session_count,
        unusual_events=unusual_events,
        min_sample_size=min_sample_size,
        history_available=history_available,
    )


def _detect_extremity(
    attr: str,
    label: str,
    target: SessionMetrics,
    prior: Sequence[SessionMetrics],
    significant: bool,
) -> UnusualEvent | None:
    """Record an extremity :class:`UnusualEvent` for one metric, or ``None``.

    Builds the metric series over the Target_Session plus the Prior_Sessions that
    actually have a value for ``attr`` (``None`` values are skipped, never
    assumed). An event is produced only when the Target_Session value is the
    strict maximum or strict minimum among at least one other collected value;
    ``sessions_spanned`` is the count of sessions considered (Property 9).
    """
    target_value = getattr(target, attr)
    if target_value is None:
        return None

    prior_values = [
        getattr(m, attr) for m in prior if getattr(m, attr) is not None
    ]
    # Need at least one other session to establish a strict extreme.
    if not prior_values:
        return None

    sessions_spanned = len(prior_values) + 1

    if target_value > max(prior_values):
        description = f"largest {label} in {sessions_spanned} sessions"
    elif target_value < min(prior_values):
        description = f"smallest {label} in {sessions_spanned} sessions"
    else:
        return None

    return UnusualEvent(
        metric=attr,
        description=description,
        sessions_spanned=sessions_spanned,
        significant=significant,
    )


def _detect_gamma_flip(
    target: SessionMetrics,
    prior: Sequence[SessionMetrics],
    significant: bool,
) -> UnusualEvent | None:
    """Record a gamma-flip :class:`UnusualEvent` iff the gamma sign changed.

    Compares the Target_Session ``gamma_sign`` to the immediately prior Session's
    (the last element of ``prior``). Returns an event if and only if both signs
    are available and differ (Property 10); otherwise ``None``.
    """
    if not prior:
        return None
    target_sign = target.gamma_sign
    prior_sign = prior[-1].gamma_sign
    if target_sign is None or prior_sign is None:
        return None
    if target_sign == prior_sign:
        return None
    return UnusualEvent(
        metric="gamma_sign",
        description="gamma flip detected versus the prior session (2 sessions)",
        sessions_spanned=2,
        significant=significant,
    )


# Terminal lifecycle statuses: reaching either records expanded Experiment_Record
# provenance and triggers a ROADMAP refresh (Req 18.2, 23.1).
_TERMINAL_STATUSES = frozenset({"supported", "rejected"})


@dataclass(frozen=True)
class ExperimentProvenance:
    """Run-level version stamps folded into refreshed Experiment_Records (Req 23.1).

    The three version fields identify the inputs and code that produced a
    verdict. They are supplied by the orchestrator (``dataset_version`` from the
    Warehouse/``SessionCatalog`` state, ``feature_version`` from
    ``FeatureEngine.version``/``FeatureResult``, and ``code_version`` from the
    package/build) and default to empty strings so the stage stays independently
    testable with synthetic inputs (Req 17.3).
    """

    dataset_version: str = ""
    feature_version: str = ""
    code_version: str = ""


@dataclass(frozen=True)
class JournalResult:
    """Output of the journal & evidence stage (Stage 5, Requirements 7, 8, 11, 12, 23).

    Captures, for the Target_Session, the per-hypothesis Evidence_Score movements
    classified for the report's evidence sections:

    * ``gained``   -- hypotheses whose Evidence_Score increased (Req 7.2, 7.3).
    * ``weakened`` -- hypotheses whose Evidence_Score decreased (Req 8.2).
    * ``rejected`` -- hypotheses whose status transitioned to ``rejected`` this
      run (Req 8.3).

    Together with ``changes`` (every evaluated hypothesis, in journal order),
    ``added`` (new trackable hypotheses inserted with status ``open``, Req 12.3),
    and ``terminal_changes`` (hypotheses whose status moved to
    ``supported``/``rejected`` this run, the ROADMAP-refresh trigger of Req 18.2
    whose expanded Experiment_Record provenance was refreshed, Req 23.1).
    """

    target_session: str
    gained: list[EvidenceChange]
    weakened: list[EvidenceChange]
    rejected: list[EvidenceChange]
    changes: list[EvidenceChange]
    added: list["Hypothesis"]
    terminal_changes: list[EvidenceChange]


def journal_evidence_stage(
    journal: "ResearchJournal",
    evidence_engine: "EvidenceEngine",
    market_summary: MarketSummary,
    history: Sequence[SessionMetrics] = (),
    *,
    new_ideas: Sequence[ResearchIdea | str] = (),
    provenance: ExperimentProvenance | None = None,
) -> JournalResult:
    """Update the Research_Journal and Evidence_Scores for the Target_Session.

    Stage 5 (Requirements 7, 8, 11, 12, 18.2, 23). Loads every tracked hypothesis
    via ``journal.list()`` (Req 12.1) and, for each one, drives the **frozen**
    ``EvidenceEngine`` (the scoring logic is never reimplemented here):

    1. ``read_score`` decodes the prior Evidence_Score from the hypothesis.
    2. ``evaluate`` maps the hypothesis to a :class:`~...evidence.Verdict` against
       the Target_Session ``MarketSummary`` and the collected ``history``.
    3. ``apply`` produces the updated score, clamped to ``[0, 100]`` (Req 11.2-11.4).
    4. ``next_status`` maps the updated score to a lifecycle status drawn only
       from ``{open, testing, supported, rejected, inconclusive}`` (Req 11.5,
       11.6, 12.4).
    5. ``persist`` writes the score/status/reason back through the existing
       ``ResearchJournal.update`` interface (Req 11.7, 12.2).

    Each evaluation yields an :class:`EvidenceChange` carrying the prior and
    updated scores, the verdict, a human-readable observation of what supported
    or contradicted the hypothesis (Req 7.3), and the status before/after. The
    changes are classified into ``gained`` (score up, Req 7.2), ``weakened``
    (score down, Req 8.2), and ``rejected`` (status transitioned to ``rejected``,
    Req 8.3).

    New rule-based trackable ideas passed in ``new_ideas`` (either
    :class:`ResearchIdea` objects or plain hypothesis strings) are inserted with
    ``journal.add(..., status="open")`` (Req 12.3) and returned in ``added``.

    For every hypothesis whose status transitions to ``supported`` or
    ``rejected`` this run, the expanded **Experiment_Record** provenance is
    refreshed: an :class:`ExperimentRecord` is built (preserving any existing
    research question) and folded back into the hypothesis tags via
    :func:`~...experiment.merge_experiment_tags` (which encodes through
    :func:`~...experiment.encode_experiment`) and persisted through
    ``journal.update`` -- so the terminal verdict carries full provenance
    (Req 23.1) and the transition is surfaced in ``terminal_changes`` to trigger
    a ROADMAP refresh (Req 18.2). Existing non-experiment tags (e.g. the
    ``evidence:<NN>`` score tag written by ``persist``) are preserved.

    Args:
        journal: the Research_Journal (used through its frozen interface only).
        evidence_engine: the frozen Evidence_Score scoring/status engine.
        market_summary: the Target_Session :class:`MarketSummary`.
        history: the collected Prior_Sessions as :class:`SessionMetrics`,
            ascending by ``session_id``; may be empty.
        new_ideas: optional rule-based trackable ideas to add with status
            ``open`` (the dedicated idea-generation stage runs separately).
        provenance: run-level version stamps folded into refreshed
            Experiment_Records; defaults to empty stamps when not supplied.

    Returns:
        JournalResult: the classified evidence changes, all changes, the added
        hypotheses, and the terminal status transitions.
    """
    prov = provenance or ExperimentProvenance()

    hypotheses = journal.list()

    changes: list[EvidenceChange] = []
    gained: list[EvidenceChange] = []
    weakened: list[EvidenceChange] = []
    rejected: list[EvidenceChange] = []
    terminal_changes: list[EvidenceChange] = []

    for h in hypotheses:
        prior_score = evidence_engine.read_score(h)
        status_before = h.status

        verdict = evidence_engine.evaluate(h, market_summary, history)
        updated_score = evidence_engine.apply(prior_score, verdict)
        status_after = evidence_engine.next_status(updated_score, status_before)

        observation = _evidence_observation(
            h, verdict, prior_score, updated_score, market_summary.session_id
        )

        # Persist score + status + reason through the frozen journal interface.
        updated_h = evidence_engine.persist(
            journal, h, updated_score, status_after, observation
        )

        change = EvidenceChange(
            hypothesis_id=h.id,
            text=h.hypothesis,
            prior_score=prior_score,
            updated_score=updated_score,
            verdict=verdict,
            observation=observation,
            status_before=status_before,
            status_after=status_after,
        )
        changes.append(change)

        if updated_score > prior_score:
            gained.append(change)
        elif updated_score < prior_score:
            weakened.append(change)

        if status_after == "rejected" and status_before != "rejected":
            rejected.append(change)

        # A transition into a terminal status refreshes Experiment_Record
        # provenance and triggers a ROADMAP update (Req 18.2, 23.1).
        if status_after in _TERMINAL_STATUSES and status_after != status_before:
            _refresh_experiment_provenance(journal, updated_h, change, prov)
            terminal_changes.append(change)

    added = _add_new_ideas(journal, new_ideas)

    return JournalResult(
        target_session=market_summary.session_id,
        gained=gained,
        weakened=weakened,
        rejected=rejected,
        changes=changes,
        added=added,
        terminal_changes=terminal_changes,
    )


def _evidence_observation(
    h: "Hypothesis",
    verdict: "Verdict",
    prior_score: int,
    updated_score: int,
    session_id: str,
) -> str:
    """Build the human-readable observation recorded for an evidence change.

    States what the Target_Session evidence did to the hypothesis (supported,
    contradicted, or provided no evidence) and the Evidence_Score movement, so
    the report can explain each increase/decrease (Req 7.3, 8.2). When the
    hypothesis declares a ``metric:<field>`` tag, the metric name is appended for
    context.
    """
    phrase = {
        "supporting": "supported by",
        "contradicting": "contradicted by",
        "absent": "received no",
    }.get(verdict.value, "evaluated against")

    metric = _hypothesis_tag_value(h, "metric")
    metric_suffix = f" on metric '{metric}'" if metric else ""

    return (
        f"Hypothesis {phrase} Target_Session {session_id} evidence{metric_suffix} "
        f"(Evidence_Score {prior_score} -> {updated_score})."
    )


def _refresh_experiment_provenance(
    journal: "ResearchJournal",
    h: "Hypothesis",
    change: EvidenceChange,
    provenance: ExperimentProvenance,
) -> None:
    """Write/refresh the expanded Experiment_Record for a terminal hypothesis.

    Decodes any existing Experiment_Record (to preserve a previously recorded
    research question), fills the eight provenance fields (Req 23.1) from the
    terminal verdict and run-level version stamps, encodes them via
    :func:`~...experiment.merge_experiment_tags` (preserving non-experiment tags
    such as the ``evidence:<NN>`` score tag), and persists the merged tag list
    through the existing ``ResearchJournal.update`` interface (Req 23.2).
    """
    existing = decode_experiment(h)
    research_question = (
        existing.research_question
        or f"Does Target_Session evidence support the hypothesis: {h.hypothesis}?"
    )
    decision = change.status_after
    next_action = (
        "Promote the supported hypothesis to a strategy backtest."
        if decision == "supported"
        else "Retire the rejected hypothesis and stop tracking it."
    )

    record = ExperimentRecord(
        research_question=research_question,
        hypothesis=h.hypothesis,
        dataset_version=provenance.dataset_version,
        feature_version=provenance.feature_version,
        code_version=provenance.code_version,
        result=change.observation,
        decision=decision,
        next_action=next_action,
    )
    merged_tags = merge_experiment_tags(getattr(h, "tags", None), record)
    journal.update(h.id, tags=merged_tags)


def _add_new_ideas(
    journal: "ResearchJournal",
    new_ideas: Sequence[ResearchIdea | str],
) -> list["Hypothesis"]:
    """Add new trackable rule-based ideas as ``open`` hypotheses (Req 12.3).

    Accepts either :class:`ResearchIdea` objects (whose ``text`` is the
    hypothesis statement) or plain strings, inserting each through the existing
    ``ResearchJournal.add`` interface with status ``open``. Blank ideas are
    skipped. Returns the created hypotheses in input order.
    """
    added: list["Hypothesis"] = []
    for idea in new_ideas:
        text = idea.text if isinstance(idea, ResearchIdea) else str(idea)
        text = text.strip()
        if not text:
            continue
        added.append(
            journal.add(
                text,
                status="open",
                reason="rule-based trackable idea added by the daily research pipeline",
            )
        )
    return added


def _hypothesis_tag_value(h: "Hypothesis", key: str) -> str | None:
    """Return the value of the first ``<key>:<value>`` tag on a hypothesis."""
    prefix = f"{key}:"
    for tag in getattr(h, "tags", None) or []:
        text = str(tag)
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return None
