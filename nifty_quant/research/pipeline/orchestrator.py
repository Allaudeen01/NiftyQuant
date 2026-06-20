"""PipelineOrchestrator -- the staged daily-research-pipeline control loop.

This module wires the per-stage logic (``stages.py``), the pure scoring modules
(``quality.py``, ``evidence.py``, ``confidence.py``), the catalog, the idea
generator, and the renderers into a single deterministic run. It runs the eight
stages in the fixed order

    quality-gate -> read -> features -> validation -> comparison -> evidence
    -> ideas -> report

threading a mutable :class:`PipelineContext` between them and routing every
executed stage through :meth:`PipelineOrchestrator._run_stage`, which records
exactly one :class:`~nifty_quant.research.pipeline.cost.StageCost` per executed
stage regardless of outcome (Req 22.2).

Error handling mirrors the design's two failure classes:

* **Quality gate FAIL** (Req 19.4) is a distinguished unrecoverable condition:
  the orchestrator halts *before* the read stage, records every failing
  ``QualityCheckResult`` in ``error_detail``, sets ``failing_stage="quality_gate"``,
  and returns a non-zero exit code -- no read/feature/downstream work runs.
* **Unrecoverable stage error** -- a :class:`~...stages.StageError` (or any
  unexpected exception) escaping a stage -- is caught at the orchestration
  boundary, recorded with its ``failing_stage``/``error_detail`` and a non-zero
  exit code.
* **Recoverable conditions** -- validation insufficient-data, an uncomputable
  market metric, an AI fallback -- are recorded as informational notes and the
  run still exits ``0``.

Dependencies (storage, feature store, journal, idea generator) are injected so
tests can substitute in-memory/synthetic implementations (Req 17.2, 17.3). With
AI disabled by default (``PipelineConfig.use_ai = False``) the whole run is
deterministic (Req 15.1, 17.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from nifty_quant import __version__ as _PACKAGE_VERSION
from nifty_quant.features.engine import FeatureEngine
from nifty_quant.research.pipeline import report as report_renderer
from nifty_quant.research.pipeline.catalog import SessionCatalog
from nifty_quant.research.pipeline.confidence import (
    ConfidenceConfig,
    ConfidenceResult,
    compute_confidence,
)
from nifty_quant.research.pipeline.cost import StageCost, run_with_cost
from nifty_quant.research.pipeline.evidence import EvidenceConfig, EvidenceEngine
from nifty_quant.research.pipeline.ideas import estimate_information_gain, prioritize
from nifty_quant.research.pipeline.models import (
    ComparisonResult,
    DailyReportModel,
    MarketRegime,
    MarketSummary,
    ResearchIdea,
    SessionMetrics,
)
from nifty_quant.research.pipeline.quality import (
    QualityConfig,
    QualityReport,
    run_quality_checks,
    score_quality,
)
from nifty_quant.research.pipeline.regime_view import market_regime
from nifty_quant.research.pipeline.roadmap import RoadmapModel, RoadmapWriter
from nifty_quant.research.pipeline.stages import (
    ExperimentProvenance,
    FeatureResult,
    JournalResult,
    ReadResult,
    StageError,
    ValidationResult,
    comparison_stage,
    feature_stage,
    journal_evidence_stage,
    market_summary_stage,
    read_stage,
    validation_stage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from nifty_quant.data.models import OptionChain
    from nifty_quant.data.storage.base import Storage
    from nifty_quant.features.store import FeatureStore
    from nifty_quant.research.journal import ResearchJournal
    from nifty_quant.research.pipeline.ideas import IdeaGenerator


# Day window bounds for a Target_Session read, matching the read stage's window
# ``[00:00:00, 23:59:59]`` so the quality gate inspects the same snapshots.
_DAY_START = time(0, 0, 0)
_DAY_END = time(23, 59, 59)

# Static control-center context rendered into ROADMAP.md (Req 18.1). The daily
# pipeline orchestrates -- and never modifies -- these frozen ``nifty_quant``
# components.
_ARCHITECTURE_VERSION = f"nifty_quant v{_PACKAGE_VERSION} (daily-research-pipeline)"
_FROZEN_COMPONENTS = [
    "nifty_quant.data.storage (Storage / ParquetStorage)",
    "nifty_quant.features.engine.FeatureEngine",
    "nifty_quant.features.store (FeatureStore / ParquetFeatureStore)",
    "nifty_quant.analytics.options",
    "nifty_quant.validation.engine.ValidationEngine",
    "nifty_quant.research.journal.ResearchJournal",
    "nifty_quant.research.regime.classify_regime",
]

# Normalised severity (0.0 best .. 1.0 worst) for each validation alert level,
# used to derive the confidence score's drift/validation factors (Req 21.2).
_ALERT_SEVERITY = {"INFO": 0.0, "WARNING": 0.5, "CRITICAL": 1.0}


@dataclass
class PipelineConfig:
    """Run configuration for the daily research pipeline.

    All fields carry deterministic defaults so a bare ``PipelineConfig()`` runs
    the most-recent collected session for ``NIFTY`` at ``5m`` with AI disabled.
    The three nested config objects use ``default_factory`` (rather than a shared
    instance) so each ``PipelineConfig`` gets its own copy.
    """

    underlying: str = "NIFTY"
    timeframe: str = "5m"
    data_dir: str = "data"
    feature_dir: str = "data"
    report_dir: str = "reports"
    journal_path: str = "reports/research_journal.jsonl"
    roadmap_path: str = "ROADMAP.md"
    session: date | None = None
    use_ai: bool = False  # AI disabled by default; rule-based is primary (Req 15.1)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    min_sample_size: int = 20
    quality: QualityConfig = field(default_factory=QualityConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)


@dataclass
class PipelineResult:
    """Outcome of one pipeline run, returned by :meth:`PipelineOrchestrator.run`.

    ``exit_code`` is ``0`` on success (including runs with only recoverable
    notes) and non-zero on an unrecoverable failure -- a quality-gate FAIL or a
    stage error. ``failing_stage``/``error_detail`` identify the failure;
    ``quality_score``/``confidence_score`` carry the run's scores when known; and
    ``stage_costs`` holds one :class:`StageCost` per executed stage (Req 22.2).
    """

    target_session: date
    collected_session_count: int
    report_path: Path
    exit_code: int
    failing_stage: str | None
    error_detail: str | None
    quality_score: int | None       # Quality_Score for the run (Req 19.5)
    confidence_score: int | None    # Research_Confidence_Score (Req 21.1)
    stage_costs: list[StageCost]    # per-stage computational cost (Req 22.2)


@dataclass
class PipelineContext:
    """Mutable state threaded between stages during a single run.

    Accumulates the per-stage outputs, the running ``stage_costs`` list (one
    entry per executed stage, Req 22.2), and the informational ``notes`` that
    recoverable conditions append (Req 4.3). Every field has a safe default so a
    failure part-way through still yields a usable context for the error result.
    """

    target_session: date
    collected_session_count: int = 0
    stage_costs: list[StageCost] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    quality_report: QualityReport | None = None
    read_result: ReadResult | None = None
    feature_result: FeatureResult | None = None
    validation_result: ValidationResult | None = None
    market_summary: MarketSummary | None = None
    comparison: ComparisonResult | None = None
    journal_result: JournalResult | None = None
    confidence_result: ConfidenceResult | None = None
    history: list[SessionMetrics] = field(default_factory=list)
    ideas: list[ResearchIdea] = field(default_factory=list)


class PipelineOrchestrator:
    """Runs the eight pipeline stages in fixed order, threading a context.

    Dependencies are injected (Req 17.2, 17.3): ``storage`` (the frozen
    Warehouse), ``feature_store`` (versioned feature persistence), ``journal``
    (the Research_Journal), and ``idea_generator`` (rule-based by default).
    """

    def __init__(
        self,
        config: PipelineConfig,
        storage: "Storage",
        feature_store: "FeatureStore",
        journal: "ResearchJournal",
        idea_generator: "IdeaGenerator",
    ) -> None:
        self.config = config
        self._storage = storage
        self._feature_store = feature_store
        self._journal = journal
        self._idea_generator = idea_generator
        # Set per run; the most recently entered stage name, used to attribute an
        # unexpected (non-StageError) exception to the stage that raised it.
        self._current_stage: str | None = None
        self._context: PipelineContext | None = None

    # --- Public API ----------------------------------------------------------

    def run(self) -> PipelineResult:
        """Execute the pipeline for the Target_Session and return its result.

        Resolves the Target_Session, runs the Data Quality Gate first and halts
        on a FAIL verdict (Req 19.1, 19.4), then threads read -> features ->
        validation -> comparison -> evidence -> ideas -> report. Every executed
        stage records one ``StageCost`` (Req 22.2). Returns a populated
        :class:`PipelineResult`; never raises for an expected failure class.
        """
        config = self.config
        catalog = SessionCatalog(self._storage, config.underlying, config.timeframe)
        target = self._resolve_target(catalog)

        # No explicit session and nothing collected: unrecoverable, nothing to do.
        if target is None:
            return PipelineResult(
                target_session=date.min,
                collected_session_count=0,
                report_path=Path(report_renderer.report_path("none", config.report_dir)),
                exit_code=1,
                failing_stage="read",
                error_detail="no collected sessions found in the warehouse",
                quality_score=None,
                confidence_score=None,
                stage_costs=[],
            )

        ctx = PipelineContext(target_session=target)
        ctx.collected_session_count = catalog.collected_count(target)
        self._context = ctx
        self._current_stage = None

        report_path = Path(
            report_renderer.report_path(target.isoformat(), config.report_dir)
        )

        try:
            # --- Stage 0: Data Quality Gate (runs before read, Req 19.1) ------
            quality_report = self._run_stage(
                "quality_gate",
                self._quality_gate,
                target,
                rows=lambda r: len(r.checks),
            )
            ctx.quality_report = quality_report
            if quality_report.verdict != "PASS":
                # Halt before the read stage on a FAIL verdict (Req 19.4).
                return self._quality_fail_result(target, quality_report, report_path)

            # --- Stage 1: Read (Req 2) ----------------------------------------
            read_result = self._run_stage(
                "read",
                read_stage,
                self._storage,
                config.underlying,
                config.timeframe,
                target,
                rows=lambda r: len(r.chains) + len(r.candles.candles),
            )
            ctx.read_result = read_result

            # Build the collected Prior_Session history (and the immediately
            # prior session's reference EOD chain / close) by reading each prior
            # session and computing its headline metrics.
            prior_sessions = catalog.prior_sessions(target)
            history, prior_eod_chain, prior_close = self._build_history(prior_sessions)
            ctx.history = history

            # --- Stage 2: Features (Req 3) ------------------------------------
            feature_engine = FeatureEngine()
            feature_result = self._run_stage(
                "features",
                feature_stage,
                feature_engine,
                self._feature_store,
                read_result,
                rows=lambda r: len(r.vectors),
            )
            ctx.feature_result = feature_result

            # --- Stage 3: Validation (Req 4) ----------------------------------
            validation_result = self._run_stage(
                "validation",
                validation_stage,
                feature_result,
                rows=lambda r: len(r.alerts),
            )
            ctx.validation_result = validation_result
            # Insufficient-data / informational conditions are recoverable notes.
            ctx.notes.extend(validation_result.notes)

            # Target_Session headline metrics (Report Section 1, Req 6). Computed
            # here as input to comparison/evidence/ideas/regime.
            self._current_stage = "comparison"
            market_summary = market_summary_stage(
                read_result,
                prior_eod_chain=prior_eod_chain,
                prior_close=prior_close,
            )
            ctx.market_summary = market_summary
            # An uncomputable metric is a recoverable condition: note it, never
            # drop it (Req 6.5).
            for name in market_summary.unavailable:
                ctx.notes.append(f"Market metric unavailable: {name}")

            # --- Stage 4: Comparison (Req 5, 9, 14) ---------------------------
            comparison = self._run_stage(
                "comparison",
                comparison_stage,
                market_summary,
                history,
                ctx.collected_session_count,
                rows=lambda r: r.prior_session_count,
                min_sample_size=config.min_sample_size,
            )
            ctx.comparison = comparison

            # --- Stage 5: Journal & Evidence (Req 7, 8, 11, 12, 18.2, 23) -----
            # The journal flush happens entirely inside this stage; later-stage
            # failures cannot corrupt it because evidence is persisted here.
            provenance = ExperimentProvenance(
                dataset_version=f"{config.underlying}:{ctx.collected_session_count} sessions",
                feature_version=feature_result.feature_version,
                code_version=_PACKAGE_VERSION,
            )
            journal_result = self._run_stage(
                "journal_evidence",
                journal_evidence_stage,
                self._journal,
                EvidenceEngine(config.evidence),
                market_summary,
                history,
                rows=lambda r: len(r.changes),
                provenance=provenance,
            )
            ctx.journal_result = journal_result

            # --- Stage 6: Ideas (Req 10, 15) ----------------------------------
            prioritized_ideas = self._run_stage(
                "ideas",
                self._build_ideas,
                market_summary,
                comparison,
                history,
                rows=lambda r: len(r),
            )
            ctx.ideas = prioritized_ideas

            # --- Cross-cutting: Market Regime (Req 20) + Confidence (Req 21) --
            regime = self._market_regime(read_result, market_summary)
            confidence = self._compute_confidence(
                quality_report, ctx.collected_session_count, validation_result, journal_result
            )
            ctx.confidence_result = confidence

            # Assemble the report model from everything computed so far. The
            # stage-cost table snapshots the costs captured before the report
            # stage (the report stage cannot time itself).
            model = DailyReportModel(
                target_session=target.isoformat(),
                market_regime=regime,
                confidence=confidence,
                quality=quality_report,
                market_summary=market_summary,
                gained=journal_result.gained,
                weakened=journal_result.weakened,
                rejected=journal_result.rejected,
                unusual_events=comparison.unusual_events,
                ideas=prioritized_ideas,
                validation_notes=list(ctx.notes),
                comparison=comparison,
                stage_costs=list(ctx.stage_costs),
            )

            # --- Stage 7: Report + ROADMAP (Req 13, 18, 19.5, 20, 21, 22.3) ---
            self._run_stage(
                "report",
                self._write_outputs,
                model,
                str(report_path),
                catalog,
                target,
                prioritized_ideas,
                quality_report,
            )

            return PipelineResult(
                target_session=target,
                collected_session_count=ctx.collected_session_count,
                report_path=report_path,
                exit_code=0,
                failing_stage=None,
                error_detail=None,
                quality_score=quality_report.score,
                confidence_score=confidence.score,
                stage_costs=list(ctx.stage_costs),
            )

        except StageError as exc:
            # A stage signalled an unrecoverable condition (Req 1.5).
            return self._error_result(
                target, report_path, exc.stage_name, exc.detail
            )
        except Exception as exc:  # noqa: BLE001 - boundary: convert to exit code
            # An unexpected exception escaped a stage; attribute it to the stage
            # that was running and surface a non-zero exit code.
            return self._error_result(
                target, report_path, self._current_stage or "unknown", str(exc)
            )

    # --- Stage cost wrapper --------------------------------------------------

    def _run_stage(
        self,
        stage_name: str,
        fn: Callable[..., Any],
        *args: Any,
        rows: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a stage through :func:`run_with_cost`, recording one StageCost.

        On success the ``StageCost`` returned by ``run_with_cost`` is appended to
        the context. If the stage raises, ``run_with_cost`` re-raises before it
        can return its cost, so a placeholder ``StageCost`` is appended here
        instead -- guaranteeing every executed stage contributes exactly one
        ``StageCost`` regardless of outcome (Req 22.2, Property 25).
        """
        assert self._context is not None  # set at the start of run()
        self._current_stage = stage_name
        try:
            result, cost = run_with_cost(stage_name, fn, *args, rows=rows, **kwargs)
        except Exception:
            self._context.stage_costs.append(
                StageCost(
                    stage_name=stage_name,
                    elapsed_seconds=0.0,
                    peak_memory_bytes=0,
                    rows_processed=0,
                )
            )
            raise
        self._context.stage_costs.append(cost)
        return result

    # --- Target-session resolution -------------------------------------------

    def _resolve_target(self, catalog: SessionCatalog) -> date | None:
        """Resolve the Target_Session: explicit ``--session`` or most recent.

        Returns ``config.session`` when supplied (Req 1.3); otherwise the most
        recent collected session (Req 1.2), or ``None`` when nothing has been
        collected and no explicit session was given.
        """
        if self.config.session is not None:
            return self.config.session
        sessions = catalog.sessions_up_to(date.max)
        return sessions[-1] if sessions else None

    # --- Stage 0 helper: the Data Quality Gate -------------------------------

    def _quality_gate(self, target: date) -> QualityReport:
        """Read the Target_Session inputs and score the Data Quality Gate.

        Reads the session's option chains and candles through the existing
        ``Storage`` interface (no new backend, Req 16.1), runs the fixed battery
        of ``Quality_Check`` functions, and aggregates them into a
        ``QualityReport`` (Req 19.2, 19.3).
        """
        start = datetime.combine(target, _DAY_START)
        end = datetime.combine(target, _DAY_END)
        chains = list(
            self._storage.read_option_chains(self.config.underlying, start, end)
        )
        candle_series = self._storage.read_candles(
            self.config.underlying, self.config.timeframe, start, end
        )
        candles = list(candle_series.candles)
        checks = run_quality_checks(chains, candles, target, self.config.quality)
        return score_quality(checks, self.config.quality, session_id=target.isoformat())

    # --- History construction -------------------------------------------------

    def _build_history(
        self, prior_sessions: "list[date]"
    ) -> "tuple[list[SessionMetrics], OptionChain | None, float | None]":
        """Build the collected Prior_Session metric history (ascending).

        Reads each Prior_Session through the read stage and computes its
        :class:`SessionMetrics` via ``market_summary_stage``, threading the
        previous session's EOD chain / close so each session's VIX/NIFTY change
        is session-relative. Returns the history plus the immediately prior
        session's reference EOD chain and close (used for the Target_Session's
        own session-relative changes). Sessions whose read yields no chains are
        skipped rather than failing the run.
        """
        history: list[SessionMetrics] = []
        prev_eod_chain: "OptionChain | None" = None
        prev_close: float | None = None

        for session in prior_sessions:  # ascending by date
            try:
                read_result = read_stage(
                    self._storage,
                    self.config.underlying,
                    self.config.timeframe,
                    session,
                )
            except StageError:
                # The catalog is derived from option chains, so this is unlikely;
                # skip defensively so a sparse prior session never aborts the run.
                continue

            summary = market_summary_stage(
                read_result,
                prior_eod_chain=prev_eod_chain,
                prior_close=prev_close,
            )
            history.append(summary.metrics)

            prev_eod_chain = read_result.eod_chain
            candles = read_result.candles.candles
            if candles:
                prev_close = candles[-1].close

        return history, prev_eod_chain, prev_close

    # --- Stage 6 helper: idea generation + prioritization --------------------

    def _build_ideas(
        self,
        market_summary: MarketSummary,
        comparison: ComparisonResult,
        history: "list[SessionMetrics]",
    ) -> list[ResearchIdea]:
        """Generate, score, and rank Research_Ideas for the report (Req 10).

        Generates ideas via the injected generator (rule-based by default; the
        AI wrapper falls back internally, Req 15.4), attaches each idea's
        ``Expected_Information_Gain`` (Req 10.3), and assigns ``Priority`` 1..N
        in descending-gain order via ``prioritize`` (Req 10.4).
        """
        raw_ideas = self._idea_generator.generate(market_summary, comparison, history)
        scored = [
            replace(
                idea,
                information_gain=estimate_information_gain(idea, comparison, history),
            )
            for idea in raw_ideas
        ]
        return prioritize(scored)

    # --- Cross-cutting: regime + confidence ----------------------------------

    def _market_regime(
        self, read_result: ReadResult, market_summary: MarketSummary
    ) -> MarketRegime:
        """Build the report's Market Regime view (Req 20).

        Supplies the Target_Session intraday close series to the regime adapter
        (which reuses the frozen ``classify_regime``); an empty series leaves the
        trend/volatility classifications ``unavailable`` (Req 20.4).
        """
        frame = read_result.candles.to_frame()
        close_series = frame["close"] if "close" in frame.columns else frame
        return market_regime(close_series, market_summary)

    def _compute_confidence(
        self,
        quality_report: QualityReport,
        collected_session_count: int,
        validation_result: ValidationResult,
        journal_result: JournalResult,
    ) -> ConfidenceResult:
        """Compute the Research_Confidence_Score after validation/evidence (Req 21).

        Derives the drift/validation severities from the captured validation
        alerts (drift severity from drift-coded alerts; alert severity from all
        alerts) and the evidence maturity from the mean updated Evidence_Score of
        the evaluated hypotheses, then feeds the deterministic weighted average.
        """
        drift_alerts = [a for a in validation_result.alerts if "drift" in a.code]
        drift_severity = self._alert_severity(drift_alerts)
        alert_severity = self._alert_severity(validation_result.alerts)
        evidence_maturity = self._evidence_maturity(journal_result)

        return compute_confidence(
            quality_score=quality_report.score,
            collected_session_count=collected_session_count,
            min_sample_size=self.config.min_sample_size,
            drift_severity=drift_severity,
            alert_severity=alert_severity,
            evidence_maturity=evidence_maturity,
            config=self.config.confidence,
        )

    @staticmethod
    def _alert_severity(alerts: "object") -> float:
        """Highest normalised severity (0.0..1.0) across the given alerts."""
        return max(
            (_ALERT_SEVERITY.get(a.level, 0.0) for a in alerts),
            default=0.0,
        )

    @staticmethod
    def _evidence_maturity(journal_result: JournalResult) -> float:
        """Mean updated Evidence_Score (0..100) across evaluated hypotheses.

        A higher mean means the evidence base is more decided / mature; an empty
        journal yields ``0`` (no maturity yet).
        """
        changes = journal_result.changes
        if not changes:
            return 0.0
        return sum(c.updated_score for c in changes) / len(changes)

    # --- Stage 7 helper: write report + ROADMAP ------------------------------

    def _write_outputs(
        self,
        model: DailyReportModel,
        report_path: str,
        catalog: SessionCatalog,
        target: date,
        prioritized_ideas: list[ResearchIdea],
        quality_report: QualityReport,
    ) -> str:
        """Render and write the Daily_Research_Report and refresh ROADMAP.md.

        Writes ``reports/research_<session>.md`` (overwriting any existing file
        for the session, Req 13.3) and rewrites the single root ``ROADMAP.md``
        control center from the current journal + warehouse + quality state
        (Req 18). Returns the report path written.
        """
        report_text = report_renderer.render(model)
        report_renderer.write(report_path, report_text)

        roadmap_model = RoadmapModel(
            architecture_version=_ARCHITECTURE_VERSION,
            frozen_components=list(_FROZEN_COMPONENTS),
            current_dataset=f"{self.config.underlying} {self.config.timeframe}",
            collected_sessions=catalog.collected_count(target),
            collected_option_chains=catalog.option_chain_count(target),
            collected_trading_days=catalog.trading_day_count(target),
            quality_score=quality_report.score,
            quality_verdict=quality_report.verdict,
            hypotheses=self._journal.list(),
            research_priorities=prioritized_ideas,
            next_milestones=[],
        )
        writer = RoadmapWriter()
        writer.write(self.config.roadmap_path, writer.render(roadmap_model))

        return report_path

    # --- Failure-result builders ---------------------------------------------

    def _quality_fail_result(
        self, target: date, quality_report: QualityReport, report_path: Path
    ) -> PipelineResult:
        """Build the halt-before-read result for a quality-gate FAIL (Req 19.4)."""
        failing = quality_report.failing_checks
        detail = "; ".join(f"{c.name}: {c.detail}" for c in failing) or (
            f"quality gate FAIL (score {quality_report.score})"
        )
        ctx = self._context
        return PipelineResult(
            target_session=target,
            collected_session_count=ctx.collected_session_count if ctx else 0,
            report_path=report_path,
            exit_code=1,
            failing_stage="quality_gate",
            error_detail=detail,
            quality_score=quality_report.score,
            confidence_score=None,
            stage_costs=list(ctx.stage_costs) if ctx else [],
        )

    def _error_result(
        self,
        target: date,
        report_path: Path,
        failing_stage: str,
        error_detail: str,
    ) -> PipelineResult:
        """Build the result for an unrecoverable stage error (Req 1.5)."""
        ctx = self._context
        quality_score = (
            ctx.quality_report.score if ctx and ctx.quality_report else None
        )
        return PipelineResult(
            target_session=target,
            collected_session_count=ctx.collected_session_count if ctx else 0,
            report_path=report_path,
            exit_code=1,
            failing_stage=failing_stage,
            error_detail=error_detail,
            quality_score=quality_score,
            confidence_score=None,
            stage_costs=list(ctx.stage_costs) if ctx else [],
        )
