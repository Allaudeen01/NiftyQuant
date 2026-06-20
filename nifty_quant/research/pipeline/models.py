"""Pipeline data models.

Plain, frozen dataclasses that carry data between the daily-research-pipeline
stages and into the report renderer. They mirror the existing ``nifty_quant``
dataclass style (``from __future__ import annotations`` + frozen dataclasses)
and hold no behaviour beyond being typed value objects.

Some fields reference types that live in sibling feature modules created by
later tasks -- ``Verdict`` (``evidence.py``), ``QualityReport`` /
``QualityCheckResult`` (``quality.py``), ``ConfidenceResult`` /
``ConfidenceBreakdown`` (``confidence.py``), and ``StageCost`` (``cost.py``).
Because ``from __future__ import annotations`` makes every annotation a lazy
string, this module imports cleanly before those modules exist; the
``TYPE_CHECKING`` block below lets static type checkers resolve them without
creating an import cycle at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from nifty_quant.research.pipeline.confidence import ConfidenceResult
    from nifty_quant.research.pipeline.cost import StageCost
    from nifty_quant.research.pipeline.evidence import Verdict
    from nifty_quant.research.pipeline.quality import QualityReport


@dataclass(frozen=True)
class SessionMetrics:
    """Per-session headline metrics used for comparison/unusual events."""
    session_id: str            # ISO date
    nifty_pct_change: float | None
    india_vix: float | None
    india_vix_pct_change: float | None
    pcr: float | None
    max_pain: float | None
    gamma_sign: int | None     # -1, 0, +1 (sign of total_gex)
    total_gex: float | None
    oi_change: float | None
    atm_iv: float | None


@dataclass(frozen=True)
class MarketSummary:
    """Target_Session metrics for report Section 1 (Req 6)."""
    session_id: str
    metrics: SessionMetrics
    unavailable: list[str]     # names of metrics that could not be computed


@dataclass(frozen=True)
class EvidenceChange:
    hypothesis_id: int
    text: str
    prior_score: int
    updated_score: int
    verdict: "Verdict"
    observation: str           # what supported/contradicted (Req 7.3)
    status_before: str
    status_after: str


@dataclass(frozen=True)
class UnusualEvent:
    metric: str
    description: str           # e.g. "largest IV expansion in 18 sessions"
    sessions_spanned: int
    significant: bool          # False while below min_sample_size (Req 9.4/14.3)


@dataclass(frozen=True)
class ResearchIdea:
    text: str                  # phrased as an investigation (Req 10.6)
    occurrence_count: int | None
    total_sessions: int
    information_gain: float = 0.0   # Expected_Information_Gain estimate (Req 10.3)
    priority: int | None = None     # 1..N, assigned after ranking; 1 = highest gain (Req 10.4)


@dataclass(frozen=True)
class MarketRegime:
    """Report Section 0 classification (Req 20). Any field None => render 'unavailable'."""
    trend: str | None          # "UP" | "DOWN" | "SIDEWAYS" (from classify_regime)
    volatility: str | None     # "HIGH" | "LOW" (from classify_regime)
    gamma_regime: str | None   # "bullish" | "neutral" | "bearish" (from gamma sign)
    pcr_level: str | None      # "low" | "neutral" | "high"


@dataclass(frozen=True)
class ExperimentRecord:
    """Expanded provenance for a unit of research work (Req 23), encoded into the
    existing Hypothesis via structured tags/fields -- no separate store (Req 23.2)."""
    research_question: str
    hypothesis: str
    dataset_version: str
    feature_version: str
    code_version: str
    result: str
    decision: str
    next_action: str


@dataclass(frozen=True)
class ComparisonResult:
    collected_session_count: int
    prior_session_count: int
    unusual_events: list[UnusualEvent]
    min_sample_size: int
    history_available: bool    # False when target is the only session (Req 5.4)


@dataclass(frozen=True)
class DailyReportModel:
    """Everything needed to render the report (regime first, then the five sections)."""
    target_session: str
    market_regime: MarketRegime          # Section 0, rendered first (Req 20.1)
    confidence: "ConfidenceResult"       # Overall Research Confidence Score (Req 21)
    quality: "QualityReport"             # Quality_Score + verdict (Req 19.5)
    market_summary: MarketSummary
    gained: list[EvidenceChange]
    weakened: list[EvidenceChange]
    rejected: list[EvidenceChange]
    unusual_events: list[UnusualEvent]
    ideas: list[ResearchIdea]            # prioritized, descending information gain (Req 10.3)
    validation_notes: list[str]
    comparison: ComparisonResult
    stage_costs: list["StageCost"]       # per-stage computational cost (Req 22.3)
