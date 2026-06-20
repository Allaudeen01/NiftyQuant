"""Daily Research Report renderer (Stage 7) -- pure markdown rendering + thin I/O.

This module turns a fully-assembled :class:`DailyReportModel` into a single,
self-contained markdown document (Req 13.1, 13.4) and writes it to disk under
``reports/`` (Req 13.1). Rendering is a **pure function** of the model -- it
performs no I/O, consults no clock, and reads no global state -- so it is
trivially unit-testable and deterministic. The only I/O lives in the thin
:func:`write` helper.

Report layout (in order):

#. **Market Regime** -- presented first, above "What happened today" (Req 20.1),
   stating the trend, volatility, gamma, and PCR classifications; any field that
   could not be computed is rendered as ``unavailable`` (Req 20.4).
#. **Overall Research Confidence Score** -- the single 0..100 score together with
   the contributing factors that determined it (Req 21.1, 21.3).
#. **Data Quality** -- the ``Quality_Score`` and PASS/FAIL verdict (Req 19.5).
#. The five required headed sections (Req 6.1, 7.1, 8.1, 9.1, 10.1):
   "What happened today", "Hypotheses that gained evidence", "Hypotheses that
   weakened", "Unusual events", and "What to investigate tomorrow". Market
   metrics that could not be computed are marked ``unavailable`` rather than
   dropped (Req 6.5); empty evidence/event/idea sections render an explicit
   "nothing here" message (Req 7.4, 8.4, 9.5, 10.7).
#. **Validation notes** -- informational notes captured during validation.
#. **Computational Cost** -- a per-stage table of time/memory/rows (Req 22.3).

The report file is named ``reports/research_<session>.md`` (Req 13.2) and
:func:`write` overwrites any existing file for that session (Req 13.3).
"""

from __future__ import annotations

import os
from typing import Sequence

from nifty_quant.research.pipeline.models import (
    ComparisonResult,
    DailyReportModel,
    EvidenceChange,
    MarketRegime,
    MarketSummary,
    ResearchIdea,
    UnusualEvent,
)

# Rendered placeholder for any value/classification that could not be computed
# (Req 6.5, 20.4). Defined once so the marker is consistent across the report.
UNAVAILABLE = "unavailable"

# Default directory the report is written under (Req 13.1).
REPORT_DIR = "reports"


# --- Filename / path helper --------------------------------------------------


def report_filename(session: str) -> str:
    """Filename identifying the Target_Session, ``research_<session>.md`` (Req 13.2)."""
    return f"research_{session}.md"


def report_path(session: str, report_dir: str = REPORT_DIR) -> str:
    """Full path ``<report_dir>/research_<session>.md`` for the Target_Session (Req 13.2)."""
    return os.path.join(report_dir, report_filename(session))


# --- Small formatting helpers (pure) -----------------------------------------


def _fmt_number(value: float | int | None, *, suffix: str = "", digits: int = 2) -> str:
    """Format a numeric value, rendering ``None`` as ``unavailable`` (Req 6.5)."""
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _fmt_pct(value: float | None) -> str:
    """Format a percentage value with a sign, ``None`` => ``unavailable``."""
    if value is None:
        return UNAVAILABLE
    return f"{value:+.2f}%"


def _fmt_gamma_sign(sign: int | None) -> str:
    """Render the gamma-exposure sign (-1/0/+1) as a readable label (Req 6.4)."""
    if sign is None:
        return UNAVAILABLE
    if sign > 0:
        return "positive (+1)"
    if sign < 0:
        return "negative (-1)"
    return "neutral (0)"


def _label_or_unavailable(value: str | None) -> str:
    """Render a classification label, ``None`` => ``unavailable`` (Req 20.4)."""
    return value if value else UNAVAILABLE


def _metric_value(summary: MarketSummary, attr: str, formatter) -> str:
    """Resolve a Market_Summary metric, marking unavailable rather than omitting.

    A metric is ``unavailable`` when its value is ``None`` or its name appears in
    ``summary.unavailable`` (Req 6.5); otherwise it is passed to ``formatter``.
    """
    if attr in summary.unavailable:
        return UNAVAILABLE
    return formatter(getattr(summary.metrics, attr))


# --- Section renderers (pure) ------------------------------------------------


def _render_regime(regime: MarketRegime) -> list[str]:
    """Render the Market Regime section, first in the report (Req 20.1, 20.2, 20.4)."""
    return [
        "## Market Regime",
        "",
        f"- Trend regime: {_label_or_unavailable(regime.trend)}",
        f"- Volatility regime: {_label_or_unavailable(regime.volatility)}",
        f"- Gamma regime: {_label_or_unavailable(regime.gamma_regime)}",
        f"- PCR level: {_label_or_unavailable(regime.pcr_level)}",
        "",
    ]


def _render_confidence(model: DailyReportModel) -> list[str]:
    """Render the Overall Research Confidence Score + its factors (Req 21.1, 21.3)."""
    confidence = model.confidence
    lines = [
        "## Overall Research Confidence Score",
        "",
        f"**{confidence.score}/100**",
        "",
        "Contributing factors:",
        "",
        "| Factor | Value | Weight |",
        "| --- | --- | --- |",
    ]
    for name, value, weight in confidence.factors:
        lines.append(f"| {name} | {value} | {weight} |")
    lines.append("")
    return lines


def _render_quality(model: DailyReportModel) -> list[str]:
    """Render the Quality_Score + PASS/FAIL verdict and any failing checks (Req 19.5)."""
    quality = model.quality
    lines = [
        "## Data Quality",
        "",
        f"**Quality_Score: {quality.score}/100 — {quality.verdict}**",
        "",
    ]
    failing = quality.failing_checks
    if failing:
        lines.append("Failing checks:")
        lines.append("")
        for check in failing:
            lines.append(f"- {check.name}: {check.detail}")
        lines.append("")
    return lines


def _render_what_happened(
    summary: MarketSummary, comparison: ComparisonResult
) -> list[str]:
    """Render Section 1 "What happened today" (Req 6.1-6.5)."""
    lines = [
        "## What happened today",
        "",
        f"- NIFTY change: {_metric_value(summary, 'nifty_pct_change', _fmt_pct)}",
        f"- India VIX change: {_metric_value(summary, 'india_vix_pct_change', _fmt_pct)}",
        f"- Put/Call ratio: {_metric_value(summary, 'pcr', lambda v: _fmt_number(v))}",
        f"- Max pain: {_metric_value(summary, 'max_pain', lambda v: _fmt_number(v))}",
        f"- Gamma exposure sign: {_metric_value(summary, 'gamma_sign', _fmt_gamma_sign)}",
        f"- Open-interest change: {_metric_value(summary, 'oi_change', lambda v: _fmt_number(v))}",
        "",
    ]
    if not comparison.history_available:
        lines.append(
            "_No historical comparison is available: the Target_Session is the "
            "only collected session._"
        )
        lines.append("")
    else:
        lines.append(
            f"_Comparison window: {comparison.prior_session_count} prior "
            f"session(s) collected (Collected_Session_Count = "
            f"{comparison.collected_session_count})._"
        )
        if comparison.collected_session_count < comparison.min_sample_size:
            lines.append(
                f"_Below the configured minimum sample size of "
                f"{comparison.min_sample_size} sessions; comparisons are not "
                f"claimed to be statistically significant._"
            )
        lines.append("")
    return lines


def _render_evidence_change(change: EvidenceChange) -> str:
    """Render one gained/weakened hypothesis with prior -> updated score (Req 7.2, 8.2)."""
    return (
        f"- Hypothesis #{change.hypothesis_id}: {change.text} — "
        f"Evidence_Score {change.prior_score} → {change.updated_score}. "
        f"{change.observation}"
    )


def _render_gained(gained: Sequence[EvidenceChange]) -> list[str]:
    """Render Section 2 "Hypotheses that gained evidence" (Req 7.1-7.4)."""
    lines = ["## Hypotheses that gained evidence", ""]
    if not gained:
        lines.append("No hypotheses gained evidence for this session.")
    else:
        lines.extend(_render_evidence_change(c) for c in gained)
    lines.append("")
    return lines


def _render_weakened(
    weakened: Sequence[EvidenceChange], rejected: Sequence[EvidenceChange]
) -> list[str]:
    """Render Section 3 "Hypotheses that weakened" incl. rejected (Req 8.1-8.4)."""
    lines = ["## Hypotheses that weakened", ""]
    if not weakened and not rejected:
        lines.append("No hypotheses weakened for this session.")
        lines.append("")
        return lines

    for change in weakened:
        lines.append(_render_evidence_change(change))
    for change in rejected:
        lines.append(
            f"- Hypothesis #{change.hypothesis_id}: {change.text} — moved to "
            f"rejected (Evidence_Score {change.prior_score} → "
            f"{change.updated_score}). Reason: {change.observation}"
        )
    lines.append("")
    return lines


def _render_unusual_event(event: UnusualEvent) -> str:
    """Render one Unusual_Event, qualifying small-sample events (Req 9.2-9.4)."""
    text = f"- {event.description}"
    if not event.significant:
        text += (
            f" (based on {event.sessions_spanned} session(s); not claimed "
            f"statistically significant)"
        )
    return text


def _render_unusual(events: Sequence[UnusualEvent]) -> list[str]:
    """Render Section 4 "Unusual events" (Req 9.1-9.5)."""
    lines = ["## Unusual events", ""]
    if not events:
        lines.append("No unusual events were detected for this session.")
    else:
        lines.extend(_render_unusual_event(e) for e in events)
    lines.append("")
    return lines


def _render_idea(idea: ResearchIdea) -> str:
    """Render one Research_Idea with its Priority and occurrence count (Req 10.3-10.5)."""
    priority = f"Priority {idea.priority}" if idea.priority is not None else "Priority -"
    text = f"- {priority}: {idea.text}"
    if idea.occurrence_count is not None:
        text += (
            f" (this pattern occurred {idea.occurrence_count} time(s) in "
            f"{idea.total_sessions} collected session(s))"
        )
    return text


def _render_ideas(ideas: Sequence[ResearchIdea]) -> list[str]:
    """Render Section 5 "What to investigate tomorrow" (Req 10.1-10.7)."""
    lines = ["## What to investigate tomorrow", ""]
    if not ideas:
        lines.append("No research ideas were generated for this session.")
    else:
        lines.extend(_render_idea(i) for i in ideas)
    lines.append("")
    return lines


def _render_validation_notes(notes: Sequence[str]) -> list[str]:
    """Render the captured validation/informational notes (Req 4.2, 4.3)."""
    lines = ["## Validation notes", ""]
    if not notes:
        lines.append("No validation notes for this session.")
    else:
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return lines


def _render_costs(model: DailyReportModel) -> list[str]:
    """Render the per-stage Computational Cost table (Req 22.3)."""
    lines = [
        "## Computational Cost",
        "",
        "| Stage | Time (s) | Peak memory (bytes) | Rows processed |",
        "| --- | --- | --- | --- |",
    ]
    for cost in model.stage_costs:
        lines.append(
            f"| {cost.stage_name} | {cost.elapsed_seconds:.4f} | "
            f"{cost.peak_memory_bytes} | {cost.rows_processed} |"
        )
    lines.append("")
    return lines


# --- Public API --------------------------------------------------------------


def render(model: DailyReportModel) -> str:
    """Render the Daily_Research_Report as a self-contained markdown string.

    Pure function of ``model`` (no I/O, no clock, no globals) producing one
    artifact containing -- in order -- the Market Regime section (first, Req
    20.1), the Overall Research Confidence Score and its factors (Req 21.1,
    21.3), the Quality_Score (Req 19.5), the five required sections (Req 6.1,
    7.1, 8.1, 9.1, 10.1) with unavailable metrics marked rather than omitted
    (Req 6.5) and empty-section messages where applicable (Req 7.4, 8.4, 9.5,
    10.7), the validation notes, and the per-stage Computational Cost table
    (Req 22.3). The whole report is self-contained (Req 13.4).
    """
    lines: list[str] = [
        f"# Daily Research Report — {model.target_session}",
        "",
    ]

    # Section 0: Market Regime (rendered first, Req 20.1).
    lines.extend(_render_regime(model.market_regime))
    # Overall Research Confidence Score + factors (Req 21).
    lines.extend(_render_confidence(model))
    # Quality_Score + verdict (Req 19.5).
    lines.extend(_render_quality(model))
    # The five required sections, in order (Req 6/7/8/9/10).
    lines.extend(_render_what_happened(model.market_summary, model.comparison))
    lines.extend(_render_gained(model.gained))
    lines.extend(_render_weakened(model.weakened, model.rejected))
    lines.extend(_render_unusual(model.unusual_events))
    lines.extend(_render_ideas(model.ideas))
    # Supporting context.
    lines.extend(_render_validation_notes(model.validation_notes))
    lines.extend(_render_costs(model))

    return "\n".join(lines).rstrip("\n") + "\n"


def write(path: str, text: str) -> None:
    """Write the rendered report to ``path``, overwriting any existing file.

    Thin I/O helper kept separate from :func:`render` so rendering stays pure
    and testable without disk access. Opening in ``"w"`` mode truncates an
    existing report for the same session, guaranteeing a single current report
    per session (Req 13.1, 13.3).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
