"""RoadmapWriter -- the single ROADMAP.md control center (Req 18).

The project keeps **one** authoritative ``ROADMAP.md`` at the repository root
(Req 18.4). On every pipeline run it is rewritten from scratch out of the
current journal + warehouse + data-quality state, so the document is always a
faithful snapshot rather than an append-only log (Req 18.2, 18.3).

The expanded control center renders, in order (Req 18.1):

* **Architecture Version** and **Frozen Components** -- static, config-supplied
  context describing the frozen ``nifty_quant`` layer the pipeline orchestrates.
* **Current Dataset** plus the three inventory counts -- **Collected Sessions**,
  **Collected Option Chains**, **Collected Trading Days** -- sourced from the
  ``SessionCatalog`` (Req 18.3).
* **Data Quality** -- the latest run's ``Quality_Score`` and PASS/FAIL verdict
  (Req 18.3); rendered as ``unavailable`` before the first scored run.
* **Open Hypotheses**, **Supported Hypotheses**, **Rejected Hypotheses** -- a
  strict three-way partition of the current journal state. Every non-terminal
  hypothesis (``open``/``testing``/``inconclusive``) is Open, every
  ``supported`` hypothesis is Supported, and every ``rejected`` hypothesis is
  Rejected -- so the three lists are disjoint and together cover the journal
  exactly (Property 19, Req 18.2).
* **Top Performing Experiments** -- every hypothesis ranked by Evidence_Score
  (highest first), decoding the ``evidence:<NN>`` tag through the
  ``EvidenceEngine`` so the journal stays the single source of truth (Req 11.7).
* **Research Priorities** -- the prioritized ``ResearchIdea`` list (Priority
  1..N, lowest number first).
* **Next Milestones** -- static/config-driven roadmap items.

Rendering is a **pure function** of the assembled :class:`RoadmapModel` (no I/O,
no clock, no globals) so it is trivially unit-testable and deterministic; the
only I/O lives in the thin :meth:`RoadmapWriter.write` helper. The module
mirrors the existing ``nifty_quant`` style (``from __future__ import
annotations`` + frozen value objects).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from nifty_quant.research.pipeline.evidence import EvidenceEngine

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from nifty_quant.research.journal import Hypothesis
    from nifty_quant.research.pipeline.models import ResearchIdea

# The single control-center document lives at the repository root (Req 18.4).
ROADMAP_PATH = "ROADMAP.md"

# Rendered placeholder for any value that is not yet available (e.g. the
# Quality_Score before the first scored run). Kept consistent with the report.
UNAVAILABLE = "unavailable"

# Terminal hypothesis statuses. Everything else is treated as Open (non-terminal)
# so the three rendered lists form a strict partition of the journal (Property 19).
_SUPPORTED = "supported"
_REJECTED = "rejected"


@dataclass(frozen=True)
class RoadmapModel:
    """Everything needed to render ``ROADMAP.md`` (Req 18.1).

    A plain frozen value object assembled by the pipeline from the current
    journal + warehouse + quality state. ``render`` consumes it directly, so the
    document is a deterministic function of this snapshot.
    """

    architecture_version: str
    frozen_components: list[str]
    current_dataset: str
    collected_sessions: int
    collected_option_chains: int
    collected_trading_days: int
    quality_score: int | None
    quality_verdict: str | None
    hypotheses: list["Hypothesis"]
    research_priorities: list["ResearchIdea"]
    next_milestones: list[str] = field(default_factory=list)


class RoadmapWriter:
    """Render and persist the single ``ROADMAP.md`` control center (Req 18)."""

    def __init__(self, evidence_engine: EvidenceEngine | None = None) -> None:
        # The engine only decodes the persisted ``evidence:<NN>`` tag, so the
        # default configuration is sufficient for ranking (Req 11.7).
        self._evidence = evidence_engine or EvidenceEngine()

    # --- Hypothesis partitioning (Property 19, Req 18.2) ---------------------

    def _partition(
        self, hypotheses: Sequence["Hypothesis"]
    ) -> tuple[list["Hypothesis"], list["Hypothesis"], list["Hypothesis"]]:
        """Split hypotheses into (open, supported, rejected) by status.

        The partition is strict: ``supported`` -> Supported, ``rejected`` ->
        Rejected, and every other (non-terminal) status -> Open. The three lists
        are disjoint and together contain every hypothesis exactly once
        (Property 19, Req 18.2).
        """
        open_h: list["Hypothesis"] = []
        supported: list["Hypothesis"] = []
        rejected: list["Hypothesis"] = []
        for h in hypotheses:
            status = getattr(h, "status", "open")
            if status == _SUPPORTED:
                supported.append(h)
            elif status == _REJECTED:
                rejected.append(h)
            else:
                open_h.append(h)
        return open_h, supported, rejected

    # --- Section renderers (pure) --------------------------------------------

    def _render_architecture(self, model: RoadmapModel) -> list[str]:
        """Render the Architecture Version + Frozen Components (Req 18.1)."""
        lines = [
            "## Architecture Version",
            "",
            f"{model.architecture_version}",
            "",
            "## Frozen Components",
            "",
        ]
        if model.frozen_components:
            lines.extend(f"- {component}" for component in model.frozen_components)
        else:
            lines.append("_No frozen components recorded._")
        lines.append("")
        return lines

    def _render_dataset(self, model: RoadmapModel) -> list[str]:
        """Render the Current Dataset + inventory counts (Req 18.1, 18.3)."""
        return [
            "## Current Dataset",
            "",
            f"{model.current_dataset}",
            "",
            f"- Collected Sessions: {model.collected_sessions}",
            f"- Collected Option Chains: {model.collected_option_chains}",
            f"- Collected Trading Days: {model.collected_trading_days}",
            "",
        ]

    def _render_quality(self, model: RoadmapModel) -> list[str]:
        """Render the latest Data Quality entry (Req 18.1, 18.3)."""
        if model.quality_score is None or model.quality_verdict is None:
            body = f"Quality_Score: {UNAVAILABLE}"
        else:
            body = f"Quality_Score: {model.quality_score}/100 — {model.quality_verdict}"
        return ["## Data Quality", "", body, ""]

    def _render_hypothesis_item(self, h: "Hypothesis") -> str:
        """Render one hypothesis line with its decoded Evidence_Score."""
        score = self._evidence.read_score(h)
        return f"- #{h.id}: {h.hypothesis} (Evidence_Score {score})"

    def _render_hypothesis_list(
        self, title: str, hypotheses: Sequence["Hypothesis"]
    ) -> list[str]:
        """Render a titled hypothesis list, or an explicit empty message."""
        lines = [f"## {title}", ""]
        if not hypotheses:
            lines.append(f"_No {title.lower()}._")
        else:
            lines.extend(self._render_hypothesis_item(h) for h in hypotheses)
        lines.append("")
        return lines

    def _render_top_experiments(
        self, hypotheses: Sequence["Hypothesis"]
    ) -> list[str]:
        """Render Top Performing Experiments ranked by Evidence_Score desc (Req 18.1).

        Decodes each hypothesis's Evidence_Score through the ``EvidenceEngine``
        and orders by score descending; ties keep journal order (stable sort), so
        the ranking is deterministic.
        """
        lines = ["## Top Performing Experiments", ""]
        if not hypotheses:
            lines.append("_No experiments recorded._")
            lines.append("")
            return lines

        ranked = sorted(
            hypotheses,
            key=lambda h: self._evidence.read_score(h),
            reverse=True,
        )
        for rank, h in enumerate(ranked, start=1):
            score = self._evidence.read_score(h)
            status = getattr(h, "status", "open")
            lines.append(
                f"{rank}. #{h.id}: {h.hypothesis} — Evidence_Score {score} "
                f"({status})"
            )
        lines.append("")
        return lines

    def _render_priorities(
        self, priorities: Sequence["ResearchIdea"]
    ) -> list[str]:
        """Render the Research Priorities (prioritized ResearchIdea list, Req 18.1)."""
        lines = ["## Research Priorities", ""]
        if not priorities:
            lines.append("_No research priorities recorded._")
            lines.append("")
            return lines

        ordered = sorted(
            priorities,
            key=lambda idea: (idea.priority is None, idea.priority or 0),
        )
        for idea in ordered:
            priority = (
                f"Priority {idea.priority}" if idea.priority is not None else "Priority -"
            )
            lines.append(f"- {priority}: {idea.text}")
        lines.append("")
        return lines

    def _render_milestones(self, milestones: Sequence[str]) -> list[str]:
        """Render the Next Milestones list (Req 18.1)."""
        lines = ["## Next Milestones", ""]
        if not milestones:
            lines.append("_No milestones recorded._")
        else:
            lines.extend(f"- {milestone}" for milestone in milestones)
        lines.append("")
        return lines

    # --- Public API ----------------------------------------------------------

    def render(self, model: RoadmapModel) -> str:
        """Render ``ROADMAP.md`` as a self-contained markdown string.

        Pure function of ``model`` (no I/O, no clock, no globals) producing the
        whole control center in order: Architecture Version + Frozen Components,
        Current Dataset + inventory counts, Data Quality, the three-way
        hypothesis partition (Open/Supported/Rejected), Top Performing
        Experiments ranked by Evidence_Score, Research Priorities, and Next
        Milestones (Req 18.1). The Open/Supported/Rejected lists form a strict
        partition of the journal (Property 19, Req 18.2).
        """
        open_h, supported, rejected = self._partition(model.hypotheses)

        lines: list[str] = ["# ROADMAP", ""]
        lines.extend(self._render_architecture(model))
        lines.extend(self._render_dataset(model))
        lines.extend(self._render_quality(model))
        lines.extend(self._render_hypothesis_list("Open Hypotheses", open_h))
        lines.extend(self._render_hypothesis_list("Supported Hypotheses", supported))
        lines.extend(self._render_hypothesis_list("Rejected Hypotheses", rejected))
        lines.extend(self._render_top_experiments(model.hypotheses))
        lines.extend(self._render_priorities(model.research_priorities))
        lines.extend(self._render_milestones(model.next_milestones))

        return "\n".join(lines).rstrip("\n") + "\n"

    def write(self, path: str, text: str) -> None:
        """Write the rendered roadmap to ``path``, overwriting any existing file.

        Thin I/O helper kept separate from :meth:`render` so rendering stays pure
        and testable without disk access. Opening in ``"w"`` mode truncates the
        existing document, keeping a single authoritative ``ROADMAP.md`` at the
        repository root (Req 18.4).
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
