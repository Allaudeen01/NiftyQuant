"""IdeaGenerator -- rule-based (primary/default) Research_Idea generation.

Research_Ideas are the "What to investigate tomorrow" section of the Daily
Research Report (Requirement 10). They are framed strictly as *investigations to
test*, never as trade recommendations (Req 10.6), and whenever an idea refers to
how often a pattern has been seen it states that count **relative to the
collected sessions** (Req 10.5).

The **rule-based generator is the primary, default hypothesis source** so that
the full run is deterministic and trade-free with the AI enhancement disabled
(Req 15.1, 15.2). The optional ``AIIdeaGenerator`` (task 18.2) only *supplements*
these ideas and falls back to the rule-based generator on any failure.

This module provides three pieces:

* ``IdeaGenerator``           -- the ``Protocol`` every generator satisfies.
* ``RuleBasedIdeaGenerator``  -- the deterministic default generator that derives
  :class:`ResearchIdea` records from the Target_Session :class:`MarketSummary`,
  the :class:`ComparisonResult` (its detected unusual events), and the collected
  :class:`SessionMetrics` history.
* ``sanitize_idea``           -- the trade-recommendation guard that raises on any
  trade-call phrasing, guaranteeing research-only output for both the rule-based
  and the AI paths (Req 10.6, 15.4).

Dataclasses/values mirror the existing ``nifty_quant`` style (``from __future__
import annotations`` + plain value objects). ``Expected_Information_Gain`` and
``Priority`` are intentionally left at their defaults here -- ranking/priority is
assigned by the prioritization logic in a later task (Req 10.3, 10.4).
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from nifty_quant.research.pipeline.models import ResearchIdea

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from nifty_quant.research.pipeline.models import (
        ComparisonResult,
        MarketSummary,
        SessionMetrics,
        UnusualEvent,
    )


# Trade-recommendation phrasing that may never appear in a Research_Idea
# (Req 10.6, 15.4). Matched case-insensitively on word boundaries so legitimate
# substrings (e.g. "buyer", "selling pressure", "entered", "context") are not
# falsely flagged, while multi-word calls allow flexible internal whitespace.
_TRADE_PHRASE_RE = re.compile(
    r"\b(?:buy|sell|enter|exit|go\s+long|go\s+short|take\s+a\s+position)\b",
    re.IGNORECASE,
)


@runtime_checkable
class IdeaGenerator(Protocol):
    """A source of :class:`ResearchIdea` records for the suggestion stage.

    Implementations derive ideas from the Target_Session :class:`MarketSummary`,
    the :class:`ComparisonResult`, and the collected Prior_Session history. The
    rule-based implementation is the deterministic default (Req 15.1).
    """

    def generate(
        self,
        summary: "MarketSummary",
        comparison: "ComparisonResult",
        history: "Sequence[SessionMetrics]",
    ) -> list[ResearchIdea]:
        ...


class RuleBasedIdeaGenerator:
    """Deterministic, primary/default Research_Idea generator (Req 15.1, 15.2).

    Derives :class:`ResearchIdea` records from observed Target_Session data
    patterns -- the unusual events detected by the comparison stage plus the
    gamma/PCR regime combination of the day -- with no randomness, AI, or
    wall-clock input, so identical inputs always yield identical ideas
    (Req 15.2, 17.1).

    Every idea:

    * is phrased as an *investigation to test*, never a trade recommendation
      (Req 10.6) -- this is enforced by passing each idea through
      :func:`sanitize_idea` before it is returned;
    * carries ``total_sessions`` set to the ``Collected_Session_Count`` and,
      where it references how often a pattern occurred, an ``occurrence_count``
      stated relative to those collected sessions (Req 10.5);
    * leaves ``information_gain`` at ``0.0`` and ``priority`` unset -- ranking and
      Priority assignment are handled by the prioritization step (Req 10.3,
      10.4).

    When no idea can be derived from the Target_Session data, an empty list is
    returned; the report stage renders the "no research ideas were generated"
    message in that case (Req 10.7).
    """

    def generate(
        self,
        summary: "MarketSummary",
        comparison: "ComparisonResult",
        history: "Sequence[SessionMetrics]",
    ) -> list[ResearchIdea]:
        """Generate ordered, research-only ideas for the Target_Session.

        Ideas are produced in a fixed, deterministic order:

        1. one idea per detected :class:`UnusualEvent` (extremity then gamma
           flip, in the order the comparison stage recorded them), and
        2. one idea for the day's gamma/PCR regime combination, counting how
           often that combination occurred across the collected sessions.

        Each candidate is validated by :func:`sanitize_idea` before inclusion, so
        the returned list is guaranteed free of trade-recommendation phrasing
        (Req 10.6).
        """
        total_sessions = comparison.collected_session_count
        priors = list(history)

        ideas: list[ResearchIdea] = []

        for event in comparison.unusual_events:
            ideas.append(self._idea_from_event(event, summary, priors, total_sessions))

        combination = self._idea_from_combination(summary, priors, total_sessions)
        if combination is not None:
            ideas.append(combination)

        # The generator controls its own phrasing, so this guard is an internal
        # invariant check: it raises only if a template regresses into a trade
        # call, never on well-formed research phrasing (Req 10.6).
        return [sanitize_idea(idea) for idea in ideas]

    # --- Per-source idea construction ----------------------------------------

    def _idea_from_event(
        self,
        event: "UnusualEvent",
        summary: "MarketSummary",
        priors: "Sequence[SessionMetrics]",
        total_sessions: int,
    ) -> ResearchIdea:
        """Build a :class:`ResearchIdea` investigating one unusual event."""
        if event.metric == "gamma_sign":
            # A gamma flip: count how often the gamma sign flipped across the
            # full collected series so the idea states an honest occurrence count
            # relative to the dataset (Req 10.5).
            flips = _count_gamma_flips(priors, summary)
            text = (
                "Investigate whether a gamma-sign flip versus the prior session "
                "tends to precede a directional follow-through, and what "
                f"conditions accompany it (gamma flips occurred {flips} times in "
                f"{total_sessions} collected sessions)."
            )
            return ResearchIdea(
                text=text,
                occurrence_count=flips,
                total_sessions=total_sessions,
            )

        # An extremity event: the Target_Session value is the strict max/min of
        # the metric over the collected sessions -- a single, unique occurrence.
        text = (
            f"Investigate what conditions preceded the {event.description} and "
            "whether such extremes mean-revert in the following sessions "
            f"(observed once in {total_sessions} collected sessions)."
        )
        return ResearchIdea(
            text=text,
            occurrence_count=1,
            total_sessions=total_sessions,
        )

    def _idea_from_combination(
        self,
        summary: "MarketSummary",
        priors: "Sequence[SessionMetrics]",
        total_sessions: int,
    ) -> ResearchIdea | None:
        """Build a regime-combination idea, or ``None`` when not derivable.

        Classifies the Target_Session by its gamma sign and its PCR level
        (relative to the collected history's PCR median), then counts how many
        Prior_Sessions share the same combination. Returns ``None`` when the
        gamma sign or PCR is unavailable, or when there is no history to count
        against, so an idea is only emitted when it is grounded in data.
        """
        if not priors:
            return None

        metrics = summary.metrics
        gamma_sign = metrics.gamma_sign
        pcr = metrics.pcr
        if gamma_sign is None or pcr is None:
            return None

        pcr_values = [m.pcr for m in priors if m.pcr is not None]
        if not pcr_values:
            return None
        pcr_median = _median([float(v) for v in pcr_values])

        target_pcr_level = _pcr_level(pcr, pcr_median)
        gamma_label = _gamma_label(gamma_sign)

        occurrences = sum(
            1
            for m in priors
            if m.gamma_sign == gamma_sign
            and m.pcr is not None
            and _pcr_level(m.pcr, pcr_median) == target_pcr_level
        )

        text = (
            f"Investigate whether sessions combining {gamma_label} gamma "
            f"exposure with {target_pcr_level} put/call ratio are followed by a "
            "consistent next-session move (this combination occurred "
            f"{occurrences} times in {total_sessions} collected sessions)."
        )
        return ResearchIdea(
            text=text,
            occurrence_count=occurrences,
            total_sessions=total_sessions,
        )


def sanitize_idea(idea: "str | ResearchIdea") -> "str | ResearchIdea":
    """Guard a Research_Idea against trade-recommendation phrasing (Req 10.6, 15.4).

    Detects trade-call phrasing -- ``buy``, ``sell``, ``enter``, ``exit``,
    ``go long``, ``go short``, ``take a position`` -- case-insensitively and on
    word boundaries, so research wording that merely contains those letters
    (e.g. "buyer", "context", "selling pressure") is **not** flagged.

    Behaviour: if any trade-recommendation phrase is found, a :class:`ValueError`
    is raised. The rule-based generator controls its own phrasing, so this never
    fires for its output; the AI path (task 18.2) reuses this same guard and
    treats the raised error as the signal to discard the offending idea and fall
    back to rule-based generation (Req 15.4).

    The input is returned unchanged when clean: a ``str`` in yields the same
    ``str`` out, and a :class:`ResearchIdea` in yields the same record out, so the
    function composes transparently in either path.
    """
    text = idea.text if isinstance(idea, ResearchIdea) else idea
    match = _TRADE_PHRASE_RE.search(text)
    if match is not None:
        raise ValueError(
            "Research_Idea contains trade-recommendation phrasing "
            f"{match.group(0)!r}; ideas must be framed as investigations only "
            "(Req 10.6)."
        )
    return idea


class AIIdeaGenerator:
    """Optional, opt-in AI wrapper that *supplements* the rule-based ideas (Req 15.3).

    The AI enhancement is disabled by default (``PipelineConfig.use_ai = False``,
    Req 15.1); only when AI is explicitly enabled does the orchestrator construct
    this generator. It never replaces the deterministic rule-based source: it
    always computes the rule-based ideas first via its ``fallback`` (the primary
    source) and then asks an injected ``backend`` for *additional*, supplemental
    research ideas. It satisfies the :class:`IdeaGenerator` ``Protocol``.

    **Backend interface (minimal, duck-typed).** The backend is an injected object
    that exposes::

        backend.propose(summary, comparison, history) -> Iterable[ResearchIdea | str]

    (a plain callable ``backend(summary, comparison, history)`` is also accepted).
    It returns zero or more candidate ideas, each either a :class:`ResearchIdea`
    or a plain ``str`` -- a ``str`` is tolerantly converted to a
    :class:`ResearchIdea` framed against the collected-session count so it stays
    consistent with the rule-based ideas (Req 10.5).

    **Sanitization & failure handling.** Every candidate is passed through
    :func:`sanitize_idea`; a candidate whose phrasing is a trade call raises and is
    discarded *individually* -- the rest of the batch survives (Req 10.6). Only a
    failure of the *backend itself* (unavailable, malformed, timeout, missing
    ``propose``/non-callable, ...) abandons the whole AI step and returns the
    rule-based result unchanged (Req 15.4).

    **Deterministic ordering.** The result lists the rule-based ideas first (in
    their deterministic order), then the accepted AI ideas in backend order.
    Duplicate AI ideas (same ``text`` as an already-included idea) are skipped so
    the supplement never repeats the rule-based output.
    """

    def __init__(self, backend: object, fallback: RuleBasedIdeaGenerator) -> None:
        self._backend = backend
        self._fallback = fallback

    @property
    def fallback(self) -> RuleBasedIdeaGenerator:
        """The rule-based generator used as the primary source and the fallback."""
        return self._fallback

    def generate(
        self,
        summary: "MarketSummary",
        comparison: "ComparisonResult",
        history: "Sequence[SessionMetrics]",
    ) -> list[ResearchIdea]:
        """Return the rule-based ideas supplemented with sanitized AI ideas.

        The rule-based ideas are computed first so they always anchor the result,
        even before the backend is consulted (Req 15.1). The backend is then asked
        for supplemental ideas; on *any* backend failure the rule-based result is
        returned unchanged (Req 15.4).
        """
        base = self._fallback.generate(summary, comparison, history)
        try:
            proposed = self._call_backend(summary, comparison, history)
        except Exception:
            # Any backend failure -> the run stays on the deterministic rule-based
            # ideas with no AI supplement (Req 15.4).
            return base

        total_sessions = comparison.collected_session_count
        seen = {idea.text for idea in base}
        supplemental: list[ResearchIdea] = []
        for candidate in proposed:
            idea = self._coerce_idea(candidate, total_sessions)
            try:
                sanitize_idea(idea)
            except ValueError:
                # A single trade-phrased AI idea is dropped; the batch survives --
                # only a backend exception triggers the full fallback (Req 10.6).
                continue
            if idea.text in seen:
                continue
            seen.add(idea.text)
            supplemental.append(idea)

        # Rule-based first, then accepted AI ideas -> deterministic ordering.
        return base + supplemental

    def _call_backend(
        self,
        summary: "MarketSummary",
        comparison: "ComparisonResult",
        history: "Sequence[SessionMetrics]",
    ) -> list:
        """Invoke the backend and return a list of candidate ideas.

        Accepts either a ``backend.propose(...)`` method or a plain callable
        backend. A missing/invalid backend raises, which :meth:`generate` catches
        and turns into the rule-based fallback (Req 15.4).
        """
        propose = getattr(self._backend, "propose", None)
        if propose is None and callable(self._backend):
            propose = self._backend
        if propose is None:
            raise TypeError(
                "AI backend must provide a propose(summary, comparison, history) "
                "method or be callable."
            )
        proposed = propose(summary, comparison, history)
        if proposed is None:
            return []
        return list(proposed)

    @staticmethod
    def _coerce_idea(
        candidate: "str | ResearchIdea", total_sessions: int
    ) -> ResearchIdea:
        """Tolerantly convert a backend candidate into a :class:`ResearchIdea`.

        A :class:`ResearchIdea` is returned unchanged; a plain ``str`` is wrapped
        as an idea with no occurrence count, framed against the collected-session
        count so it stays consistent with the rule-based ideas (Req 10.5).
        """
        if isinstance(candidate, ResearchIdea):
            return candidate
        return ResearchIdea(
            text=str(candidate),
            occurrence_count=None,
            total_sessions=total_sessions,
        )


# --- Information gain & prioritization (Req 10.3, 10.4) ----------------------

# Weights for the deterministic Expected_Information_Gain estimate. They sum to
# 1.0 so the returned gain is normalised to ``[0.0, 1.0]``. Rarity dominates
# (a rarer / more surprising pattern is expected to teach the most), while the
# sampling factor down-weights ideas drawn from a thin, not-yet-significant
# dataset.
_GAIN_W_RARITY = 0.7
_GAIN_W_SAMPLE = 0.3

# Neutral rarity prior used when a pattern's frequency is unknown (no
# ``occurrence_count``) or cannot be normalised (no collected sessions): neither
# surprising nor unsurprising.
_GAIN_NEUTRAL_RARITY = 0.5


def estimate_information_gain(
    idea: ResearchIdea,
    comparison: "ComparisonResult",
    history: "Sequence[SessionMetrics]",
) -> float:
    """Deterministic Expected_Information_Gain estimate for a Research_Idea (Req 10.3).

    Returns a normalised score in ``[0.0, 1.0]`` -- higher means the
    investigation is expected to teach more. The estimate combines two
    deterministic, data-grounded factors with fixed weights (no randomness, AI,
    or wall-clock input), so identical inputs always yield the same score:

    * **Rarity / surprise** (weight ``_GAIN_W_RARITY``). A pattern that has been
      observed in only a small fraction of the collected sessions is more
      surprising, so it scores higher::

          rarity = 1 - (occurrence_count / total_sessions)

      ``occurrence_count`` is clamped to ``[0, total_sessions]`` so the term stays
      in ``[0, 1]``. When the frequency is unknown (``occurrence_count is None``)
      or cannot be normalised (``total_sessions <= 0``), a neutral prior of
      ``_GAIN_NEUTRAL_RARITY`` is used instead of guessing.

    * **Sampling confidence** (weight ``_GAIN_W_SAMPLE``). An idea drawn from a
      better-sampled dataset is more trustworthy and therefore expected to teach
      a cleaner lesson. This factor scales the number of collected sessions
      against the configured significance threshold, saturating at 1.0::

          sample_confidence = min(1.0, collected_session_count / min_sample_size)

      The sampling context comes from ``comparison`` (its
      ``collected_session_count`` and ``min_sample_size``); when the comparison
      reports no collected sessions the length of ``history`` is used as the
      fallback session count so the factor is still grounded in real data.

    The final estimate is the weighted sum
    ``_GAIN_W_RARITY * rarity + _GAIN_W_SAMPLE * sample_confidence``, which lies
    in ``[0.0, 1.0]`` because both factors are in ``[0, 1]`` and the weights sum
    to one. The function is pure and does not mutate ``idea``; callers attach the
    result via ``dataclasses.replace(idea, information_gain=...)``.
    """
    # --- Rarity / surprise -------------------------------------------------
    if idea.occurrence_count is None or idea.total_sessions <= 0:
        rarity = _GAIN_NEUTRAL_RARITY
    else:
        occurrences = max(0, min(idea.occurrence_count, idea.total_sessions))
        rarity = 1.0 - (occurrences / idea.total_sessions)

    # --- Sampling confidence ----------------------------------------------
    collected = comparison.collected_session_count
    if collected <= 0:
        collected = len(history)
    min_sample = comparison.min_sample_size
    if min_sample > 0:
        sample_confidence = min(1.0, collected / min_sample)
    else:
        sample_confidence = 1.0

    return _GAIN_W_RARITY * rarity + _GAIN_W_SAMPLE * sample_confidence


def prioritize(ideas: "Sequence[ResearchIdea]") -> list[ResearchIdea]:
    """Rank Research_Ideas by descending Expected_Information_Gain (Req 10.3, 10.4).

    Sorts the ideas by their already-computed ``information_gain`` in descending
    order and assigns each a ``Priority`` of ``1..N`` in that ranked order, with
    Priority 1 being the highest-gain idea -- the first experiment expected to
    teach the most (Req 10.4). The sort is **stable**: ideas with equal
    ``information_gain`` keep their original input order, so prioritization is
    deterministic for tied gains.

    Because :class:`ResearchIdea` is a frozen dataclass, a new instance is
    returned for each idea (via :func:`dataclasses.replace`) with ``priority``
    set; the inputs are left unmutated. The assigned priorities are exactly the
    contiguous integers ``1..len(ideas)`` with no gaps or duplicates. An empty
    input yields an empty list.
    """
    # Python's sort is stable, and ``reverse=True`` preserves that stability for
    # equal keys, so tied gains retain their original relative order (Req 10.4).
    ranked = sorted(ideas, key=lambda idea: idea.information_gain, reverse=True)
    return [replace(idea, priority=rank) for rank, idea in enumerate(ranked, start=1)]


# --- Internal helpers --------------------------------------------------------


def _count_gamma_flips(
    priors: "Sequence[SessionMetrics]", summary: "MarketSummary"
) -> int:
    """Count gamma-sign flips across the full collected session series.

    The series is the Prior_Sessions (ascending) followed by the Target_Session.
    Sessions with an unavailable gamma sign are skipped, and a flip is counted
    for each adjacent pair of *available* signs that differ. Deterministic.
    """
    signs = [m.gamma_sign for m in priors if m.gamma_sign is not None]
    target_sign = summary.metrics.gamma_sign
    if target_sign is not None:
        signs.append(target_sign)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _gamma_label(gamma_sign: int) -> str:
    """Human-readable label for a gamma sign (-1/0/+1)."""
    if gamma_sign > 0:
        return "positive"
    if gamma_sign < 0:
        return "negative"
    return "neutral"


def _pcr_level(pcr: float, median: float) -> str:
    """Classify a PCR value relative to the collected history median."""
    if pcr > median:
        return "elevated"
    if pcr < median:
        return "depressed"
    return "in-line"


def _median(values: Sequence[float]) -> float:
    """Median of a non-empty numeric sequence (deterministic, no imports)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
