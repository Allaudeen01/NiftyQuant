"""EvidenceEngine -- pure Evidence_Score scoring and status logic.

The Evidence_Score (Req 11) is an integer in ``0..100`` maintained per
``Hypothesis``. It rises as supporting evidence accumulates, falls when
contradicting evidence appears, and decays toward ``0`` across sessions where a
hypothesis receives no evidence at all. The score is **persisted inside the
existing ``Hypothesis`` record** as a structured ``evidence:<NN>`` tag (no
separate store, Req 12.5) -- the journal stays the single source of truth
(Req 11.7).

This module holds only the *pure* scoring logic so it is fully unit-testable
with synthetic inputs (Req 17.3) and deterministic across identical runs
(Req 17.1):

* ``read_score``    -- decode the ``evidence:<NN>`` tag from a ``Hypothesis``
  (defaulting to ``config.initial_score``).
* ``encode_score``  -- produce an updated tag list with the ``evidence:<NN>``
  tag replacing any existing ``evidence:*`` tag, plus the mirrored
  ``confidence = NN/100`` value (Req 11.7, 12.5).
* ``apply``         -- the core pure update, clamped to ``[0, 100]``.
* ``next_status``   -- map a score to a lifecycle status in
  ``{open, testing, supported, rejected, inconclusive}`` honouring the
  configured reject/support thresholds (Req 11.5, 11.6, 12.4).
* ``evaluate``      -- a deterministic rule that maps a hypothesis (by its tags)
  plus the Target_Session ``MarketSummary`` and prior ``SessionMetrics`` history
  to a ``Verdict`` (SUPPORTING / CONTRADICTING / ABSENT).
* ``persist``       -- write an updated Evidence_Score (encoded as the
  ``evidence:<NN>`` tag + mirrored ``confidence``), status, and reason back
  through the existing ``ResearchJournal.update`` without altering its
  interface (Req 11.7, 12.2, 12.5).

Dataclasses/enums mirror the existing ``nifty_quant`` style (``from __future__
import annotations`` + plain frozen value objects).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from nifty_quant.research.journal import Hypothesis, ResearchJournal
    from nifty_quant.research.pipeline.models import MarketSummary, SessionMetrics


# Matches a single, well-formed evidence tag like ``evidence:42``. Anchored so a
# tag such as ``evidence:abc`` or ``evidence:`` is ignored (falls back to the
# configured initial score).
_EVIDENCE_TAG_RE = re.compile(r"^evidence:(\d{1,3})$")

# Prefix used to recognise *any* evidence tag (well-formed or stale) when
# re-encoding a score, so a previous ``evidence:*`` tag is always replaced rather
# than duplicated.
_EVIDENCE_TAG_PREFIX = "evidence:"

# SessionMetrics fields whose natural reference point is zero: a positive value
# is an "up" move, a negative value a "down" move. The remaining numeric fields
# are *levels* and are judged relative to the median of the collected history.
_SIGNED_METRICS = frozenset(
    {
        "nifty_pct_change",
        "india_vix_pct_change",
        "oi_change",
        "gamma_sign",
        "total_gex",
    }
)
_LEVEL_METRICS = frozenset(
    {
        "india_vix",
        "pcr",
        "max_pain",
        "atm_iv",
    }
)

# Direction-tag synonyms understood by ``evaluate``.
_UP_DIRECTIONS = frozenset({"up", "rise", "rises", "increase", "positive", "higher"})
_DOWN_DIRECTIONS = frozenset({"down", "fall", "falls", "decrease", "negative", "lower"})


@dataclass
class EvidenceConfig:
    """Tunable constants for Evidence_Score updates and status thresholds (Req 11)."""

    support_increment: int = 10      # added on a SUPPORTING verdict
    contradict_decrement: int = 15   # subtracted on a CONTRADICTING verdict
    decay_per_session: int = 5       # subtracted on an ABSENT (no-evidence) session
    support_threshold: int = 80      # score >= this  => status "supported"
    reject_threshold: int = 20       # score <= this  => status "rejected"
    initial_score: int = 50          # default when a hypothesis has no evidence tag


class Verdict(Enum):
    """The evidence outcome for a hypothesis on the Target_Session."""

    SUPPORTING = "supporting"        # observation supports the hypothesis => raise
    CONTRADICTING = "contradicting"  # observation contradicts it          => lower
    ABSENT = "absent"                # no evidence this session             => decay


class EvidenceEngine:
    """Pure Evidence_Score scoring/status logic (Req 11, 12.4)."""

    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self.config = config or EvidenceConfig()

    # --- Score persistence (decode + encode + journal integration) -----------

    def read_score(self, h: "Hypothesis") -> int:
        """Decode the ``evidence:<NN>`` tag from a hypothesis.

        Returns the first well-formed ``evidence:<NN>`` tag value clamped to
        ``[0, 100]``. When no such tag exists (or it is malformed), falls back to
        ``config.initial_score`` (Req 11.7).
        """
        for tag in getattr(h, "tags", None) or []:
            match = _EVIDENCE_TAG_RE.match(str(tag))
            if match:
                return _clamp(int(match.group(1)))
        return _clamp(self.config.initial_score)

    def encode_score(
        self, tags: "Sequence[str] | None", score: int
    ) -> tuple[list[str], float]:
        """Encode an Evidence_Score into a tag list and mirrored confidence.

        Produces ``(new_tags, confidence)`` where ``new_tags`` is ``tags`` with
        any existing ``evidence:*`` tag removed and a single ``evidence:<NN>``
        tag (``NN`` = the score clamped to ``[0, 100]``) appended, and
        ``confidence`` is the mirrored ``NN / 100`` value for readability
        (Req 11.7, 12.5). The non-evidence tags keep their original order, so the
        encoding round-trips losslessly through ``ResearchJournal`` (Property 4).
        """
        clamped = _clamp(score)
        kept = [str(t) for t in (tags or []) if not str(t).startswith(_EVIDENCE_TAG_PREFIX)]
        kept.append(f"{_EVIDENCE_TAG_PREFIX}{clamped}")
        return kept, clamped / 100.0

    def persist(
        self,
        journal: "ResearchJournal",
        h: "Hypothesis",
        score: int,
        status: str,
        reason: str,
    ) -> "Hypothesis":
        """Persist an Evidence_Score, status, and reason for a hypothesis.

        Encodes ``score`` into the hypothesis tags as ``evidence:<NN>`` (mirrored
        into ``confidence = NN/100``) and writes the score/status/reason/tags
        changes through the existing ``ResearchJournal.update`` (Req 12.2, 12.5),
        keeping the journal the single source of truth (Req 11.7). The frozen
        ``ResearchJournal`` interface is used unchanged. Returns the updated
        ``Hypothesis`` as reported by ``ResearchJournal.update``.
        """
        tags, confidence = self.encode_score(getattr(h, "tags", None), score)
        return journal.update(
            h.id,
            status=status,
            confidence=confidence,
            reason=reason,
            tags=tags,
        )

    # --- Core pure update ----------------------------------------------------

    def apply(self, score: int, verdict: Verdict) -> int:
        """Apply a verdict to a score, clamped to ``[0, 100]`` (Req 11.1-11.4).

        * ``SUPPORTING``    -> ``min(100, score + support_increment)``
        * ``CONTRADICTING`` -> ``max(0,  score - contradict_decrement)``
        * ``ABSENT``        -> ``max(0,  score - decay_per_session)``
        """
        cfg = self.config
        if verdict is Verdict.SUPPORTING:
            return min(100, score + cfg.support_increment)
        if verdict is Verdict.CONTRADICTING:
            return max(0, score - cfg.contradict_decrement)
        if verdict is Verdict.ABSENT:
            return max(0, score - cfg.decay_per_session)
        raise ValueError(f"unknown verdict {verdict!r}")

    # --- Status mapping ------------------------------------------------------

    def next_status(self, score: int, current: str) -> str:
        """Map an Evidence_Score to a lifecycle status (Req 11.5, 11.6, 12.4).

        * ``score <= reject_threshold``  -> ``"rejected"``
        * ``score >= support_threshold`` -> ``"supported"``
        * otherwise the hypothesis stays in its current non-terminal status; a
          terminal status (``supported``/``rejected``) that no longer matches its
          threshold reopens to ``"testing"``, and a fresh ``"open"`` hypothesis
          that has received scoring becomes ``"testing"``.

        The result is always one of ``{open, testing, supported, rejected,
        inconclusive}``.
        """
        cfg = self.config
        if score <= cfg.reject_threshold:
            return "rejected"
        if score >= cfg.support_threshold:
            return "supported"

        # Middle band: not terminal. Keep a meaningful non-terminal status.
        if current in ("supported", "rejected"):
            # Was terminal but the score drifted back into the middle band.
            return "testing"
        if current == "open":
            # Has now been scored, so it is actively under test.
            return "testing"
        if current in ("testing", "inconclusive"):
            return current
        # Unknown/invalid incoming status: normalise to a valid non-terminal one.
        return "testing"

    # --- Verdict evaluation --------------------------------------------------

    def evaluate(
        self,
        h: "Hypothesis",
        summary: "MarketSummary",
        history: "Sequence[SessionMetrics]",
    ) -> Verdict:
        """Deterministically map a hypothesis to a ``Verdict`` from its tags.

        The hypothesis declares what it is about through two structured tags:

        * ``metric:<field>``  -- a ``SessionMetrics`` field name
          (e.g. ``metric:nifty_pct_change``, ``metric:pcr``, ``metric:gamma_sign``).
        * ``direction:<dir>`` -- the predicted direction of that metric for the
          Target_Session, where ``<dir>`` is an up-synonym (``up``/``rise``/
          ``increase``/``positive``/``higher``) or a down-synonym (``down``/
          ``fall``/``decrease``/``negative``/``lower``).

        The Target_Session value is read from ``summary.metrics``; its observed
        direction is the sign relative to a reference point (zero for *change*
        metrics, the median of the collected ``history`` for *level* metrics):

        * observed direction matches the predicted direction -> ``SUPPORTING``
        * observed direction is the opposite                 -> ``CONTRADICTING``
        * no usable metric/direction tag, the metric is unavailable, there is no
          baseline for a level metric, or the metric did not move ->
          ``ABSENT`` (which drives Evidence_Score decay, Req 11.4)

        The rule is a pure function of the inputs, so identical inputs always
        yield the same verdict (Req 17.1).
        """
        tags = [str(t) for t in (getattr(h, "tags", None) or [])]

        field = _tag_value(tags, "metric")
        predicted = _direction_sign(_tag_value(tags, "direction"))
        if field is None or predicted == 0:
            return Verdict.ABSENT

        value = _metric_value(summary, field)
        if value is None:
            return Verdict.ABSENT

        reference = _reference_for(field, history)
        if reference is None:
            return Verdict.ABSENT

        observed = _sign(value - reference)
        if observed == 0:
            return Verdict.ABSENT  # no movement => no evidence this session

        return Verdict.SUPPORTING if observed == predicted else Verdict.CONTRADICTING


# --- Internal helpers --------------------------------------------------------


def _clamp(score: int) -> int:
    """Clamp an integer score into the inclusive ``[0, 100]`` band."""
    return max(0, min(100, int(score)))


def _tag_value(tags: Sequence[str], key: str) -> str | None:
    """Return the value of the first ``<key>:<value>`` tag, lower-cased."""
    prefix = f"{key}:"
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix):].strip().lower()
    return None


def _direction_sign(direction: str | None) -> int:
    """Map a direction-tag value to ``+1`` (up), ``-1`` (down), or ``0`` (none)."""
    if direction is None:
        return 0
    if direction in _UP_DIRECTIONS:
        return 1
    if direction in _DOWN_DIRECTIONS:
        return -1
    return 0


def _sign(x: float) -> int:
    """Sign of ``x`` as ``+1`` / ``-1`` / ``0``."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _metric_value(summary: "MarketSummary", field: str) -> float | None:
    """Read a numeric metric from ``summary.metrics`` by field name.

    Returns ``None`` when the field is unknown or its value is unavailable
    (``None``), so the caller can treat it as an ABSENT (no-evidence) session.
    """
    if field not in _SIGNED_METRICS and field not in _LEVEL_METRICS:
        return None
    metrics = getattr(summary, "metrics", None)
    if metrics is None:
        return None
    value = getattr(metrics, field, None)
    return None if value is None else float(value)


def _reference_for(field: str, history: "Sequence[SessionMetrics]") -> float | None:
    """Reference point used to judge a metric's observed direction.

    * *signed* metrics (changes / signs) are judged against ``0``.
    * *level* metrics are judged against the median of the collected history for
      that field; when no historical value is available the direction cannot be
      established, so ``None`` is returned (=> ABSENT).
    """
    if field in _SIGNED_METRICS:
        return 0.0

    values = [
        float(getattr(m, field))
        for m in history
        if getattr(m, field, None) is not None
    ]
    if not values:
        return None
    return _median(values)


def _median(values: Sequence[float]) -> float:
    """Median of a non-empty numeric sequence (deterministic, no imports)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
