"""Experiment Registry -- encode/decode an ``ExperimentRecord`` into the journal.

The expanded ``Experiment_Record`` provenance (Req 23) records eight fields for
a unit of research work: the research question, the hypothesis, three version
stamps (dataset / feature / code), the result, the decision taken, and the next
action. Rather than introducing a separate experiment store (Req 23.2), the
record is folded into the *existing* ``Hypothesis`` so the ``ResearchJournal``
stays the single source of truth and is persisted only through its frozen
``add`` / ``update`` / ``list`` interface.

The mapping mirrors the "Experiment_Record persistence model" table in the
design:

==================  =============================  ============
ExperimentRecord    Hypothesis carrier             Encoding
==================  =============================  ============
research_question   tag ``exp.rq``                 base64url
hypothesis          native ``Hypothesis.hypothesis``  plain
dataset_version     tag ``exp.dataset``            plain
feature_version     tag ``exp.feature``            plain
code_version        tag ``exp.code``               plain
result              tag ``exp.result``             base64url
decision            tag ``exp.decision``           plain
next_action         tag ``exp.next``               base64url
==================  =============================  ============

Free-text fields (research_question, result, next_action) may contain arbitrary
characters -- newlines, colons, commas -- that do not survive a flat tag list
cleanly, so they are base64url-encoded (URL-safe alphabet, padding stripped) and
decoded as the exact inverse, guaranteeing a lossless round-trip (Property 26).
The short version/decision tokens are stored plainly; the decoder splits each
tag on its first ``:`` only, so even a value that itself contains ``:`` is
recovered unchanged.

This module is pure (no I/O): callers persist the returned
``(hypothesis_text, tags)`` through ``ResearchJournal.add``/``update`` and read
back a ``Hypothesis`` to hand to :func:`decode_experiment`.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Sequence

from nifty_quant.research.pipeline.models import ExperimentRecord

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from nifty_quant.research.journal import Hypothesis


# Structured tag prefixes for the seven tag-carried fields (the eighth field,
# ``hypothesis``, lives in the native ``Hypothesis.hypothesis`` slot).
_RQ_PREFIX = "exp.rq:"
_DATASET_PREFIX = "exp.dataset:"
_FEATURE_PREFIX = "exp.feature:"
_CODE_PREFIX = "exp.code:"
_RESULT_PREFIX = "exp.result:"
_DECISION_PREFIX = "exp.decision:"
_NEXT_PREFIX = "exp.next:"

# Fields whose free text is base64url-encoded so it survives the tag list.
_B64_PREFIXES = frozenset({_RQ_PREFIX, _RESULT_PREFIX, _NEXT_PREFIX})

# All experiment-tag prefixes, used when re-encoding to drop stale exp.* tags.
_ALL_EXP_PREFIXES = (
    _RQ_PREFIX,
    _DATASET_PREFIX,
    _FEATURE_PREFIX,
    _CODE_PREFIX,
    _RESULT_PREFIX,
    _DECISION_PREFIX,
    _NEXT_PREFIX,
)


def encode_experiment(record: ExperimentRecord) -> tuple[str, list[str]]:
    """Encode an ``ExperimentRecord`` into ``(hypothesis_text, tags)``.

    Returns the text destined for the native ``Hypothesis.hypothesis`` field and
    the seven structured ``exp.*`` tags carrying the remaining fields. Free-text
    fields are base64url-encoded; version/decision fields are stored plainly.
    The tags are emitted in a fixed order so encoding is deterministic.

    Callers pass ``hypothesis_text`` and ``tags`` to ``ResearchJournal.add`` (for
    a new experiment) or ``ResearchJournal.update`` (to refresh provenance on an
    existing hypothesis). To preserve non-experiment tags (e.g. ``evidence:<NN>``)
    on an update, merge the returned tags with the existing ones via
    :func:`merge_experiment_tags`.
    """
    tags = [
        f"{_RQ_PREFIX}{_b64_encode(record.research_question)}",
        f"{_DATASET_PREFIX}{record.dataset_version}",
        f"{_FEATURE_PREFIX}{record.feature_version}",
        f"{_CODE_PREFIX}{record.code_version}",
        f"{_RESULT_PREFIX}{_b64_encode(record.result)}",
        f"{_DECISION_PREFIX}{record.decision}",
        f"{_NEXT_PREFIX}{_b64_encode(record.next_action)}",
    ]
    return record.hypothesis, tags


def decode_experiment(h: "Hypothesis") -> ExperimentRecord:
    """Decode a ``Hypothesis`` back into an ``ExperimentRecord``.

    Reads the native ``hypothesis`` text and the seven ``exp.*`` tags, reversing
    the base64url encoding of the free-text fields. Any missing tag decodes to an
    empty string, so the function is total over every ``Hypothesis``. This is the
    exact inverse of :func:`encode_experiment` for any record it produced.
    """
    tags = [str(t) for t in (getattr(h, "tags", None) or [])]
    return ExperimentRecord(
        research_question=_b64_decode(_tag_value(tags, _RQ_PREFIX)),
        hypothesis=str(getattr(h, "hypothesis", "")),
        dataset_version=_tag_value(tags, _DATASET_PREFIX),
        feature_version=_tag_value(tags, _FEATURE_PREFIX),
        code_version=_tag_value(tags, _CODE_PREFIX),
        result=_b64_decode(_tag_value(tags, _RESULT_PREFIX)),
        decision=_tag_value(tags, _DECISION_PREFIX),
        next_action=_b64_decode(_tag_value(tags, _NEXT_PREFIX)),
    )


def merge_experiment_tags(
    existing: "Sequence[str] | None", record: ExperimentRecord
) -> list[str]:
    """Merge experiment tags for ``record`` onto ``existing`` non-experiment tags.

    Drops any stale ``exp.*`` tag from ``existing`` (preserving the order of the
    remaining tags, e.g. an ``evidence:<NN>`` tag) and appends the freshly encoded
    experiment tags. Convenience for refreshing provenance via
    ``ResearchJournal.update`` without clobbering unrelated tags.
    """
    _, exp_tags = encode_experiment(record)
    kept = [
        str(t)
        for t in (existing or [])
        if not any(str(t).startswith(p) for p in _ALL_EXP_PREFIXES)
    ]
    return kept + exp_tags


# --- Internal helpers --------------------------------------------------------


def _tag_value(tags: Sequence[str], prefix: str) -> str:
    """Return the value of the first ``<prefix><value>`` tag, or ``""``.

    Splits on the prefix only, so a value that itself contains ``:`` is returned
    unchanged.
    """
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return ""


def _b64_encode(text: str) -> str:
    """URL-safe base64-encode ``text`` (UTF-8), stripping ``=`` padding."""
    encoded = base64.urlsafe_b64encode((text or "").encode("utf-8"))
    return encoded.decode("ascii").rstrip("=")


def _b64_decode(token: str) -> str:
    """Inverse of :func:`_b64_encode`: restore padding and UTF-8 decode."""
    if not token:
        return ""
    padding = "=" * (-len(token) % 4)
    decoded = base64.urlsafe_b64decode(token.encode("ascii") + padding.encode("ascii"))
    return decoded.decode("utf-8")
