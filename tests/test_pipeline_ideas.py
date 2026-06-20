"""Unit tests for Research_Idea information-gain estimation and prioritization.

Covers ``estimate_information_gain`` and ``prioritize`` in
``nifty_quant.research.pipeline.ideas`` (Req 10.3, 10.4).
"""

from dataclasses import replace

from nifty_quant.research.pipeline.ideas import (
    estimate_information_gain,
    prioritize,
)
from nifty_quant.research.pipeline.models import ComparisonResult, ResearchIdea


def _comparison(collected: int, min_sample: int = 20) -> ComparisonResult:
    return ComparisonResult(
        collected_session_count=collected,
        prior_session_count=max(0, collected - 1),
        unusual_events=[],
        min_sample_size=min_sample,
        history_available=collected > 1,
    )


def _idea(occurrence_count, total_sessions, gain=0.0) -> ResearchIdea:
    return ResearchIdea(
        text="Investigate something.",
        occurrence_count=occurrence_count,
        total_sessions=total_sessions,
        information_gain=gain,
    )


# --- estimate_information_gain ----------------------------------------------


def test_gain_is_normalised_between_zero_and_one():
    idea = _idea(occurrence_count=5, total_sessions=20)
    gain = estimate_information_gain(idea, _comparison(20), [])
    assert 0.0 <= gain <= 1.0


def test_rarer_pattern_scores_higher():
    common = _idea(occurrence_count=18, total_sessions=20)
    rare = _idea(occurrence_count=1, total_sessions=20)
    comparison = _comparison(20)
    assert estimate_information_gain(rare, comparison, []) > estimate_information_gain(
        common, comparison, []
    )


def test_better_sampled_scores_higher_for_same_rarity():
    idea = _idea(occurrence_count=1, total_sessions=20)
    thin = estimate_information_gain(idea, _comparison(5, min_sample=20), [])
    thick = estimate_information_gain(idea, _comparison(20, min_sample=20), [])
    assert thick > thin


def test_unknown_occurrence_uses_neutral_rarity():
    idea = _idea(occurrence_count=None, total_sessions=20)
    # collected == min_sample => sample_confidence == 1.0; rarity == 0.5 prior.
    gain = estimate_information_gain(idea, _comparison(20, min_sample=20), [])
    assert gain == 0.7 * 0.5 + 0.3 * 1.0


def test_deterministic_repeated_calls():
    idea = _idea(occurrence_count=3, total_sessions=15)
    comparison = _comparison(15)
    first = estimate_information_gain(idea, comparison, [])
    second = estimate_information_gain(idea, comparison, [])
    assert first == second


def test_history_used_when_comparison_reports_zero_collected():
    idea = _idea(occurrence_count=1, total_sessions=10)
    history = [None] * 10  # only length is consulted as a fallback count
    with_history = estimate_information_gain(idea, _comparison(0, min_sample=20), history)
    without_history = estimate_information_gain(idea, _comparison(0, min_sample=20), [])
    assert with_history > without_history


# --- prioritize -------------------------------------------------------------


def test_prioritize_orders_by_descending_gain():
    ideas = [_idea(1, 10, gain=0.2), _idea(1, 10, gain=0.9), _idea(1, 10, gain=0.5)]
    ranked = prioritize(ideas)
    assert [i.information_gain for i in ranked] == [0.9, 0.5, 0.2]
    assert [i.priority for i in ranked] == [1, 2, 3]


def test_prioritize_assigns_contiguous_priorities():
    ideas = [_idea(1, 10, gain=g) for g in (0.1, 0.4, 0.4, 0.7)]
    ranked = prioritize(ideas)
    assert [i.priority for i in ranked] == [1, 2, 3, 4]


def test_prioritize_stable_on_ties():
    a = replace(_idea(1, 10, gain=0.5), text="A")
    b = replace(_idea(1, 10, gain=0.5), text="B")
    c = replace(_idea(1, 10, gain=0.5), text="C")
    ranked = prioritize([a, b, c])
    # Equal gains keep original input order.
    assert [i.text for i in ranked] == ["A", "B", "C"]


def test_prioritize_does_not_mutate_inputs():
    ideas = [_idea(1, 10, gain=0.5), _idea(1, 10, gain=0.9)]
    prioritize(ideas)
    assert all(i.priority is None for i in ideas)


def test_prioritize_empty_returns_empty():
    assert prioritize([]) == []
