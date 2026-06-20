"""Overall Research Confidence Score (Req 21) -- pure, deterministic scoring.

This module computes the single ``Research_Confidence_Score`` reported for the
Target_Session. The score combines five normalised factors -- data quality,
sample size, feature drift, validation alerts, and evidence maturity (Req 21.2)
-- into one integer percentage in ``0..100`` (Req 21.1) using fixed configured
weights.

The computation is a **pure function** of values already produced earlier in
the run (``Quality_Score``, ``Collected_Session_Count``, validation severities,
evidence maturity). It consumes no wall-clock time, no memory cost, and no
randomness, so running the pipeline twice on identical inputs yields an
identical score (Req 21.4).

Factor normalisation (each mapped onto ``0..100``):

==================  =========================================================
``data_quality``    the ``Quality_Score`` as-is (already ``0..100``)
``sample_size``     ``min(100, round(100 * collected / min_sample_size))``;
                    ``0`` when ``min_sample_size <= 0`` cannot be evaluated
``feature_drift``   inverse of drift severity: ``100 - round(severity * 100)``
``validation``      inverse of alert severity: ``100 - round(severity * 100)``
``evidence_maturity`` the maturity value clamped into ``0..100``
==================  =========================================================

``drift_severity`` and ``alert_severity`` are expected on a normalised ``0.0``
(no drift / no alerts -- best) to ``1.0`` (maximally degraded -- worst) scale;
values are multiplied by :data:`SEVERITY_SCALE` and inverted so that low
severity yields a high factor. ``evidence_maturity`` is likewise expected on a
``0..100`` scale and is simply clamped.

Dataclasses mirror the existing ``nifty_quant`` style: ``from __future__ import
annotations`` plus frozen dataclasses holding only typed values.
"""

from __future__ import annotations

from dataclasses import dataclass

# Multiplier mapping a normalised ``0.0..1.0`` severity onto the ``0..100`` factor
# scale before inversion. Defined as a module constant so the normalisation is
# explicit and deterministic.
SEVERITY_SCALE = 100.0


@dataclass(frozen=True)
class ConfidenceConfig:
    """Per-factor weights for the Research_Confidence_Score (Req 21.2).

    The five weights sum to 100; the score is the weighted average of the five
    normalised factors. Frozen so the configuration is immutable and the
    computation stays deterministic (Req 21.4).
    """

    w_data_quality: int = 30       # from Quality_Score
    w_sample_size: int = 20        # from Collected_Session_Count vs min_sample_size
    w_feature_drift: int = 20      # inverse of validation drift severity
    w_validation: int = 15         # inverse of validation alert severity
    w_evidence_maturity: int = 15  # from spread/age of Evidence_Scores


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Per-factor normalised contributions shown in the report (Req 21.3)."""

    data_quality: int
    sample_size: int
    feature_drift: int
    validation: int
    evidence_maturity: int


@dataclass(frozen=True)
class ConfidenceResult:
    """Research_Confidence_Score plus its contributing factors (Req 21.1/21.3)."""

    score: int                            # Research_Confidence_Score, 0..100
    breakdown: ConfidenceBreakdown
    factors: list[tuple[str, int, int]]   # (factor_name, factor_value, weight) for rendering


def _clamp_score(value: float) -> int:
    """Round ``value`` to the nearest int and clamp it into ``0..100``."""
    return max(0, min(100, round(value)))


def _inverse_severity(severity: float) -> int:
    """Map a normalised severity onto an inverted ``0..100`` factor.

    Low severity (good) yields a high factor; high severity (bad) yields a low
    factor. ``severity`` is multiplied by :data:`SEVERITY_SCALE`, subtracted
    from 100, then clamped.
    """
    return _clamp_score(100.0 - severity * SEVERITY_SCALE)


def compute_confidence(
    quality_score: int,
    collected_session_count: int,
    min_sample_size: int,
    drift_severity: float,
    alert_severity: float,
    evidence_maturity: float,
    config: ConfidenceConfig,
) -> ConfidenceResult:
    """Compute the Research_Confidence_Score (Req 21).

    Pure, deterministic weighted average of the five normalised factors, each in
    ``0..100``, using the weights in ``config``. The result score is rounded to
    an int and clamped into ``0..100`` (Req 21.1). Because every input is a value
    already computed earlier in the run and no clock/memory/randomness is
    consulted, identical inputs always produce an identical result (Req 21.4).

    Factor normalisation is documented at module level; ``min_sample_size <= 0``
    is guarded by treating the sample-size factor as ``0`` (cannot be evaluated).
    """
    # --- Normalise each factor into 0..100 ---------------------------------
    data_quality = _clamp_score(quality_score)

    if min_sample_size <= 0:
        sample_size = 0
    else:
        sample_size = min(
            100, _clamp_score(100.0 * collected_session_count / min_sample_size)
        )

    feature_drift = _inverse_severity(drift_severity)
    validation = _inverse_severity(alert_severity)
    evidence = _clamp_score(evidence_maturity)

    breakdown = ConfidenceBreakdown(
        data_quality=data_quality,
        sample_size=sample_size,
        feature_drift=feature_drift,
        validation=validation,
        evidence_maturity=evidence,
    )

    # --- Weighted average using the configured weights ---------------------
    factors: list[tuple[str, int, int]] = [
        ("data_quality", data_quality, config.w_data_quality),
        ("sample_size", sample_size, config.w_sample_size),
        ("feature_drift", feature_drift, config.w_feature_drift),
        ("validation", validation, config.w_validation),
        ("evidence_maturity", evidence, config.w_evidence_maturity),
    ]

    total_weight = sum(weight for _, _, weight in factors)
    if total_weight <= 0:
        score = 0
    else:
        weighted_sum = sum(value * weight for _, value, weight in factors)
        score = _clamp_score(weighted_sum / total_weight)

    return ConfidenceResult(score=score, breakdown=breakdown, factors=factors)
