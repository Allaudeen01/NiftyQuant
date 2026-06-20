"""Market regime adapter (Req 20) -- a thin view over the frozen classifier.

This module supplies the report's first section (the **Market Regime**). It is
deliberately *not* a regime classifier of its own: trend and volatility come
straight from the existing, frozen
:func:`nifty_quant.research.regime.classify_regime` (Req 20.3). The only labels
this adapter computes itself are the two option-derived signals the pipeline has
already produced and carried on the :class:`MarketSummary`:

* the **gamma regime** -- a label for the sign of the session's total gamma
  exposure (``gamma_sign``); and
* the **PCR level** -- a label bucketing the put/call ratio (``pcr``).

Any classification that cannot be computed -- because ``classify_regime`` could
not run, or because the underlying option metric is unavailable -- is recorded
as ``None`` so the renderer marks that field ``unavailable`` (Req 20.4). The
adapter performs no regime math beyond the gamma/PCR labelling.

Thresholds (documented and fixed so the labelling is deterministic):

================  ========================================================
gamma regime      ``gamma_sign`` ``+1`` -> ``"bullish"``, ``0`` ->
                  ``"neutral"``, ``-1`` -> ``"bearish"``; ``None`` ->
                  ``None`` (unavailable)
PCR level         ``pcr < PCR_LOW`` -> ``"low"``; ``PCR_LOW <= pcr <=
                  PCR_HIGH`` -> ``"neutral"``; ``pcr > PCR_HIGH`` ->
                  ``"high"``; ``None`` -> ``None`` (unavailable)
================  ========================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nifty_quant.research.pipeline.models import MarketRegime
from nifty_quant.research.regime import RegimeConfig, classify_regime

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard pandas import here
    import pandas as pd

    from nifty_quant.research.pipeline.models import MarketSummary

# PCR bucket boundaries. Below ``PCR_LOW`` is call-heavy ("low"); above
# ``PCR_HIGH`` is put-heavy ("high"); the inclusive band between them is
# "neutral". Fixed module constants so the labelling is deterministic.
PCR_LOW = 0.9
PCR_HIGH = 1.1

# Map the sign of total gamma exposure onto a regime label (Req 20.2).
_GAMMA_REGIME = {1: "bullish", 0: "neutral", -1: "bearish"}


def _gamma_regime(gamma_sign: int | None) -> str | None:
    """Label the gamma-exposure sign; ``None`` when the sign is unavailable."""
    if gamma_sign is None:
        return None
    return _GAMMA_REGIME.get(gamma_sign)


def _pcr_level(pcr: float | None) -> str | None:
    """Bucket the put/call ratio; ``None`` when PCR is unavailable."""
    if pcr is None:
        return None
    if pcr < PCR_LOW:
        return "low"
    if pcr > PCR_HIGH:
        return "high"
    return "neutral"


def market_regime(
    candle_history: "pd.Series",
    summary: "MarketSummary",
    config: RegimeConfig | None = None,
) -> MarketRegime:
    """Build the report's Market Regime view (Req 20).

    Reuses :func:`classify_regime` for ``trend``/``volatility`` (Req 20.3) and
    labels the option-derived ``gamma_sign``/``pcr`` carried on ``summary``
    into a gamma regime and a PCR level (Req 20.2). Any classification that
    cannot be computed -- a failing/empty ``classify_regime`` call or a missing
    option metric -- is recorded as ``None`` so the renderer marks it
    ``unavailable`` (Req 20.4).
    """
    trend: str | None = None
    volatility: str | None = None
    if candle_history is not None and len(candle_history) > 0:
        try:
            regime = classify_regime(candle_history, config or RegimeConfig())
        except Exception:
            # No trend/volatility math here: on any failure leave both
            # unavailable for the renderer (Req 20.3/20.4).
            trend = None
            volatility = None
        else:
            trend = regime.trend
            volatility = regime.volatility

    metrics = summary.metrics
    return MarketRegime(
        trend=trend,
        volatility=volatility,
        gamma_regime=_gamma_regime(metrics.gamma_sign),
        pcr_level=_pcr_level(metrics.pcr),
    )
