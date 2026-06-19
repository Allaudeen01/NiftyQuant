"""FeatureEngine: deterministic feature computation from market events.

Maintains a bounded rolling candle buffer per symbol and recomputes the
configured indicators on each new candle (bounded window => bounded cost). It
also folds in the latest option-chain-derived features (PCR, max pain, ATM IV,
IV rank, gamma exposure) for the matching underlying.

Determinism: given the same event stream and the same :class:`FeatureConfig`
(hence the same ``FEATURE_VERSION``), the produced :class:`FeatureVector`
sequence is identical. That is what makes backtests reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from nifty_quant.analytics import indicators as ind
from nifty_quant.analytics import options as opt
from nifty_quant.features.vector import FeatureVector
from nifty_quant.feed.events import CandleEvent, OptionChainEvent

FEATURE_VERSION = "v1"


@dataclass(frozen=True)
class FeatureConfig:
    """Definition of the feature set. Change => bump ``version``."""

    version: str = FEATURE_VERSION
    ema_periods: tuple[int, ...] = (20, 50, 200)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    buffer_size: int = 300
    iv_history_len: int = 252
    risk_free_rate: float = 0.065

    def min_buffer(self) -> int:
        needed = max(
            (*self.ema_periods, self.rsi_period, self.macd_slow,
             self.atr_period, self.adx_period, self.bb_period)
        )
        return max(self.buffer_size, needed + 5)


@dataclass
class _SymbolState:
    candles: deque = field(default_factory=lambda: deque())
    option_features: dict[str, float] = field(default_factory=dict)
    iv_history: deque = field(default_factory=lambda: deque())


class FeatureEngine:
    """Turns market events into versioned :class:`FeatureVector` objects."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._state: dict[str, _SymbolState] = {}

    @property
    def version(self) -> str:
        return self.config.version

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(candles=deque(maxlen=self.config.min_buffer()))
            self._state[symbol] = st
        return st

    # --- option features ----------------------------------------------------

    def on_option_chain(self, event: OptionChainEvent) -> None:
        """Update the latest option-derived features for the underlying."""
        chain = event.chain
        st = self._state_for(chain.underlying)
        cfg = self.config

        feats: dict[str, float] = {}
        try:
            feats["pcr_oi"] = opt.put_call_ratio(chain, by="oi")
        except Exception:  # pragma: no cover - defensive on sparse chains
            pass
        try:
            feats["max_pain"] = opt.max_pain(chain)
        except Exception:  # pragma: no cover
            pass

        atm_iv = opt.atm_iv(chain, r=cfg.risk_free_rate)
        if atm_iv is not None:
            feats["atm_iv"] = atm_iv
            feats["iv_rank"] = opt.iv_rank(atm_iv, list(st.iv_history))
            st.iv_history.append(atm_iv)
            while len(st.iv_history) > cfg.iv_history_len:
                st.iv_history.popleft()

        try:
            gex = opt.gamma_exposure(chain, r=cfg.risk_free_rate)
            feats["gamma_exposure"] = gex.total_gex
        except Exception:  # pragma: no cover
            pass

        st.option_features = feats

    # --- candle features ----------------------------------------------------

    def on_candle(self, event: CandleEvent) -> FeatureVector:
        """Append a candle and emit the current feature vector."""
        st = self._state_for(event.symbol)
        st.candles.append(event.candle)

        values = self._compute_candle_features(st)
        # Merge the latest option features for this symbol.
        values.update(st.option_features)

        return FeatureVector(
            timestamp=event.timestamp,
            symbol=event.symbol,
            version=self.config.version,
            values=values,
        )

    def _compute_candle_features(self, st: _SymbolState) -> dict[str, float]:
        cfg = self.config
        candles = list(st.candles)
        df = pd.DataFrame(
            {
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
            },
            index=pd.DatetimeIndex([c.timestamp for c in candles]),
        )
        close = df["close"]
        out: dict[str, float] = {"close": float(close.iloc[-1])}

        for p in cfg.ema_periods:
            out[f"ema_{p}"] = _last(ind.ema(close, p))
        out[f"rsi_{cfg.rsi_period}"] = _last(ind.rsi(close, cfg.rsi_period))

        macd = ind.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        out["macd_line"] = _last(macd["macd"])
        out["macd_signal"] = _last(macd["signal"])
        out["macd_hist"] = _last(macd["hist"])

        out[f"atr_{cfg.atr_period}"] = _last(ind.atr(df, cfg.atr_period))

        adx = ind.adx(df, cfg.adx_period)
        out[f"adx_{cfg.adx_period}"] = _last(adx["adx"])
        out["plus_di"] = _last(adx["plus_di"])
        out["minus_di"] = _last(adx["minus_di"])

        bb = ind.bollinger_bands(close, cfg.bb_period)
        out["bb_bandwidth"] = _last(bb["bandwidth"])

        return out


def _last(series: pd.Series) -> float:
    """Last value of a series as a float (NaN if empty/NaN)."""
    if len(series) == 0:
        return float("nan")
    return float(series.iloc[-1])
