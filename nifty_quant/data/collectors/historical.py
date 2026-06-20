"""Historical data collector.

Bridges a :class:`MarketDataProvider` (the source, e.g. Groww) and a
:class:`Storage` (the sink, e.g. Parquet). It contains no broker- or
format-specific logic -- both sides are interfaces -- so it works unchanged
against any provider/backend combination.
"""

from __future__ import annotations

from datetime import date

from nifty_quant.data.providers.base import MarketDataProvider
from nifty_quant.data.storage.base import Storage
from nifty_quant.log import get_logger

_log = get_logger("collectors.historical")


class HistoricalCollector:
    """Fetch historical candles / option chains and persist them."""

    def __init__(self, provider: MarketDataProvider, storage: Storage) -> None:
        self.provider = provider
        self.storage = storage

    def collect_candles(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> int:
        """Fetch candles for [start, end] and persist them. Returns count."""
        series = self.provider.get_ohlcv(symbol, timeframe, start, end)
        count = self.storage.write_candles(series)
        _log.event(
            "candles_collected",
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            count=count,
        )
        return count

    def collect_option_chain(self, underlying: str, expiry: date) -> int:
        """Fetch the current option-chain snapshot and persist it. Returns rows."""
        chain = self.provider.get_option_chain(underlying, expiry)
        rows = self.storage.write_option_chain(chain)
        _log.event(
            "option_chain_collected",
            underlying=underlying,
            expiry=expiry.isoformat(),
            rows=rows,
        )
        return rows
