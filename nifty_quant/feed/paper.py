"""PaperFeed: forward-in-time feed that polls a live provider.

Polls a :class:`MarketDataProvider` on a fixed interval and emits the same
:class:`CandleEvent` / :class:`OptionChainEvent` objects a backtest replay
produces -- so the identical engine, feature pipeline, risk gate, strategy, and
validation run forward in real time with no code changes.

Two safety/utility properties:
- **Persistence by default**: if a :class:`Storage` is provided, everything
  polled is written (idempotently), so each session grows the historical
  dataset for future backtests.
- **No live orders**: paper trading pairs this feed with a SimulatedBroker. The
  real broker's order gate is untouched and stays shut.

Robustness: a failing poll is logged and skipped; the loop keeps running unless
``max_consecutive_errors`` is exceeded.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from nifty_quant.data.providers.base import MarketDataProvider
from nifty_quant.data.storage.base import Storage
from nifty_quant.feed.base import MarketFeed
from nifty_quant.feed.clock import Clock, RealClock
from nifty_quant.feed.events import CandleEvent, OptionChainEvent
from nifty_quant.log import get_logger

_log = get_logger("feed.paper")


class PaperFeed(MarketFeed):
    """Polls a provider and emits live market events."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        candle_specs: list[tuple[str, str]] | None = None,
        chain_specs: list[tuple[str, date]] | None = None,
        poll_interval_seconds: float = 60.0,
        clock: Clock | None = None,
        storage: Storage | None = None,
        lookback_days: int = 5,
        emit_initial_history: bool = True,
        max_consecutive_errors: int = 5,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.candle_specs = list(candle_specs or [])
        self.chain_specs = list(chain_specs or [])
        self.poll_interval = poll_interval_seconds
        self.clock = clock or RealClock()
        self.storage = storage
        self.lookback_days = lookback_days
        self.emit_initial_history = emit_initial_history
        self.max_consecutive_errors = max_consecutive_errors

        self._last_candle_ts: dict[tuple[str, str], datetime] = {}
        self._first_poll = True
        self._stopped = False
        self._consecutive_errors = 0

    def stop(self) -> None:
        """Request a graceful stop after the current poll."""
        self._stopped = True

    # --- polling ------------------------------------------------------------

    def poll_once(self) -> int:
        """Run a single poll cycle. Returns the number of events emitted."""
        emitted = 0
        emit_history = self.emit_initial_history or not self._first_poll

        for symbol, timeframe in self.candle_specs:
            emitted += self._poll_candles(symbol, timeframe, emit_history)
        for underlying, expiry in self.chain_specs:
            emitted += self._poll_chain(underlying, expiry)

        self._first_poll = False
        return emitted

    def _poll_candles(self, symbol: str, timeframe: str, emit_history: bool) -> int:
        end = self.clock.now().date()
        start = end - timedelta(days=self.lookback_days)
        series = self.provider.get_ohlcv(symbol, timeframe, start, end)
        if self.storage is not None and len(series):
            self.storage.write_candles(series)

        key = (symbol, timeframe)
        last_ts = self._last_candle_ts.get(key)
        count = 0
        for candle in series.candles:
            is_new = last_ts is None or candle.timestamp > last_ts
            if not is_new:
                continue
            # On the very first poll, only emit history if asked.
            if self._first_poll and not emit_history:
                continue
            self._dispatch(
                CandleEvent(
                    timestamp=candle.timestamp,
                    symbol=symbol,
                    timeframe=timeframe,
                    candle=candle,
                )
            )
            count += 1
        if series.candles:
            self._last_candle_ts[key] = series.candles[-1].timestamp
        return count

    def _poll_chain(self, underlying: str, expiry: date) -> int:
        chain = self.provider.get_option_chain(underlying, expiry)
        if self.storage is not None and chain.quotes:
            self.storage.write_option_chain(chain)
        self._dispatch(OptionChainEvent(timestamp=chain.timestamp, chain=chain))
        return 1

    # --- run loop -----------------------------------------------------------

    def run(self, max_polls: int | None = None) -> int:
        """Poll repeatedly until stopped / max_polls reached. Returns total events.

        ``max_polls=None`` runs until :meth:`stop` is called (production use with
        a real clock). Tests pass a finite ``max_polls`` with a ManualClock.
        """
        _log.event(
            "paper_feed_started",
            poll_interval=self.poll_interval,
            candle_specs=[f"{s}:{t}" for s, t in self.candle_specs],
            chain_specs=[f"{u}:{e.isoformat()}" for u, e in self.chain_specs],
            persisted=self.storage is not None,
        )
        total = 0
        polls = 0
        while not self._stopped:
            try:
                total += self.poll_once()
                self._consecutive_errors = 0
            except Exception as exc:  # keep the live loop alive
                self._consecutive_errors += 1
                _log.event(
                    "paper_feed_poll_error",
                    level=40,  # ERROR
                    error=str(exc),
                    consecutive=self._consecutive_errors,
                )
                if self._consecutive_errors >= self.max_consecutive_errors:
                    _log.event("paper_feed_aborted_errors", level=40)
                    break

            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            if not self._stopped:
                self.clock.sleep(self.poll_interval)

        _log.event("paper_feed_stopped", polls=polls, events=total)
        return total
