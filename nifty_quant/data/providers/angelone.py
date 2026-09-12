"""Angel One SmartAPI provider.

Maps Angel One's SmartConnect SDK (``smartapi-python``) onto the platform's
provider-agnostic interfaces. As with the Groww provider, nothing outside this
module imports the SDK or sees an Angel-specific payload -- the rest of the
system only sees :class:`OHLCVSeries` / :class:`OptionChain` / order models.

Design (identical philosophy to GrowwProvider):
- SDK imported lazily, so the package imports without ``smartapi-python``.
- The authenticated SmartConnect client is injected; tests pass a fake.
- ``from_env`` builds and logs in using environment variables.
- Order methods are gated behind ``live_trading_enabled`` (default False).

Historical candles are the first-class capability here (your starting point).
SmartAPI caps the date range per ``getCandleData`` call by interval, so
:meth:`get_ohlcv` automatically chunks long ranges and concatenates them.

Auth env vars (never hard-code):
    ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_MPIN (or ANGEL_PASSWORD),
    ANGEL_TOTP_SECRET
"""

from __future__ import annotations

import binascii
import os
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from nifty_quant.data.models import Candle, OHLCVSeries, OptionChain, OptionQuote
from nifty_quant.data.providers.base import (
    BrokerProvider,
    OrderKind,
    OrderRequest,
    OrderResult,
    OrderSide,
)
from nifty_quant.log import get_logger

_log = get_logger("providers.angelone")

# Our timeframe -> Angel interval constant.
_TIMEFRAME_TO_INTERVAL: dict[str, str] = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "60m": "ONE_HOUR",
    "1d": "ONE_DAY",
}

# Max days per getCandleData request, by Angel interval.
_MAX_DAYS_PER_REQUEST: dict[str, int] = {
    "ONE_MINUTE": 30,
    "THREE_MINUTE": 60,
    "FIVE_MINUTE": 100,
    "TEN_MINUTE": 100,
    "FIFTEEN_MINUTE": 200,
    "THIRTY_MINUTE": 200,
    "ONE_HOUR": 400,
    "ONE_DAY": 2000,
}

# Well-known NSE index tokens (avoid needing the full scrip master just to
# pull index history). Override/extend via the constructor.
_DEFAULT_TOKENS: dict[str, str] = {
    "NIFTY": "99926000",       # Nifty 50
    "BANKNIFTY": "99926009",   # Nifty Bank
}


class LiveTradingDisabledError(RuntimeError):
    """Raised when an order method is called while live trading is disabled."""


class AngelOneProvider(BrokerProvider):
    """Concrete :class:`BrokerProvider` backed by Angel One SmartConnect."""

    def __init__(
        self,
        client: Any,
        *,
        exchange: str = "NSE",
        symbol_tokens: dict[str, str] | None = None,
        live_trading_enabled: bool = False,
        request_pause: float = 0.34,
        instrument_master: Any = None,
        spot_cache_ttl: float = 30.0,
    ) -> None:
        """Wrap an already-authenticated SmartConnect client.

        ``request_pause`` is the minimum interval (seconds) enforced between
        ANY two SDK calls (global pacing to respect SmartAPI rate limits; set 0
        in tests). ``spot_cache_ttl`` briefly caches the spot so building
        several option chains in one polling cycle does not re-fetch it.
        ``instrument_master`` resolves option tokens; created lazily if absent.
        """
        self._client = client
        self.exchange = exchange
        self.symbol_tokens = {**_DEFAULT_TOKENS, **(symbol_tokens or {})}
        self.live_trading_enabled = live_trading_enabled
        self.request_pause = request_pause
        self._instrument_master = instrument_master
        self.spot_cache_ttl = spot_cache_ttl
        self._last_api_call = 0.0
        self._spot_cache: dict[str, tuple[float, float]] = {}

    def _throttle(self) -> None:
        """Enforce a minimum interval between consecutive SDK calls."""
        if self.request_pause <= 0:
            return
        wait = self.request_pause - (time.monotonic() - self._last_api_call)
        if wait > 0:
            time.sleep(wait)
        self._last_api_call = time.monotonic()

    # --- constructors / auth ------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AngelOneProvider":
        """Authenticate from environment variables and return a provider."""
        api_key = os.environ.get("ANGEL_API_KEY")
        client_code = os.environ.get("ANGEL_CLIENT_CODE")
        mpin = os.environ.get("ANGEL_MPIN") or os.environ.get("ANGEL_PASSWORD")
        totp_secret = os.environ.get("ANGEL_TOTP_SECRET")
        missing = [
            name for name, val in (
                ("ANGEL_API_KEY", api_key),
                ("ANGEL_CLIENT_CODE", client_code),
                ("ANGEL_MPIN/ANGEL_PASSWORD", mpin),
                ("ANGEL_TOTP_SECRET", totp_secret),
            ) if not val
        ]
        if missing:
            raise RuntimeError(f"missing Angel env vars: {', '.join(missing)}")

        SmartConnect = _import_smartconnect()
        try:
            import pyotp  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError("pyotp is required: pip install pyotp") from exc

        client = SmartConnect(api_key=api_key)
        totp = _generate_totp(totp_secret)
        session = client.generateSession(client_code, mpin, totp)
        if not session or not session.get("status", True):
            raise RuntimeError(f"Angel login failed: {session}")
        _log.event("angel_session_created", client_code=client_code)
        return cls(client, **kwargs)

    # --- read-only market data ---------------------------------------------

    def resolve_token(self, symbol: str) -> str:
        token = self.symbol_tokens.get(symbol.upper())
        if token is None:
            raise KeyError(
                f"no symbol token for {symbol!r}; pass it via symbol_tokens="
                "{'SYM': 'token'} (look it up in the Angel scrip master)."
            )
        return token

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        *,
        token: str | None = None,
    ) -> OHLCVSeries:
        """Historical candles, auto-chunked to respect per-request range caps."""
        if timeframe not in _TIMEFRAME_TO_INTERVAL:
            raise ValueError(
                f"unsupported timeframe {timeframe!r}; "
                f"valid: {sorted(_TIMEFRAME_TO_INTERVAL)}"
            )
        interval = _TIMEFRAME_TO_INTERVAL[timeframe]
        symbol_token = token or self.resolve_token(symbol)
        max_days = _MAX_DAYS_PER_REQUEST[interval]

        candles: list[Candle] = []
        seen: set[datetime] = set()
        for chunk_start, chunk_end in _chunk_ranges(start, end, max_days):
            params = {
                "exchange": self.exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": f"{chunk_start.isoformat()} 09:15",
                "todate": f"{chunk_end.isoformat()} 15:30",
            }
            self._throttle()
            resp = self._client.getCandleData(params)
            for row in (resp or {}).get("data", []) or []:
                candle = _map_candle(row)
                if candle.timestamp not in seen:
                    seen.add(candle.timestamp)
                    candles.append(candle)

        candles.sort(key=lambda c: c.timestamp)
        _log.event(
            "angel_candles_fetched",
            symbol=symbol, timeframe=timeframe, count=len(candles),
        )
        return OHLCVSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    def get_spot(self, symbol: str) -> float:
        """Live index spot via LTP (cached briefly), daily-candle as fallback.

        Uses ``getMarketData`` LTP on the index token for the TRUE live index
        level -- correct for moneyness / ATM selection / realized-move research.
        Falls back to the most recent daily-candle close only if the LTP call
        fails or returns nothing. Cached for ``spot_cache_ttl`` so several
        option chains in one polling cycle share one spot read.
        """
        key = symbol.upper()
        cached = self._spot_cache.get(key)
        now = time.monotonic()
        if cached is not None and (now - cached[0]) < self.spot_cache_ttl:
            return cached[1]

        spot = self._live_index_ltp(symbol)
        if spot is None:
            # Fallback: most recent daily candle close (lagged, last resort).
            end = datetime.now().date()
            start = end - timedelta(days=7)
            series = self.get_ohlcv(symbol, "1d", start, end)
            if not series.candles:
                raise RuntimeError(
                    f"no spot for {symbol!r} (LTP and daily candle both empty)"
                )
            spot = series.candles[-1].close
        self._spot_cache[key] = (now, spot)
        return spot

    def _live_index_ltp(self, symbol: str) -> float | None:
        """Live index LTP via getMarketData; None on any failure (caller falls back)."""
        try:
            token = self.resolve_token(symbol)
        except KeyError:
            return None
        try:
            self._throttle()
            resp = self._client.getMarketData("LTP", {self.exchange: [token]})
        except Exception:  # noqa: BLE001 - any failure -> fall back to candle
            return None
        data = (resp or {}).get("data", {}) if isinstance(resp, dict) else {}
        fetched = data.get("fetched", []) or []
        if not fetched:
            return None
        ltp = float(fetched[0].get("ltp", 0.0) or 0.0)
        return ltp if ltp > 0 else None

    def get_option_chain(self, underlying: str, expiry: date) -> OptionChain:
        """Build an option-chain snapshot from the instrument master + market data.

        Angel has no single chain endpoint, so we resolve the CE/PE contract
        tokens for (underlying, expiry) from the scrip master, then fetch live
        LTP / open interest / depth via ``getMarketData`` in batches of 50.
        Implied volatility is left unset; the analytics layer derives it from
        the mid price when needed.
        """
        master = self._get_instrument_master()
        instruments = master.option_instruments(underlying, expiry)
        if not instruments:
            raise RuntimeError(
                f"no {underlying} OPTIDX contracts for expiry {expiry.isoformat()} "
                "in the scrip master (check the expiry date / refresh the master)."
            )

        by_token = {ins.token: ins for ins in instruments}
        fetched = self._fetch_market_data([ins.token for ins in instruments])

        quotes: list[OptionQuote] = []
        for item in fetched:
            token = str(item.get("symbolToken") or item.get("token") or "")
            ins = by_token.get(token)
            if ins is None:
                continue
            bid, ask = _best_bid_ask(item)
            quotes.append(OptionQuote(
                strike=ins.strike,
                option_type=ins.option_type,
                expiry=expiry,
                last_price=float(item.get("ltp", 0.0) or 0.0),
                bid=bid,
                ask=ask,
                volume=float(item.get("tradeVolume", 0.0) or 0.0),
                open_interest=float(item.get("opnInterest", 0.0) or 0.0),
                implied_volatility=None,
            ))

        spot = self.get_spot(underlying)
        return OptionChain(
            underlying=underlying,
            spot=spot,
            expiry=expiry,
            timestamp=datetime.now(),
            quotes=tuple(quotes),
        )

    def get_india_vix(self) -> float | None:
        """Latest India VIX value via getMarketData LTP (None if unavailable)."""
        master = self._get_instrument_master()
        token = master.index_token("INDIA VIX", exch_seg="NSE")
        if token is None:
            return None
        self._throttle()
        resp = self._client.getMarketData("LTP", {"NSE": [token]})
        data = (resp or {}).get("data", {}) if isinstance(resp, dict) else {}
        fetched = data.get("fetched", []) or []
        if not fetched:
            return None
        return float(fetched[0].get("ltp", 0.0) or 0.0)

    def _get_instrument_master(self):
        if self._instrument_master is None:
            from nifty_quant.data.providers.angel_instruments import InstrumentMaster
            self._instrument_master = InstrumentMaster()
        return self._instrument_master

    def _fetch_market_data(self, tokens: list[str]) -> list[dict]:
        """Call getMarketData FULL in batches of <= 50 NFO tokens."""
        fetched: list[dict] = []
        for i in range(0, len(tokens), 50):
            batch = tokens[i:i + 50]
            self._throttle()
            resp = self._client.getMarketData("FULL", {"NFO": batch})
            data = (resp or {}).get("data", {}) if isinstance(resp, dict) else {}
            fetched.extend(data.get("fetched", []) or [])
        return fetched

    # --- dgp-v2 additions -------------------------------------------------
    # These are NEW methods rather than changes to the ones above, so every
    # existing dgp-v1 caller keeps its exact behaviour and byte-identical
    # semantics. See nifty_quant/data/dgp.py for the versioning contract.

    def _fetch_market_data_timed(
        self, tokens: list[str], *, batch_size: int = 50,
        failed_batches: list[int] | None = None,
    ) -> tuple[list[dict], dict[str, dict]]:
        """Batched fetch that records WHEN each batch actually returned.

        Returns ``(items, timing_by_token)`` where timing carries
        ``batch_index`` and ``observed_mono`` (a ``time.monotonic()`` reading,
        converted to wall-clock by the caller against a single per-poll anchor
        so an NTP step cannot make observations appear to move backwards).

        ``failed_batches`` is appended to rather than raised on, so one bad
        batch no longer discards the successful ones. Under dgp-v1 a single
        failing batch propagated out of the whole chain fetch and every batch
        was refetched after a 4-12s backoff, silently widening the gap between
        the recorded timestamp and the actual observation.
        """
        items: list[dict] = []
        timing: dict[str, dict] = {}
        for bi, i in enumerate(range(0, len(tokens), batch_size)):
            batch = tokens[i:i + batch_size]
            try:
                self._throttle()
                resp = self._client.getMarketData("FULL", {"NFO": batch})
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                _log.event("batch_fetch_failed", level=40, batch_index=bi,
                           n_tokens=len(batch), error=str(exc)[:120])
                if failed_batches is not None:
                    failed_batches.append(bi)
                continue
            observed_mono = time.monotonic()
            data = (resp or {}).get("data", {}) if isinstance(resp, dict) else {}
            got = data.get("fetched", []) or []
            items.extend(got)
            for it in got:
                tok = str(it.get("symbolToken") or it.get("token") or "")
                timing[tok] = {"batch_index": bi, "observed_mono": observed_mono}
        return items, timing

    def get_option_chain_timed(
        self, underlying: str, expiry: date, *,
        strike_band: tuple[float, float] | None = None,
        shuffle_seed: int | None = None,
        fetch_attempt: int = 0,
    ) -> tuple[list[OptionQuote], dict]:
        """dgp-v2 chain fetch: per-contract timing, pre-filtered band, shuffled order.

        Returns ``(quotes, meta)``. Every quote carries ``observed_ts`` as a
        monotonic reading (converted by the caller), plus ``batch_index``,
        ``fetch_rank``, ``token`` and ``trading_symbol``.

        Crucially this does NOT call ``get_spot()`` -- the caller observes spot
        before and after the whole poll. Under dgp-v1 spot was read last,
        inside this call, behind a cache, which made near-expiry quotes older
        than spot but far-expiry quotes NEWER than spot (the unsafe direction)
        and forced EXP034 to restrict itself to the near expiry.
        """
        master = self._get_instrument_master()
        instruments = master.option_instruments(
            underlying, expiry, strike_band=strike_band, shuffle_seed=shuffle_seed)
        if not instruments:
            raise RuntimeError(
                f"no {underlying} OPTIDX contracts for expiry {expiry.isoformat()} "
                f"within band {strike_band} in the scrip master.")

        by_token = {ins.token: ins for ins in instruments}
        rank_by_token = {ins.token: r for r, ins in enumerate(instruments)}
        failed: list[int] = []
        fetched, timing = self._fetch_market_data_timed(
            [ins.token for ins in instruments], failed_batches=failed)

        quotes: list[OptionQuote] = []
        for item in fetched:
            token = str(item.get("symbolToken") or item.get("token") or "")
            ins = by_token.get(token)
            if ins is None:
                continue
            bid, ask = _best_bid_ask(item)
            t = timing.get(token, {})
            quotes.append(OptionQuote(
                strike=ins.strike,
                option_type=ins.option_type,
                expiry=expiry,
                last_price=float(item.get("ltp", 0.0) or 0.0),
                bid=bid,
                ask=ask,
                volume=float(item.get("tradeVolume", 0.0) or 0.0),
                open_interest=float(item.get("opnInterest", 0.0) or 0.0),
                implied_volatility=None,
                observed_ts=t.get("observed_mono"),
                batch_index=t.get("batch_index"),
                fetch_rank=rank_by_token.get(token),
                fetch_attempt=fetch_attempt,
                token=token,
                trading_symbol=ins.trading_symbol,
            ))
        meta = {
            "n_instruments": len(instruments),
            "n_quotes": len(quotes),
            "failed_batches": failed,
            "n_batches": (len(instruments) + 49) // 50,
            "shuffle_seed": shuffle_seed,
            "strike_band": strike_band,
        }
        return quotes, meta

    def get_spot_observed(self, symbol: str) -> tuple[float | None, float, str]:
        """Spot with its observation time and provenance.

        Returns ``(value, observed_mono, source)``. ``source`` is
        ``live_ltp``, ``daily_close_fallback`` or ``unavailable``.

        dgp-v1's ``get_spot()`` returned a bare float and fell back silently to
        the last daily close when the live LTP call failed -- which pinned spot
        to a constant for the whole of 2026-06-26 and was not discovered until
        an audit three experiments later. Here the fallback is reported, never
        silent. The 30s cache is deliberately bypassed so that a pre-fetch and
        post-fetch reading are genuinely independent observations.
        """
        spot = self._live_index_ltp(symbol)
        if spot is not None and spot > 0:
            return float(spot), time.monotonic(), "live_ltp"
        try:
            end = datetime.now().date()
            series = self.get_ohlcv(symbol, "1d", end - timedelta(days=7), end)
            if series.candles:
                return float(series.candles[-1].close), time.monotonic(), "daily_close_fallback"
        except Exception as exc:  # noqa: BLE001
            _log.event("spot_fallback_failed", level=40, error=str(exc)[:120])
        return None, time.monotonic(), "unavailable"

    # --- execution (gated) --------------------------------------------------

    def place_order(self, order: OrderRequest) -> OrderResult:
        self._require_live_trading()
        params = {
            "variety": "NORMAL",
            "tradingsymbol": order.trading_symbol,
            "symboltoken": self.symbol_tokens.get(order.trading_symbol, ""),
            "transactiontype": order.side.value,
            "exchange": "NFO",
            "ordertype": (
                "LIMIT" if order.kind is OrderKind.LIMIT else "MARKET"
            ),
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": order.price or 0,
            "quantity": order.quantity,
        }
        resp = self._client.placeOrder(params)
        order_id = resp if isinstance(resp, str) else str(
            (resp or {}).get("data", {}).get("orderid", "")
        )
        return OrderResult(broker_order_id=order_id, status="SUBMITTED",
                           raw=resp if isinstance(resp, dict) else {"orderid": resp})

    def cancel_order(self, broker_order_id: str) -> bool:
        self._require_live_trading()
        resp = self._client.cancelOrder(broker_order_id, "NORMAL")
        if isinstance(resp, dict):
            return bool(resp.get("status", False))
        return resp is not None

    def get_orderbook(self) -> list[dict]:
        resp = self._client.orderBook()
        if isinstance(resp, dict):
            return resp.get("data") or []
        return list(resp or [])

    # --- internals ----------------------------------------------------------

    def _require_live_trading(self) -> None:
        if not self.live_trading_enabled:
            raise LiveTradingDisabledError(
                "Live trading is disabled. Construct with "
                "live_trading_enabled=True only after risk + paper validation."
            )


# --- module-level helpers (pure, testable) ---------------------------------


def _clean_totp_secret(secret: str) -> str:
    """Normalise a TOTP secret: drop surrounding whitespace and inner spaces."""
    return (secret or "").strip().replace(" ", "")


def _pad_base32(secret: str) -> str:
    """Pad a base32 string with '=' to a multiple of 8 chars (pyotp needs this)."""
    return secret + "=" * (-len(secret) % 8)


def _normalize_totp_secret(secret: str) -> str:
    """Clean + pad a TOTP secret so it decodes regardless of source formatting."""
    return _pad_base32(_clean_totp_secret(secret))


def _generate_totp(secret: str) -> str:
    """Generate the current 6-digit TOTP, with a clear error on a bad secret."""
    import pyotp  # type: ignore

    normalized = _normalize_totp_secret(secret)
    try:
        return pyotp.TOTP(normalized).now()
    except binascii.Error as exc:
        raise RuntimeError(
            "ANGEL_TOTP_SECRET is not a valid base32 TOTP secret. Use the "
            "secret STRING from the Angel 'Enable TOTP' QR setup (only letters "
            "A-Z and digits 2-7), not the 6-digit code and not the otpauth URL."
        ) from exc


def _best_bid_ask(item: dict) -> tuple[float, float]:
    """Extract best bid/ask from a getMarketData FULL depth payload."""
    depth = item.get("depth") or {}
    buys = depth.get("buy") or []
    sells = depth.get("sell") or []
    bid = float(buys[0]["price"]) if buys else 0.0
    ask = float(sells[0]["price"]) if sells else 0.0
    return bid, ask


def _import_smartconnect():  # pragma: no cover - thin import shim
    try:
        from SmartApi import SmartConnect  # type: ignore
        return SmartConnect
    except ImportError:
        pass
    try:
        from smartapi import SmartConnect  # type: ignore
        return SmartConnect
    except ImportError as exc:
        raise ImportError(
            "smartapi-python is not installed. Run: pip install smartapi-python"
        ) from exc


def _map_candle(row: list) -> Candle:
    """Map an Angel candle row [datetime, o, h, l, c, v] to a Candle.

    The datetime is ISO-8601 with a +05:30 offset; we strip the tz to a naive
    IST wall-clock timestamp for consistency with the rest of the platform.
    """
    dt = pd.Timestamp(row[0])
    if dt.tz is not None:
        dt = dt.tz_localize(None)
    return Candle(
        timestamp=dt.to_pydatetime(),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]) if len(row) > 5 and row[5] is not None else 0.0,
    )


def _chunk_ranges(
    start: date, end: date, max_days: int
) -> list[tuple[date, date]]:
    """Split [start, end] into <= max_days chunks (inclusive)."""
    if end < start:
        return []
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks
