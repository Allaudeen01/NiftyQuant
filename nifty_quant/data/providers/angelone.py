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
    ) -> None:
        """Wrap an already-authenticated SmartConnect client.

        ``request_pause`` throttles chunked historical requests to respect
        SmartAPI rate limits (set 0 in tests). ``instrument_master`` is an
        :class:`InstrumentMaster` used to resolve option tokens; created lazily
        if not supplied.
        """
        self._client = client
        self.exchange = exchange
        self.symbol_tokens = {**_DEFAULT_TOKENS, **(symbol_tokens or {})}
        self.live_trading_enabled = live_trading_enabled
        self.request_pause = request_pause
        self._instrument_master = instrument_master

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
            resp = self._client.getCandleData(params)
            for row in (resp or {}).get("data", []) or []:
                candle = _map_candle(row)
                if candle.timestamp not in seen:
                    seen.add(candle.timestamp)
                    candles.append(candle)
            if self.request_pause > 0:
                time.sleep(self.request_pause)

        candles.sort(key=lambda c: c.timestamp)
        _log.event(
            "angel_candles_fetched",
            symbol=symbol, timeframe=timeframe, count=len(candles),
        )
        return OHLCVSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    def get_spot(self, symbol: str) -> float:
        """Latest close via the most recent daily candle (no extra endpoint)."""
        end = datetime.now().date()
        start = end - timedelta(days=7)
        series = self.get_ohlcv(symbol, "1d", start, end)
        if not series.candles:
            raise RuntimeError(f"no recent candles for {symbol!r}")
        return series.candles[-1].close

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
            resp = self._client.getMarketData("FULL", {"NFO": batch})
            data = (resp or {}).get("data", {}) if isinstance(resp, dict) else {}
            fetched.extend(data.get("fetched", []) or [])
            if self.request_pause > 0:
                time.sleep(self.request_pause)
        return fetched

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
