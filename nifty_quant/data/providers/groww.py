"""Groww broker provider.

Maps Groww's official Python SDK (``growwapi``) onto the platform's
provider-agnostic interfaces. The rest of the system never imports ``growwapi``
or sees a Groww-specific payload -- it only sees :class:`OptionChain`,
:class:`OHLCVSeries`, and the order models.

Design notes
------------
- The SDK is imported lazily inside the auth helpers, so this module (and the
  whole package) imports fine even when ``growwapi`` is not installed. That
  keeps unit tests runnable offline.
- The authenticated SDK client is injected into ``__init__``. Tests pass a fake
  client; production code uses the ``from_api_key`` / ``from_totp`` / ``from_env``
  constructors. This makes every mapping function testable without a network.
- Order placement is gated behind ``live_trading_enabled`` (default False).
  Live execution is a late-phase concern; until then any attempt to trade
  raises loudly rather than silently sending a real order.

Security
--------
Never hard-code API keys. Use environment variables (see ``from_env``). The
option-chain IV from Groww is a *percentage* (e.g. 25.34); it is converted to a
fraction (0.2534) when mapped into :class:`OptionQuote`.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

from nifty_quant.data.models import (
    Candle,
    OHLCVSeries,
    OptionChain,
    OptionQuote,
    OptionType,
)
from nifty_quant.data.providers.base import (
    BrokerProvider,
    OrderKind,
    OrderRequest,
    OrderResult,
    OrderSide,
)

# Map our timeframe strings to Groww's interval_in_minutes.
_TIMEFRAME_TO_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
    "4h": 240,
    "240m": 240,
    "1d": 1440,
    "1w": 10080,
}


class LiveTradingDisabledError(RuntimeError):
    """Raised when an order method is called while live trading is disabled."""


class GrowwProvider(BrokerProvider):
    """Concrete :class:`BrokerProvider` backed by the Groww SDK."""

    def __init__(
        self,
        client: Any,
        *,
        exchange: str = "NSE",
        live_trading_enabled: bool = False,
    ) -> None:
        """Wrap an already-authenticated Groww SDK client.

        Parameters
        ----------
        client:
            An authenticated ``growwapi.GrowwAPI`` instance (or any object
            exposing the same methods -- handy for tests).
        exchange:
            Default exchange code ("NSE"/"BSE").
        live_trading_enabled:
            Must be explicitly True before any order can be placed/cancelled.
        """
        self._client = client
        self.exchange = exchange
        self.live_trading_enabled = live_trading_enabled

    # --- constructors / auth ------------------------------------------------

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        secret: str,
        **kwargs: Any,
    ) -> "GrowwProvider":
        """Authenticate using the API-key + secret flow (daily approval)."""
        GrowwAPI = _import_groww()
        access_token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
        return cls(GrowwAPI(access_token), **kwargs)

    @classmethod
    def from_totp(
        cls,
        api_key: str,
        totp_secret: str,
        **kwargs: Any,
    ) -> "GrowwProvider":
        """Authenticate using the TOTP flow (no expiry)."""
        GrowwAPI = _import_groww()
        try:
            import pyotp  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "pyotp is required for the TOTP flow: pip install pyotp"
            ) from exc
        totp = pyotp.TOTP(totp_secret).now()
        access_token = GrowwAPI.get_access_token(api_key=api_key, totp=totp)
        return cls(GrowwAPI(access_token), **kwargs)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "GrowwProvider":
        """Build from environment variables.

        Preferred (TOTP, no expiry):
            GROWW_API_KEY, GROWW_TOTP_SECRET
        Or (API key + secret, daily approval):
            GROWW_API_KEY, GROWW_API_SECRET
        """
        api_key = os.environ.get("GROWW_API_KEY")
        if not api_key:
            raise RuntimeError("GROWW_API_KEY is not set in the environment")
        totp_secret = os.environ.get("GROWW_TOTP_SECRET")
        if totp_secret:
            return cls.from_totp(api_key, totp_secret, **kwargs)
        secret = os.environ.get("GROWW_API_SECRET")
        if secret:
            return cls.from_api_key(api_key, secret, **kwargs)
        raise RuntimeError(
            "Set GROWW_TOTP_SECRET (preferred) or GROWW_API_SECRET to authenticate"
        )

    # --- read-only market data ---------------------------------------------

    def get_spot(self, symbol: str, segment: str = "CASH") -> float:
        """Latest traded price for an index/stock symbol (e.g. 'NIFTY')."""
        key = f"{self.exchange}_{symbol}"
        resp = self._client.get_ltp(
            segment=segment, exchange_trading_symbols=key
        )
        if key not in resp:
            raise KeyError(f"LTP response missing key {key!r}: {resp!r}")
        return float(resp[key])

    def get_quote(
        self, trading_symbol: str, segment: str = "CASH"
    ) -> dict:
        """Full real-time quote (depth, OI, IV, ...) as the raw broker dict."""
        return self._client.get_quote(
            exchange=self.exchange,
            segment=segment,
            trading_symbol=trading_symbol,
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        segment: str = "CASH",
    ) -> OHLCVSeries:
        """Historical candles mapped into an :class:`OHLCVSeries`."""
        if timeframe not in _TIMEFRAME_TO_MINUTES:
            raise ValueError(
                f"unsupported timeframe {timeframe!r}; "
                f"valid: {sorted(_TIMEFRAME_TO_MINUTES)}"
            )
        interval = _TIMEFRAME_TO_MINUTES[timeframe]
        resp = self._client.get_historical_candle_data(
            trading_symbol=symbol,
            exchange=self.exchange,
            segment=segment,
            start_time=f"{start.isoformat()} 00:00:00",
            end_time=f"{end.isoformat()} 23:59:59",
            interval_in_minutes=interval,
        )
        candles = [_map_candle(row) for row in resp.get("candles", [])]
        return OHLCVSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    def get_option_chain(
        self,
        underlying: str,
        expiry: date,
    ) -> OptionChain:
        """Groww option chain mapped into an :class:`OptionChain` snapshot."""
        resp = self._client.get_option_chain(
            exchange=self.exchange,
            underlying=underlying,
            expiry_date=expiry.isoformat(),
        )
        spot = float(resp["underlying_ltp"])
        ts = datetime.now()
        quotes: list[OptionQuote] = []
        for strike_str, leg in resp.get("strikes", {}).items():
            strike = float(strike_str)
            for key, otype in (("CE", OptionType.CALL), ("PE", OptionType.PUT)):
                contract = leg.get(key)
                if not contract:
                    continue
                quotes.append(
                    _map_option_quote(contract, strike, otype, expiry)
                )
        return OptionChain(
            underlying=underlying,
            spot=spot,
            expiry=expiry,
            timestamp=ts,
            quotes=tuple(quotes),
        )

    # --- execution (gated) --------------------------------------------------

    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place a real order. Refuses unless live trading is explicitly on."""
        self._require_live_trading()
        client = self._client
        resp = client.place_order(
            trading_symbol=order.trading_symbol,
            quantity=order.quantity,
            validity=client.VALIDITY_DAY,
            exchange=getattr(client, f"EXCHANGE_{self.exchange}"),
            segment=client.SEGMENT_FNO,
            product=client.PRODUCT_MIS,
            order_type=(
                client.ORDER_TYPE_LIMIT
                if order.kind is OrderKind.LIMIT
                else client.ORDER_TYPE_MARKET
            ),
            transaction_type=(
                client.TRANSACTION_TYPE_BUY
                if order.side is OrderSide.BUY
                else client.TRANSACTION_TYPE_SELL
            ),
            price=order.price,
            trigger_price=order.trigger_price,
            order_reference_id=order.reference_id,
        )
        return OrderResult(
            broker_order_id=str(
                resp.get("groww_order_id") or resp.get("order_id", "")
            ),
            status=str(resp.get("order_status", "UNKNOWN")),
            raw=resp,
        )

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a working order. Refuses unless live trading is enabled."""
        self._require_live_trading()
        resp = self._client.cancel_order(groww_order_id=broker_order_id)
        status = str(resp.get("order_status", "")).upper()
        return status in ("CANCELLED", "CANCELLATION_REQUESTED")

    def get_orderbook(self) -> list[dict]:
        """Return the session order list (read-only, no gate)."""
        resp = self._client.get_order_list()
        if isinstance(resp, dict):
            return resp.get("order_list", [])
        return list(resp)

    # --- internals ----------------------------------------------------------

    def _require_live_trading(self) -> None:
        if not self.live_trading_enabled:
            raise LiveTradingDisabledError(
                "Live trading is disabled. Construct the provider with "
                "live_trading_enabled=True only after the risk engine and "
                "paper-trading validation are in place."
            )


# --- module-level mapping helpers (pure, easy to test) ---------------------


def _import_groww():  # pragma: no cover - thin import shim
    try:
        from growwapi import GrowwAPI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "growwapi is not installed. Run: pip install --upgrade growwapi"
        ) from exc
    return GrowwAPI


def _map_candle(row: list) -> Candle:
    """Map a Groww candle row [epoch_sec, o, h, l, c, v] to a Candle."""
    ts, o, h, low, c, *rest = row
    volume = float(rest[0]) if rest else 0.0
    return Candle(
        timestamp=datetime.fromtimestamp(int(ts)),
        open=float(o),
        high=float(h),
        low=float(low),
        close=float(c),
        volume=volume,
    )


def _map_option_quote(
    contract: dict,
    strike: float,
    option_type: OptionType,
    expiry: date,
) -> OptionQuote:
    """Map one Groww option contract dict to an :class:`OptionQuote`.

    Groww reports IV as a percentage (e.g. 25.34); convert to a fraction.
    """
    greeks = contract.get("greeks") or {}
    iv_pct = greeks.get("iv")
    iv_frac = float(iv_pct) / 100.0 if iv_pct not in (None, 0) else None
    return OptionQuote(
        strike=strike,
        option_type=option_type,
        expiry=expiry,
        last_price=float(contract.get("ltp", 0.0)),
        open_interest=float(contract.get("open_interest", 0.0)),
        volume=float(contract.get("volume", 0.0)),
        implied_volatility=iv_frac,
    )
