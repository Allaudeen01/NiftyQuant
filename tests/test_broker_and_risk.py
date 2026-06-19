"""Tests for fill models, the simulated broker, and the risk engine."""

from datetime import date, datetime

import pytest

from nifty_quant.backtest.broker import (
    BidAskFill,
    FixedSlippage,
    MidPriceFill,
    PercentSlippage,
    Quote,
    SimulatedBroker,
)
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.intents import ApprovedOrder, TradeIntent
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.data.models import Candle, OptionType
from nifty_quant.data.providers.base import OrderKind, OrderSide
from nifty_quant.feed.events import CandleEvent

INST = Instrument("NIFTY", InstrumentType.INDEX)
T0 = datetime(2025, 1, 2, 9, 15)


# --- fill models -----------------------------------------------------------


def test_mid_price_fill():
    q = Quote(last=101, bid=100, ask=102)
    assert MidPriceFill().fill_price(OrderSide.BUY, q) == pytest.approx(101)
    assert MidPriceFill().fill_price(OrderSide.SELL, q) == pytest.approx(101)


def test_bid_ask_fill_pays_spread():
    q = Quote(last=101, bid=100, ask=102)
    assert BidAskFill().fill_price(OrderSide.BUY, q) == pytest.approx(102)
    assert BidAskFill().fill_price(OrderSide.SELL, q) == pytest.approx(100)


def test_bid_ask_fill_falls_back_to_last():
    q = Quote(last=101)  # no two-sided quote
    assert BidAskFill().fill_price(OrderSide.BUY, q) == pytest.approx(101)


def test_fixed_slippage_is_adverse():
    q = Quote(last=100)
    assert FixedSlippage(0.5).fill_price(OrderSide.BUY, q) == pytest.approx(100.5)
    assert FixedSlippage(0.5).fill_price(OrderSide.SELL, q) == pytest.approx(99.5)


def test_percent_slippage_is_adverse():
    q = Quote(last=100)
    assert PercentSlippage(0.01).fill_price(OrderSide.BUY, q) == pytest.approx(101)
    assert PercentSlippage(0.01).fill_price(OrderSide.SELL, q) == pytest.approx(99)


# --- broker ----------------------------------------------------------------


def _candle_event(close, ts=T0):
    return CandleEvent(
        timestamp=ts,
        symbol="NIFTY",
        timeframe="5m",
        candle=Candle(ts, close, close + 1, close - 1, close, 1000),
    )


def test_broker_fills_from_candle_quote():
    broker = SimulatedBroker(fill_model=MidPriceFill())
    broker.on_event(_candle_event(100.0))
    order = ApprovedOrder(T0, INST, OrderSide.BUY, 1)
    fill = broker.execute(order)
    assert fill is not None
    assert fill.price == pytest.approx(100.0)


def test_broker_skips_when_no_quote():
    broker = SimulatedBroker()
    order = ApprovedOrder(T0, INST, OrderSide.BUY, 1)
    assert broker.execute(order) is None


def test_broker_limit_order_not_crossed_is_unfilled():
    broker = SimulatedBroker(fill_model=MidPriceFill())
    broker.on_event(_candle_event(100.0))
    # Want to BUY only at <= 95, market is 100 -> no fill.
    order = ApprovedOrder(T0, INST, OrderSide.BUY, 1, kind=OrderKind.LIMIT, limit_price=95.0)
    assert broker.execute(order) is None


# --- risk engine -----------------------------------------------------------


def _intent(side, qty=None, flatten=False):
    return TradeIntent(T0, INST, side, quantity=qty, flatten=flatten)


def test_risk_uses_default_quantity():
    re = BasicRiskEngine(default_quantity=5)
    decision = re.evaluate(_intent(OrderSide.BUY), Portfolio(), {})
    assert decision.approved
    assert decision.order.quantity == 5


def test_risk_respects_quantity_hint():
    re = BasicRiskEngine(default_quantity=5)
    decision = re.evaluate(_intent(OrderSide.BUY, qty=12), Portfolio(), {})
    assert decision.order.quantity == 12


def test_risk_blocks_short_when_disabled():
    re = BasicRiskEngine(allow_short=False)
    decision = re.evaluate(_intent(OrderSide.SELL, qty=3), Portfolio(), {})
    assert not decision.approved


def test_risk_flatten_resolves_side_and_size():
    re = BasicRiskEngine()
    pf = Portfolio()
    from nifty_quant.backtest.intents import Fill

    pf.apply_fill(Fill(T0, INST, OrderSide.BUY, 7, 100.0))
    decision = re.evaluate(_intent(OrderSide.SELL, flatten=True), pf, {})
    assert decision.approved
    assert decision.order.side is OrderSide.SELL
    assert decision.order.quantity == 7


def test_risk_per_instrument_cap():
    re = BasicRiskEngine(default_quantity=10, max_units_per_instrument=15)
    pf = Portfolio()
    from nifty_quant.backtest.intents import Fill

    pf.apply_fill(Fill(T0, INST, OrderSide.BUY, 10, 100.0))
    decision = re.evaluate(_intent(OrderSide.BUY, qty=10), pf, {})
    # Only room for 5 more before hitting the cap of 15.
    assert decision.approved
    assert decision.order.quantity == 5


def test_risk_gross_exposure_cap():
    re = BasicRiskEngine(default_quantity=100, max_gross_exposure=5000)
    decision = re.evaluate(
        _intent(OrderSide.BUY, qty=100), Portfolio(), {INST.key: 100.0}
    )
    # 100 * 100 = 10000 > 5000 cap.
    assert not decision.approved
