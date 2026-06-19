"""Tests for option-chain analytics and the data models."""

from datetime import date, datetime

import pytest

from nifty_quant.analytics import options as opt
from nifty_quant.data.models import (
    OptionChain,
    OptionQuote,
    OptionType,
)


def _chain():
    expiry = date(2026, 1, 27)
    ts = datetime(2026, 1, 20, 10, 0, 0)
    spot = 25050.0
    quotes = []
    for strike in [24800, 24900, 25000, 25100, 25200]:
        # Synthetic OI: more puts piled below spot, calls above.
        call_oi = 1000 + (strike - 24800)
        put_oi = 1000 + (25200 - strike)
        quotes.append(
            OptionQuote(
                strike=strike,
                option_type=OptionType.CALL,
                expiry=expiry,
                last_price=max(spot - strike, 5.0),
                bid=0.0,
                ask=0.0,
                open_interest=call_oi,
                volume=call_oi * 2,
                implied_volatility=0.13,
            )
        )
        quotes.append(
            OptionQuote(
                strike=strike,
                option_type=OptionType.PUT,
                expiry=expiry,
                last_price=max(strike - spot, 5.0),
                open_interest=put_oi,
                volume=put_oi * 2,
                implied_volatility=0.13,
            )
        )
    return OptionChain(
        underlying="NIFTY", spot=spot, expiry=expiry, timestamp=ts, quotes=quotes
    )


def test_atm_strike():
    chain = _chain()
    assert chain.atm_strike() == 25000


def test_pcr_oi_positive():
    chain = _chain()
    pcr = opt.put_call_ratio(chain, by="oi")
    assert pcr > 0


def test_pcr_volume_matches_oi_here():
    # Volume is 2x OI for every contract, so the ratio is identical.
    chain = _chain()
    assert opt.put_call_ratio(chain, by="oi") == pytest.approx(
        opt.put_call_ratio(chain, by="volume")
    )


def test_max_pain_is_a_listed_strike():
    chain = _chain()
    mp = opt.max_pain(chain)
    assert mp in chain.strikes()


def test_iv_rank_basic():
    assert opt.iv_rank(15.0, [10.0, 20.0]) == pytest.approx(0.5)
    assert opt.iv_rank(10.0, [10.0, 20.0]) == pytest.approx(0.0)
    assert opt.iv_rank(20.0, [10.0, 20.0]) == pytest.approx(1.0)


def test_iv_rank_empty_is_nan():
    import math

    assert math.isnan(opt.iv_rank(15.0, []))


def test_iv_percentile():
    assert opt.iv_percentile(15.0, [10, 12, 14, 16, 18]) == pytest.approx(0.6)


def test_gamma_exposure_runs():
    chain = _chain()
    gex = opt.gamma_exposure(chain)
    assert isinstance(gex.total_gex, float)
    assert set(gex.per_strike).issubset(set(chain.strikes()))


def test_atm_iv_recovers_input():
    chain = _chain()
    # All contracts carry iv=0.13, so ATM iv should be ~0.13.
    assert opt.atm_iv(chain) == pytest.approx(0.13, abs=1e-9)


def test_candle_validation():
    from nifty_quant.data.models import Candle

    with pytest.raises(ValueError):
        Candle(datetime(2026, 1, 1), open=10, high=5, low=8, close=9)
