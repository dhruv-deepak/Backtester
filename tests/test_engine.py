import numpy as np
import pandas as pd
import pytest

from backtester.engine import Backtester, CostModel, extract_trades

FREE = CostModel(commission_bps=0, half_spread_bps=0, vol_slippage=0, borrow_bps_annual=0, financing_bps_annual=0)


@pytest.fixture
def bars():
    #            day0   day1   day2   day3
    return pd.DataFrame(
        {"Open": [100, 102, 110, 99], "High": [101, 106, 112, 101], "Low": [99, 101, 104, 97],
         "Close": [100, 105, 108, 100], "Volume": [1, 1, 1, 1]},
        index=pd.bdate_range("2024-01-01", periods=4), dtype=float,
    )


def test_signal_is_traded_with_a_one_bar_delay(bars):
    # decide long at day0's close -> position held over day1
    res = Backtester(bars, costs=FREE, execution="close").run(pd.Series([1, 0, 0, 0], index=bars.index, dtype=float))
    assert res.positions.tolist() == [0, 1, 0, 0]
    assert res.returns.iloc[1] == pytest.approx(105 / 100 - 1)
    assert res.returns.iloc[[0, 2, 3]].abs().max() == 0


def test_next_open_fill_splits_gap_and_intraday(bars):
    # long decided at close of day1, filled at day2 open (110); exit decided day2, filled day3 open (99)
    sig = pd.Series([0, 1, 0, 0], index=bars.index, dtype=float)
    res = Backtester(bars, costs=FREE, execution="next_open").run(sig)
    assert res.returns.iloc[1] == 0                             # still flat on day1
    assert res.returns.iloc[2] == pytest.approx(108 / 110 - 1)  # missed the 105->110 gap, earned intraday
    assert res.returns.iloc[3] == pytest.approx(99 / 108 - 1)   # held through the overnight gap, out at the open


def test_costs_charged_per_unit_traded(bars):
    costs = CostModel(commission_bps=10, half_spread_bps=0, vol_slippage=0, borrow_bps_annual=0)
    sig = pd.Series([1, -1, -1, 0], index=bars.index, dtype=float)
    res = Backtester(bars, costs=costs, execution="close").run(sig)
    traded = res.positions.diff().abs().fillna(0)
    assert traded.tolist() == [0, 1, 2, 0]  # held = [0, 1, -1, -1]; the final 0 is never filled
    expected = (1 - traded * 0.001) * (1 + res.gross_returns) - 1
    np.testing.assert_allclose(res.returns, expected)


def test_flat_signal_earns_nothing(random_walk):
    res = Backtester(random_walk).run(pd.Series(0.0, index=random_walk.index))
    assert (res.returns == 0).all() and len(res.trades) == 0


def test_buy_and_hold_matches_asset_without_costs(random_walk):
    res = Backtester(random_walk, costs=FREE, execution="close").run(pd.Series(1.0, index=random_walk.index))
    asset = random_walk["Close"].pct_change().fillna(0)
    np.testing.assert_allclose(res.returns.iloc[1:], asset.iloc[1:])


def test_short_borrow_and_leverage_carry(bars):
    costs = CostModel(commission_bps=0, half_spread_bps=0, vol_slippage=0,
                      borrow_bps_annual=252, financing_bps_annual=504)  # 1bp/day, 2bp/day
    free = Backtester(bars, costs=FREE, execution="close")
    carry = Backtester(bars, costs=costs, execution="close")
    for pos, daily in [(-1.0, 1e-4), (2.0, 2e-4)]:
        sig = pd.Series(pos, index=bars.index)
        diff = free.run(sig).returns - carry.run(sig).returns
        assert diff.iloc[1:].to_numpy() == pytest.approx(daily)


def test_trade_extraction():
    idx = pd.bdate_range("2024-01-01", periods=8)
    pos = pd.Series([0, 1, 1, 0, -1, -1, 1, 0], index=idx, dtype=float)
    ret = pd.Series([0, .01, .02, -.001, .01, -.02, .03, 0], index=idx)
    t = extract_trades(pos, ret)
    assert t["direction"].tolist() == ["long", "short", "long"]
    assert t["days"].tolist() == [2, 2, 1]
    # first trade includes its flat exit day (where the exit cost lands)
    assert t["return"].iloc[0] == pytest.approx(1.01 * 1.02 * 0.999 - 1)
