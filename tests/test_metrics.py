import numpy as np
import pandas as pd
import pytest

from backtester import metrics


def test_sortino_uses_downside_deviation_over_all_periods():
    r = pd.Series([0.02, -0.01, 0.02, -0.01])
    dd = np.sqrt((0.01**2 + 0.01**2) / 4)
    assert metrics.downside_deviation(r) == pytest.approx(dd)
    assert metrics.sortino_ratio(r) == pytest.approx(r.mean() / dd * np.sqrt(252))


def test_sortino_defined_when_losses_are_identical():
    # the old std-of-negatives version returned NaN here: identical losses have zero dispersion
    r = pd.Series([0.01, -0.01] * 50)
    assert np.isfinite(metrics.sortino_ratio(r))


def test_max_drawdown_and_duration():
    r = pd.Series([0.10, -0.50, 0.20, 0.50, 0.10])  # equity: 1.1, .55, .66, .99, 1.089
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)
    assert metrics.max_drawdown_duration(r) == 4


def test_cagr():
    r = pd.Series([0.0] * 251 + [0.10])
    assert metrics.cagr(r) == pytest.approx(0.10)


def test_trade_stats():
    trades = pd.DataFrame({"return": [0.1, -0.05, 0.02, -0.01], "days": [10, 5, 3, 2]})
    s = metrics.trade_stats(trades)
    assert s["Win Rate"] == 0.5
    assert s["Profit Factor"] == pytest.approx(0.12 / 0.06)
    assert s["Avg Hold (days)"] == 5
