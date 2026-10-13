"""
strategies.py — Signal generators.

Each strategy function takes a price DataFrame and returns a Series of
TARGET positions aligned to the same index (+1 long, 0 flat, -1 short;
fractional/levered values are allowed once an overlay like volatility
targeting is applied).

Contract: the value at row t may only depend on rows <= t. The engine
handles the timing of when that decision can actually be traded.
validation.check_no_lookahead() enforces this contract empirically, and the
test suite runs it against every strategy registered here.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _hysteresis(enter_long, enter_short, exit_long, exit_short) -> np.ndarray:
    """
    Turn entry/exit conditions into a stateful position: once in a trade,
    stay in it until its exit condition fires. Entries take priority over
    exits on the same bar (so a long can flip straight to short).

    Path-dependent, so it's a plain loop — clearer than a vectorised trick.
    """
    n = len(enter_long)
    pos = np.zeros(n)
    state = 0.0
    for t in range(n):
        if enter_long[t]:
            state = 1.0
        elif enter_short[t]:
            state = -1.0
        elif (state > 0 and exit_long[t]) or (state < 0 and exit_short[t]):
            state = 0.0
        pos[t] = state
    return pos


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------

def moving_average_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Trend following: long while the fast MA is above the slow MA, flat otherwise."""
    close = df["Close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def momentum(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Time-series momentum: long if the trailing `lookback`-day return is positive, short if negative."""
    past_return = df["Close"].pct_change(lookback)
    return np.sign(past_return).fillna(0.0)


def mean_reversion(df: pd.DataFrame, window: int = 20, entry_z: float = 2.0) -> pd.Series:
    """
    Bollinger-band mean reversion with proper exits: enter long when the
    z-score drops below -entry_z, short above +entry_z, and hold until price
    reverts back through its rolling mean (z crosses 0).

    (Holding only *while* outside the band — the naive version — exits
    exactly when the trade starts working, and churns costs.)
    """
    close = df["Close"]
    z = ((close - close.rolling(window).mean()) / close.rolling(window).std()).to_numpy()
    with np.errstate(invalid="ignore"):
        pos = _hysteresis(
            enter_long=z < -entry_z,
            enter_short=z > entry_z,
            exit_long=z >= 0,
            exit_short=z <= 0,
        )
    return pd.Series(pos, index=df.index)


def donchian_breakout(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    """
    Turtle-style channel breakout: go long when the close breaks above the
    highest high of the prior `entry` days, short below the lowest low.
    Exit on a break of the shorter `exit`-day channel in the other direction.
    The channels use .shift(1) so today's bar isn't part of its own threshold.
    """
    close = df["Close"]
    upper_in = df["High"].rolling(entry).max().shift(1)
    lower_in = df["Low"].rolling(entry).min().shift(1)
    upper_out = df["High"].rolling(exit).max().shift(1)
    lower_out = df["Low"].rolling(exit).min().shift(1)
    pos = _hysteresis(
        enter_long=(close > upper_in).to_numpy(),
        enter_short=(close < lower_in).to_numpy(),
        exit_long=(close < lower_out).to_numpy(),
        exit_short=(close > upper_out).to_numpy(),
    )
    return pd.Series(pos, index=df.index)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


# --------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------

def volatility_target(
    signal: pd.Series,
    df: pd.DataFrame,
    target_vol: float = 0.15,
    span: int = 30,
    max_leverage: float = 2.0,
    rebalance_band: float = 0.25,
) -> pd.Series:
    """
    Scale positions so the strategy runs at roughly `target_vol` annualised
    volatility: size = signal * target_vol / forecast_vol, capped at
    `max_leverage`. Forecast vol is an EWMA of returns up to and including
    today, so the sizing is causal.

    Why it matters: volatility clusters. Cutting size when vol spikes is one
    of the few robust ways to improve risk-adjusted returns and drawdowns.

    `rebalance_band`: only resize when the target differs from the current
    position by more than this fraction of it. Without a band, the position
    is nudged every single day and the trading costs eat much of the benefit.
    """
    returns = df["Close"].pct_change()
    forecast_vol = returns.ewm(span=span, min_periods=span).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / forecast_vol).clip(upper=max_leverage)
    target = (signal * scale).fillna(0.0).to_numpy()

    out = np.empty(len(target))
    current = 0.0
    for t, tgt in enumerate(target):
        if tgt == 0 or np.sign(tgt) != np.sign(current) or abs(tgt - current) > rebalance_band * abs(current):
            current = tgt
        out[t] = current
    return pd.Series(out, index=signal.index)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    name: str
    fn: Callable[..., pd.Series]
    params: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)  # parameter grid for optimisation
    constraint: Callable[[dict], bool] | None = None  # filters invalid combos

    def signal(self, df: pd.DataFrame, **overrides) -> pd.Series:
        return self.fn(df, **{**self.params, **overrides})

    def combos(self) -> list[dict]:
        """All parameter combinations in the grid that satisfy the constraint."""
        keys = list(self.grid)
        out = [dict(zip(keys, values)) for values in product(*self.grid.values())]
        return [c for c in out if self.constraint is None or self.constraint(c)]


STRATEGIES: dict[str, Strategy] = {
    "ma": Strategy(
        "MA Crossover", moving_average_crossover,
        params={"fast": 20, "slow": 100},
        grid={"fast": [5, 10, 20, 30, 50], "slow": [50, 100, 150, 200, 250]},
        constraint=lambda p: p["fast"] < p["slow"],
    ),
    "momentum": Strategy(
        "Momentum", momentum,
        params={"lookback": 60},
        grid={"lookback": [10, 20, 40, 60, 90, 120, 180, 250]},
    ),
    "meanrev": Strategy(
        "Mean Reversion", mean_reversion,
        params={"window": 20, "entry_z": 2.0},
        grid={"window": [10, 15, 20, 30, 40], "entry_z": [1.0, 1.5, 2.0, 2.5]},
    ),
    "breakout": Strategy(
        "Donchian Breakout", donchian_breakout,
        params={"entry": 55, "exit": 20},
        grid={"entry": [20, 40, 55, 80, 100], "exit": [10, 20, 30, 40]},
        constraint=lambda p: p["exit"] < p["entry"],
    ),
}

BUY_AND_HOLD = Strategy("Buy & Hold", buy_and_hold)
