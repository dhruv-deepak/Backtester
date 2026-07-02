"""
strategies.py — Signal generators.

Each strategy function takes a price DataFrame (must contain 'Close') and
returns a Series of target positions aligned to the same index:
    +1  -> fully long
     0  -> flat / no position
    -1  -> fully short

IMPORTANT: signals are generated using only data available up to and
including the current day (no lookahead). The engine is responsible for
shifting signals forward by one day before applying them, so that a
strategy 'decided' on day t only earns the return realized on day t+1.
This mirrors how you'd actually trade: you can't act on a close price
until after it's printed.
"""

import numpy as np
import pandas as pd


def moving_average_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """
    Classic trend-following strategy: go long when the fast moving average
    is above the slow moving average, flat otherwise.
    """
    fast_ma = df["Close"].rolling(fast).mean()
    slow_ma = df["Close"].rolling(slow).mean()

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[fast_ma > slow_ma] = 1
    signal[fast_ma < slow_ma] = 0  # flat rather than short — keeps it simple/interpretable
    return signal


def momentum(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Time-series momentum: go long if the asset's return over the past
    `lookback` days is positive, short if negative.
    """
    past_return = df["Close"].pct_change(lookback)

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[past_return > 0] = 1
    signal[past_return < 0] = -1
    return signal


def mean_reversion(df: pd.DataFrame, window: int = 20, num_std: float = 1.0) -> pd.Series:
    """
    Bollinger-Band-style mean reversion: go long when price is more than
    `num_std` standard deviations BELOW its rolling mean (expecting it to
    revert up), short when it's that far ABOVE (expecting reversion down).
    """
    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std = df["Close"].rolling(window).std()

    z_score = (df["Close"] - rolling_mean) / rolling_std

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[z_score < -num_std] = 1
    signal[z_score > num_std] = -1
    return signal


STRATEGY_REGISTRY = {
    "Moving Average Crossover": moving_average_crossover,
    "Momentum": momentum,
    "Mean Reversion": mean_reversion,
}
