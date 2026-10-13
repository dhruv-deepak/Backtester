"""
features.py — Feature engineering for the ML strategy.

Every feature at row t uses only data up to and including the close of day
t (rolling windows and lags, never negative shifts). The label at row t is
the direction of the NEXT day's return — the thing you're trying to predict
at the close of day t. On the final row that return hasn't happened yet, so
the label is NaN: the row can be predicted but not trained or scored on.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix plus a 'label' column (1 = next day up, 0 = down, NaN = unknown)."""
    close = df["Close"]
    ret = close.pct_change()
    feat = pd.DataFrame(index=df.index)

    # recent price action
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{lag}d"] = close.pct_change(lag)

    # trend / mean-reversion, scale-free
    for window in [5, 10, 20, 50, 100]:
        feat[f"px_to_ma_{window}"] = close / close.rolling(window).mean() - 1

    # volatility level and regime
    for window in [5, 20, 60]:
        feat[f"vol_{window}d"] = ret.rolling(window).std()
    feat["vol_ratio_5_60"] = feat["vol_5d"] / feat["vol_60d"]

    feat["rsi_14"] = compute_rsi(close, 14)
    feat["z_score_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    # candle structure: where did today close within its range? how big was the gap?
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    feat["close_in_range"] = (close - df["Low"]) / rng
    feat["overnight_gap"] = df["Open"] / close.shift(1) - 1

    if df["Volume"].gt(0).any():
        feat["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

    next_ret = ret.shift(-1)
    feat["label"] = np.where(next_ret.isna(), np.nan, (next_ret > 0).astype(float))

    return feat.replace([np.inf, -np.inf], np.nan)
