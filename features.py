"""
features.py — Feature engineering for the ML-based signal strategy.

All features at row t are computed using ONLY data up to and including day t
(rolling windows, lags) — none peek into the future. The label is the
direction of the NEXT day's return, so training a model on (features_t,
label_t) and using it to predict at time t is exactly what you can do live:
you know features_t at the close of day t, and you're predicting day t+1.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index — momentum oscillator, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral value where undefined (early rows / no losses)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix from price data. Returns a DataFrame of features
    plus a 'label' column (1 if next day's return > 0, else 0), aligned so
    that row t's features can legitimately be used to predict row t's label
    (which represents the return realized going INTO day t+1).
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]

    daily_return = close.pct_change()

    # Lagged returns — recent price action
    for lag in [1, 2, 3, 5, 10]:
        feat[f"return_lag_{lag}"] = daily_return.shift(lag - 1).rolling(1).sum() if lag == 1 else close.pct_change(lag)

    # Moving-average ratios — trend/mean-reversion signal, scale-free
    for window in [5, 10, 20, 50]:
        ma = close.rolling(window).mean()
        feat[f"price_to_ma_{window}"] = close / ma - 1

    # Volatility — recent realized vol
    for window in [5, 10, 20]:
        feat[f"volatility_{window}"] = daily_return.rolling(window).std()

    # RSI — momentum oscillator
    feat["rsi_14"] = compute_rsi(close, 14)

    # Volume features (if available)
    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20).mean()
        feat["volume_ratio"] = df["Volume"] / vol_ma.replace(0, np.nan)

    # Bollinger-style z-score
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    feat["z_score_20"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    # Label: direction of the NEXT day's return (this is what we're predicting)
    feat["label"] = (daily_return.shift(-1) > 0).astype(int)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat
