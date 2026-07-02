"""
data.py — Historical price data loader.

Two sources:
  1. load_yahoo()      -> real historical OHLCV data via yfinance (requires internet)
  2. generate_synthetic() -> synthetic Geometric Brownian Motion price series,
                              useful for testing the engine offline or for
                              stress-testing strategies against known statistical
                              properties (drift, volatility) you control.
"""

import numpy as np
import pandas as pd


def load_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for `ticker` between `start` and `end` (YYYY-MM-DD).
    Requires internet access and the `yfinance` package:
        pip install yfinance

    Returns a DataFrame indexed by date with a 'Close' column at minimum.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol/date range.")

    # yfinance sometimes returns MultiIndex columns for single tickers; flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Open", "High", "Low", "Close", "Volume"]]


def generate_synthetic(
    days: int = 1000,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic daily price series using Geometric Brownian Motion:
        dS = mu * S * dt + sigma * S * dW

    Parameters
    ----------
    days  : number of trading days to simulate
    s0    : starting price
    mu    : annualized drift (expected return)
    sigma : annualized volatility
    seed  : RNG seed for reproducibility

    Returns a DataFrame with a synthetic date index and a 'Close' column,
    shaped like real market data so it's a drop-in replacement for load_yahoo().
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252  # one trading day, in years

    # log-return increments
    daily_drift = (mu - 0.5 * sigma**2) * dt
    daily_shock = sigma * np.sqrt(dt) * rng.standard_normal(days)
    log_returns = daily_drift + daily_shock

    log_prices = np.log(s0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    df = pd.DataFrame(index=dates)
    df["Close"] = prices
    # fabricate plausible OHLV around Close for completeness
    df["Open"] = df["Close"].shift(1).fillna(s0) * (1 + rng.normal(0, 0.002, days))
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.004, days)))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.004, days)))
    df["Volume"] = rng.integers(1_000_000, 5_000_000, days)

    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_price_data(ticker: str = None, start: str = None, end: str = None, synthetic_days: int = 1000) -> pd.DataFrame:
    """
    Convenience wrapper: tries real Yahoo data first (if ticker given and
    yfinance/internet available), falls back to synthetic data otherwise.
    This lets main.py run identically online or offline.
    """
    if ticker and start and end:
        try:
            return load_yahoo(ticker, start, end)
        except Exception as e:
            print(f"[data] Falling back to synthetic data — could not fetch '{ticker}': {e}")

    return generate_synthetic(days=synthetic_days)
