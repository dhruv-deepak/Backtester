"""
data.py — Price data loading.

Sources:
  load_yahoo()          real daily OHLCV via yfinance (split/dividend adjusted),
                        cached to disk so repeated runs are fast and reproducible
  load_csv()            any local OHLCV CSV with a date column
  generate_synthetic()  simulated prices. With trend=0 this is a pure random
                        walk (GBM): nothing is predictable, so any "edge" a
                        strategy shows on it is noise by construction — the
                        null hypothesis made concrete. With trend > 0, a slowly
                        wandering hidden drift is mixed in: a planted edge that
                        trend-following strategies SHOULD find and mean
                        reversion should lose to (a positive control).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def _clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and normalise an OHLCV frame. Fails loudly rather than guessing."""
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["Open", "Close"])
    df["Volume"] = df["Volume"].fillna(0)

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{source}: contains non-positive prices")
    if len(df) < 100:
        raise ValueError(f"{source}: only {len(df)} rows — too short to backtest meaningfully")
    return df.astype(float)


def load_yahoo(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Daily OHLCV for `ticker` between `start` and `end` (YYYY-MM-DD).

    Uses auto_adjust=True so all OHLC prices are adjusted for splits AND
    dividends — i.e. returns are total returns. Using unadjusted prices is a
    classic silent bug: a 2-for-1 split looks like a -50% crash.
    """
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
    if use_cache and cache_file.exists():
        return _clean(pd.read_csv(cache_file, index_col=0, parse_dates=True), str(cache_file))

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}' ({start} to {end}). Check the symbol/date range.")

    # yfinance returns MultiIndex columns (field, ticker) even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = _clean(df, f"yahoo:{ticker}")
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(cache_file)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from a CSV whose first column is the date."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().title() for c in df.columns]
    return _clean(df, path)


def generate_synthetic(
    days: int = 2000,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    trend: float = 0.0,
    trend_halflife: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulated daily OHLCV.

    Daily log returns = constant drift + hidden trend + noise, where the
    hidden trend is a persistent AR(1) process (half-life `trend_halflife`
    days). `trend` sets its standard deviation as a fraction of daily
    volatility; noise is scaled down so total volatility stays ~`sigma`.
    trend=0 gives standard GBM. Even trend=0.1 is a strong, tradeable edge.

    Each day's return is split into an overnight gap (Close_{t-1} -> Open_t)
    and an intraday move (Open_t -> Close_t), so execution timing matters:
    a strategy that can only trade at the next open misses the gap.
    """
    if not 0 <= trend < 1:
        raise ValueError("trend must be in [0, 1)")

    rng = np.random.default_rng(seed)
    sd = sigma / np.sqrt(252)
    phi = 0.5 ** (1 / trend_halflife)

    shocks = rng.standard_normal(days) * trend * sd * np.sqrt(1 - phi**2)
    hidden = np.zeros(days)
    for t in range(1, days):
        hidden[t] = phi * hidden[t - 1] + shocks[t]

    noise = rng.standard_normal(days) * sd * np.sqrt(1 - trend**2)
    log_ret = (mu - 0.5 * sigma**2) / 252 + hidden + noise

    close = s0 * np.exp(np.cumsum(log_ret))
    prev_close = np.concatenate([[s0], close[:-1]])

    # ~30% of each day's move happens overnight, plus a little idiosyncratic gap noise
    overnight = 0.3 * log_ret + rng.normal(0, 0.1 * sd, days)
    open_ = prev_close * np.exp(overnight)

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.normal(0, 0.3 * sd, days)))
    low = body_lo * np.exp(-np.abs(rng.normal(0, 0.3 * sd, days)))
    volume = rng.lognormal(mean=15, sigma=0.3, size=days).round()

    # fixed start date so runs are reproducible regardless of when you run them
    index = pd.bdate_range(start="2012-01-02", periods=days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def load_price_data(
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    csv: str | None = None,
    **synthetic_kwargs,
) -> tuple[pd.DataFrame, str]:
    """
    Load prices from whichever source was requested. Returns (df, description).

    Deliberately does NOT silently fall back to synthetic data when a download
    fails: a typo in a ticker should be an error, not a backtest on random noise
    that looks like a real result.
    """
    if csv:
        return load_csv(csv), f"CSV {csv}"
    if ticker:
        if not (start and end):
            raise ValueError("--ticker requires --start and --end")
        return load_yahoo(ticker, start, end), f"{ticker} {start} to {end}"

    df = generate_synthetic(**synthetic_kwargs)
    trend = synthetic_kwargs.get("trend", 0.0)
    kind = "random walk (no edge exists)" if trend == 0 else f"hidden trend {trend} (planted edge)"
    return df, f"synthetic {kind}, {len(df)} days"
