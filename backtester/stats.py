"""
stats.py — Is the result real, or is it luck?

A Sharpe ratio is an ESTIMATE from a finite, noisy, fat-tailed sample. Three
tools to put an error bar on it:

  probabilistic_sharpe_ratio (PSR)
      P(true Sharpe > benchmark), given the observed Sharpe, the sample
      length, and the skewness/kurtosis of returns (negative skew and fat
      tails make a Sharpe estimate less trustworthy).
      Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".

  deflated_sharpe_ratio (DSR)
      PSR, but the benchmark is the Sharpe you'd EXPECT the best of N
      worthless strategies to show by pure luck. If you try 50 parameter
      combos and report the best, you must clear this bar — selection bias
      is the #1 way backtests lie.
      Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".

  bootstrap_sharpe_ci
      Confidence interval via a circular block bootstrap. Blocks (rather than
      resampling single days) preserve volatility clustering and
      autocorrelation, which an iid bootstrap would destroy.

All Sharpe ratios inside this module are PER-PERIOD (daily, not annualised),
because that is the unit the PSR formula is derived in.
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def _per_period_sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return r.mean() / sd if sd > 0 else np.nan


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """
    P(true Sharpe > sr_benchmark). `sr_benchmark` is per-period (daily);
    divide an annual figure by sqrt(252).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sr = _per_period_sharpe(r)
    if n < 3 or np.isnan(sr):
        return np.nan
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # raw kurtosis: 3 for a normal distribution
    var = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if var <= 0:
        return np.nan
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(var)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(trial_sharpes) -> float:
    """
    Expected maximum per-period Sharpe among N independent trials whose
    TRUE Sharpe is zero, given the observed spread of the trials' Sharpes
    (False Strategy Theorem). Grows with the number of trials.
    """
    srs = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n = len(srs)
    if n < 2:
        return 0.0
    sd = srs.std(ddof=1)
    return float(sd * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    ))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes) -> float:
    """
    P(true Sharpe > what the best of N lucky trials would show).
    `trial_sharpes`: per-period Sharpes of EVERY variant tried, including
    this one. Rule of thumb: DSR > 0.95 is evidence of a real edge.
    """
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(trial_sharpes))


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    ci: float = 0.95,
    block: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Annualised Sharpe confidence interval from a circular block bootstrap."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 10 or not r.std() > 0:
        return (np.nan, np.nan)
    block = block or max(1, round(n ** (1 / 3)))
    n_blocks = -(-n // block)  # ceil

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    samples = r[idx.reshape(n_boot, -1)[:, :n]]

    sd = samples.std(axis=1, ddof=1)
    sharpes = np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1), 0.0) * np.sqrt(TRADING_DAYS)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(sharpes, [alpha, 1 - alpha])
    return float(lo), float(hi)


def robustness_report(results, trial_returns=None) -> pd.DataFrame:
    """
    Sharpe, its bootstrap CI, PSR and DSR for each BacktestResult.
    `trial_returns`: returns of every variant tried (for DSR); defaults to `results`.
    """
    trials = trial_returns if trial_returns is not None else [r.returns for r in results]
    trial_srs = [_per_period_sharpe(t) for t in trials]
    sr0 = expected_max_sharpe(trial_srs)

    rows = {}
    for res in results:
        r = res.returns
        lo, hi = bootstrap_sharpe_ci(r)
        rows[res.name] = {
            "Sharpe": _per_period_sharpe(r) * np.sqrt(TRADING_DAYS),
            "95% CI low": lo,
            "95% CI high": hi,
            "PSR (SR>0)": probabilistic_sharpe_ratio(r),
            "DSR": probabilistic_sharpe_ratio(r, sr0),
        }
    df = pd.DataFrame(rows).T
    df.attrs["n_trials"] = len(trial_srs)
    df.attrs["sr0_annual"] = sr0 * np.sqrt(TRADING_DAYS)
    return df
