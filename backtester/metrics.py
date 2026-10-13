"""
metrics.py — Performance metrics.

Return-based metrics take a Series of daily net returns. summarize() also
uses the positions and trade log from a BacktestResult, so that trading
statistics (win rate, holding period, turnover) are measured per TRADE
rather than per day — a per-day win rate counts the cost-only exit day as a
"loss" and the many flat days as nothing, which distorts it badly.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return np.nan
    return growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe: mean excess return per unit of total volatility."""
    excess = returns - risk_free_rate / TRADING_DAYS
    sd = excess.std()
    if not sd > 0:
        return np.nan
    return excess.mean() / sd * np.sqrt(TRADING_DAYS)


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """
    Root-mean-square of shortfalls below the minimum acceptable return,
    averaged over ALL periods (gains count as zero shortfall).

    Note: this is NOT the std of the negative returns — that common shortcut
    measures dispersion *among* losing days around their own mean, so a
    strategy losing exactly 1% on every down day would show zero downside risk.
    """
    shortfall = np.minimum(returns - mar, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    daily_rf = risk_free_rate / TRADING_DAYS
    dd = downside_deviation(returns, daily_rf)
    if not dd > 0:
        return np.nan
    return (returns.mean() - daily_rf) / dd * np.sqrt(TRADING_DAYS)


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction."""
    return drawdown_series(returns).min()


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch (in trading days) spent below a previous equity high."""
    underwater = drawdown_series(returns) < 0
    runs = underwater.groupby((~underwater).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if not mdd < 0:
        return np.nan
    return cagr(returns) / abs(mdd)


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"Trades": 0, "Win Rate": np.nan, "Profit Factor": np.nan, "Avg Hold (days)": np.nan}
    r = trades["return"]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "Trades": len(trades),
        "Win Rate": (r > 0).mean(),
        "Profit Factor": gains / losses if losses > 0 else np.inf,
        "Avg Hold (days)": trades["days"].mean(),
    }


def summarize(result) -> dict:
    """All headline metrics for a BacktestResult."""
    r = result.returns
    out = {
        "CAGR": cagr(r),
        "Volatility": annualized_volatility(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Max Drawdown": max_drawdown(r),
        "Max DD Days": max_drawdown_duration(r),
        "Calmar": calmar_ratio(r),
        "Exposure": (result.positions != 0).mean(),
        "Turnover/yr": result.positions.diff().abs().sum() / len(r) * TRADING_DAYS,
        "Cost Drag/yr": result.costs.mean() * TRADING_DAYS,
    }
    out.update(trade_stats(result.trades))
    return out
