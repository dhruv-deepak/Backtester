"""
metrics.py — Standard quant performance metrics.

These are the numbers an interviewer will actually ask you to explain:
Sharpe, Sortino, max drawdown, CAGR, win rate, volatility. Each function
takes a Series of periodic (daily) strategy returns.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate implied by a daily return series."""
    equity_curve = (1 + returns).cumprod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0 or equity_curve.iloc[-1] <= 0:
        return np.nan
    return equity_curve.iloc[-1] ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of total volatility.
    risk_free_rate is annualized; converted to a daily rate internally.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Like Sharpe, but only penalizes downside volatility — answers
    "how much return am I getting per unit of BAD risk?"
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    """
    Largest peak-to-trough decline in the equity curve, as a negative
    fraction (e.g. -0.23 means a 23% drawdown at the worst point).
    """
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def win_rate(returns: pd.Series) -> float:
    """Fraction of active (non-zero) days that were profitable."""
    active = returns[returns != 0]
    if len(active) == 0:
        return np.nan
    return (active > 0).mean()


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by max drawdown magnitude — return per unit of worst-case pain."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr(returns) / abs(mdd)


def summarize(returns: pd.Series) -> dict:
    """Compute all metrics at once, returned as a dict for easy tabulation."""
    return {
        "CAGR": cagr(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "Win Rate": win_rate(returns),
    }
