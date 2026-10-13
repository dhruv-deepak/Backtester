"""
optimize.py — Parameter search, done honestly.

grid_search()
    Backtests every parameter combination on the full sample. Useful for a
    sensitivity heatmap: a good strategy sits on a broad plateau, while an
    overfit one is a lone spike surrounded by losers. The best full-sample
    combo is IN-SAMPLE by definition, so its Sharpe must be deflated by the
    number of combos tried (see stats.deflated_sharpe_ratio).

walk_forward_optimize()
    The out-of-sample answer: repeatedly pick the best params on a training
    window, then trade them on the following unseen test window. Stitching
    the test windows together gives an equity curve made only of decisions
    that could have been made in real time.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import Backtester, BacktestResult
from .stats import _per_period_sharpe

TRADING_DAYS = 252


def _label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def grid_search(strategy, df: pd.DataFrame, bt: Backtester, transform=None):
    """
    Returns (table, results): one row per combo with its params and
    annualised Sharpe/CAGR, plus the BacktestResult for each combo.
    `transform(signal, df)` is an optional overlay such as vol targeting.
    """
    rows, results = [], []
    for params in strategy.combos():
        sig = strategy.signal(df, **params)
        if transform:
            sig = transform(sig, df)
        res = bt.run(sig, name=f"{strategy.name} ({_label(params)})")
        r = res.returns
        rows.append({**params, "Sharpe": _per_period_sharpe(r) * np.sqrt(TRADING_DAYS),
                     "CAGR": (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1})
        results.append(res)
    return pd.DataFrame(rows), results


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame            # per fold: dates, chosen params, in-sample vs out-of-sample Sharpe
    oos: BacktestResult            # stitched out-of-sample backtest
    efficiency: float              # OOS Sharpe / mean in-sample Sharpe of the chosen params


def walk_forward_optimize(
    strategy,
    df: pd.DataFrame,
    bt: Backtester,
    train_days: int = 756,
    test_days: int = 126,
    anchored: bool = False,
    transform=None,
) -> WalkForwardResult:
    """
    train_days : length of each optimisation window (rolling unless `anchored`)
    test_days  : length of each out-of-sample window that follows it
    anchored   : True = training window always starts at the first bar
    """
    if train_days + test_days > len(df):
        raise ValueError(f"need at least {train_days + test_days} rows for walk-forward, have {len(df)}")

    combos = strategy.combos()

    # Each combo's signal is computed once on the full history and then sliced.
    # This is only valid because strategies are causal — the value at t doesn't
    # depend on data after t — which validation.check_no_lookahead verifies.
    signals = []
    for params in combos:
        sig = strategy.signal(df, **params)
        signals.append(transform(sig, df) if transform else sig)
    returns = [bt.run(s).returns for s in signals]

    idx = df.index
    stitched = pd.Series(0.0, index=idx)
    folds = []
    for test_start in range(train_days, len(idx), test_days):
        train_lo = 0 if anchored else test_start - train_days
        is_slice = slice(train_lo, test_start)
        oos_slice = slice(test_start, min(test_start + test_days, len(idx)))

        is_sharpes = np.array([_per_period_sharpe(r.iloc[is_slice]) for r in returns])
        best = int(np.nanargmax(np.where(np.isfinite(is_sharpes), is_sharpes, -np.inf)))
        stitched.iloc[oos_slice] = signals[best].iloc[oos_slice]

        folds.append({
            "train_start": idx[train_lo].date(),
            "test_start": idx[oos_slice.start].date(),
            "test_end": idx[oos_slice.stop - 1].date(),
            "params": _label(combos[best]),
            "IS Sharpe": is_sharpes[best] * np.sqrt(TRADING_DAYS),
            "OOS Sharpe": _per_period_sharpe(returns[best].iloc[oos_slice]) * np.sqrt(TRADING_DAYS),
        })

    # run the stitched signal through the engine once, so the cost of switching
    # parameters at fold boundaries is charged properly
    full = bt.run(stitched, name=f"{strategy.name} (walk-forward)")
    oos = full.slice(idx[train_days], None)

    folds = pd.DataFrame(folds)
    oos_sharpe = _per_period_sharpe(oos.returns) * np.sqrt(TRADING_DAYS)
    mean_is = folds["IS Sharpe"].mean()
    # a ratio against a near-zero denominator is meaningless, so require a real in-sample edge
    efficiency = oos_sharpe / mean_is if mean_is > 0.1 else np.nan
    return WalkForwardResult(folds=folds, oos=oos, efficiency=efficiency)
