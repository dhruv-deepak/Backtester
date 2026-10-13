"""
validation.py — Automated lookahead-bias detection.

Lookahead bias is the most common way backtests cheat, and it is invisible
in the results: the equity curve just looks great. Code review catches some
of it; this catches it empirically.

The test: a causal signal's value at row t must be identical whether it is
computed on the full history or on data truncated at t. If peeking at
future rows changes the past, the strategy is using information it wouldn't
have had live. Classic culprits: shift(-1), centered rolling windows,
normalising by full-sample mean/std, fitting a model on the whole dataset,
and bfill().
"""

from typing import Callable

import numpy as np
import pandas as pd


class LookaheadError(AssertionError):
    pass


def check_no_lookahead(
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    df: pd.DataFrame,
    n_cuts: int = 6,
    min_history: int = 260,
    atol: float = 1e-9,
) -> None:
    """
    Raise LookaheadError if `signal_fn` uses future data.

    Recomputes the signal on `n_cuts` truncated histories and checks every
    value up to each cut against the full-history signal.
    """
    full = signal_fn(df).reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    lo = min(min_history, len(df) // 2)
    cuts = np.unique(np.linspace(lo, len(df) - 2, n_cuts).astype(int))

    for cut in cuts:
        partial = signal_fn(df.iloc[: cut + 1]).reindex(df.index[: cut + 1]).fillna(0.0).to_numpy(dtype=float)
        bad = np.flatnonzero(~np.isclose(partial, full[: cut + 1], atol=atol, rtol=0))
        if len(bad):
            first = bad[0]
            raise LookaheadError(
                f"signal at {df.index[first].date()} is {partial[first]:.4g} with data up to "
                f"{df.index[cut].date()} but {full[first]:.4g} with the full history — "
                f"it depends on future data"
            )
