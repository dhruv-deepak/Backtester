import numpy as np
import pandas as pd

from backtester.stats import (
    bootstrap_sharpe_ci, deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio,
)


def test_psr_is_calibrated_under_the_null():
    """With no true edge, PSR > 0.95 should happen ~5% of the time — not more."""
    rng = np.random.default_rng(0)
    hits = [probabilistic_sharpe_ratio(rng.normal(0, 0.01, 500)) > 0.95 for _ in range(2000)]
    assert 0.03 < np.mean(hits) < 0.07


def test_psr_detects_a_real_edge():
    r = np.random.default_rng(1).normal(0.001, 0.01, 2520)  # annual Sharpe ~1.6 over 10y
    assert probabilistic_sharpe_ratio(r) > 0.99


def test_expected_max_sharpe_grows_with_trials():
    rng = np.random.default_rng(2)
    small = expected_max_sharpe(rng.normal(0, 0.03, 5))
    large = expected_max_sharpe(rng.normal(0, 0.03, 500))
    assert 0 < small < large


def test_deflation_bites_on_the_best_of_many_noise_strategies():
    """Pick the best of 200 worthless strategies: PSR is fooled, DSR is not."""
    rng = np.random.default_rng(3)
    trials = rng.normal(0, 0.01, size=(200, 750))
    srs = trials.mean(1) / trials.std(1, ddof=1)
    best = trials[srs.argmax()]
    assert probabilistic_sharpe_ratio(best) > 0.95
    assert deflated_sharpe_ratio(best, srs) < 0.95


def test_bootstrap_ci_brackets_point_estimate():
    r = pd.Series(np.random.default_rng(4).normal(0.0005, 0.01, 1000))
    lo, hi = bootstrap_sharpe_ci(r)
    sr = r.mean() / r.std() * np.sqrt(252)
    assert lo < sr < hi
