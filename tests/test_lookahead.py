"""Every signal source must be causal — and the audit must actually catch cheaters."""

import numpy as np
import pandas as pd
import pytest

from backtester.features import build_features
from backtester.ml import walk_forward_ml
from backtester.strategies import STRATEGIES, volatility_target
from backtester.validation import LookaheadError, check_no_lookahead


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_strategies_are_causal(random_walk, key):
    check_no_lookahead(STRATEGIES[key].signal, random_walk)


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_vol_targeted_strategies_are_causal(random_walk, key):
    s = STRATEGIES[key]
    check_no_lookahead(lambda d: volatility_target(s.signal(d), d), random_walk)


def test_features_are_causal(random_walk):
    cols = [c for c in build_features(random_walk).columns if c != "label"]
    for col in cols:
        check_no_lookahead(lambda d, c=col: build_features(d)[c], random_walk, n_cuts=3)


def test_ml_signal_is_causal(random_walk):
    fn = lambda d: walk_forward_ml(d, model="logreg", min_train_size=300, retrain_every=50).signal
    check_no_lookahead(fn, random_walk, n_cuts=3, min_history=400)


# --- canaries: the audit is only worth something if it fails on real cheats ---

def test_audit_catches_future_return():
    df = pd.DataFrame({"Close": np.random.default_rng(0).lognormal(0, 0.01, 600).cumprod()},
                      index=pd.bdate_range("2020-01-01", periods=600))
    cheat = lambda d: np.sign(d["Close"].pct_change().shift(-1)).fillna(0)
    with pytest.raises(LookaheadError):
        check_no_lookahead(cheat, df)


def test_audit_catches_full_sample_normalisation(random_walk):
    # z-scoring with the WHOLE sample's mean/std quietly leaks the future
    def cheat(d):
        c = d["Close"]
        return np.sign((c - c.mean()) / c.std())
    with pytest.raises(LookaheadError):
        check_no_lookahead(cheat, random_walk)


def test_audit_catches_centered_window(random_walk):
    cheat = lambda d: (d["Close"] > d["Close"].rolling(21, center=True).mean()).astype(float)
    with pytest.raises(LookaheadError):
        check_no_lookahead(cheat, random_walk)
