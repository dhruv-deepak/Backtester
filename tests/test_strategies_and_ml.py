import numpy as np
import pandas as pd

from backtester.engine import Backtester
from backtester.ml import walk_forward_ml
from backtester.optimize import walk_forward_optimize
from backtester.stats import probabilistic_sharpe_ratio
from backtester.strategies import STRATEGIES, mean_reversion, volatility_target


def test_positive_control_trend_strategies_find_planted_edge(trending):
    bt = Backtester(trending)
    for key in ["momentum", "breakout"]:
        r = bt.run(STRATEGIES[key].signal(trending)).returns
        assert probabilistic_sharpe_ratio(r) > 0.95, key


def test_mean_reversion_holds_until_reversion():
    close = [100.0, 101.0] * 10 + [90, 92, 95, 99, 102, 103]
    df = pd.DataFrame({"Close": close}, index=pd.bdate_range("2024-01-01", periods=len(close)))
    pos = mean_reversion(df, window=20, entry_z=2.0)
    assert pos.iloc[20] == 1             # entered on the plunge
    assert (pos.iloc[21:23] == 1).all()  # still holding while below the mean
    assert pos.iloc[-1] == 0             # exited after reverting


def test_vol_target_hits_target(trending):
    sig = pd.Series(1.0, index=trending.index)
    sized = volatility_target(sig, trending, target_vol=0.10, max_leverage=10)
    r = (sized.shift(1) * trending["Close"].pct_change()).iloc[100:]
    assert abs(r.std() * np.sqrt(252) - 0.10) < 0.02


def test_ml_predicts_live_bar_and_scores_only_known_labels(random_walk):
    res = walk_forward_ml(random_walk, model="logreg", min_train_size=300, retrain_every=50)
    assert not np.isnan(res.predictions["prob_up"].iloc[-1])  # the bar you'd trade tomorrow...
    assert np.isnan(res.predictions["label"].iloc[-1])        # ...whose outcome is unknown
    assert res.n_predictions == len(res.predictions) - 1
    assert res.p_value > 0.05  # no edge in a random walk


def test_walk_forward_is_out_of_sample(trending):
    bt = Backtester(trending)
    wf = walk_forward_optimize(STRATEGIES["momentum"], trending, bt, train_days=756, test_days=252)
    assert wf.oos.returns.index[0] == trending.index[756]
    assert (pd.to_datetime(wf.folds["train_start"]) < pd.to_datetime(wf.folds["test_start"])).all()
    assert len(wf.folds) == int(np.ceil((len(trending) - 756) / 252))
