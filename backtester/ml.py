"""
ml.py — ML trading signal via walk-forward classification.

Can a classifier predict next-day direction well enough, out of sample, to
beat simple rules net of costs? The two classic mistakes this avoids:

  1. Shuffled train/test splits, which train on the future and test on the past.
  2. Train-once models, which go stale as market regimes drift.

Instead the model is refit every `retrain_every` days on an expanding (or
rolling) window of strictly past data, then predicts the next block of days.

Label timing: row j's label is the return from close j to close j+1, which
is only KNOWN at close j+1. A model fit at the close of day s can therefore
train on rows j < s — never on row s itself.

The honest question isn't "did it make money" but "did it predict better
than chance?" — answered with a binomial test against the majority-class
baseline, plus AUC (threshold-free ranking skill, 0.5 = none).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features

MODELS = {
    # strongly regularised linear baseline — added complexity must beat this
    "logreg": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
    # shallow trees with big leaves: financial labels are mostly noise
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=50, max_features="sqrt",
        random_state=seed, n_jobs=-1,
    ),
    "gbm": lambda seed: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=80,
        l2_regularization=1.0, random_state=seed,
    ),
}


@dataclass
class MLResult:
    signal: pd.Series             # target positions, ready for the engine
    predictions: pd.DataFrame     # per out-of-sample row: prob_up, label
    accuracy: float
    baseline: float               # accuracy of always guessing the majority class
    p_value: float                # binomial test: is accuracy > baseline by more than chance?
    auc: float
    n_predictions: int
    feature_importance: pd.Series | None


def _importance(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "steps"):  # logreg pipeline: |coef| on standardised features
        return np.abs(model.steps[-1][1].coef_[0])
    return None


def walk_forward_ml(
    df: pd.DataFrame,
    model: str = "rf",
    min_train_size: int = 504,
    retrain_every: int = 21,
    train_window: int | None = None,
    prob_threshold: float = 0.53,
    random_state: int = 42,
) -> MLResult:
    """
    min_train_size : predictable rows required before the first prediction
    retrain_every  : refit cadence in trading days
    train_window   : None = expanding window; N = only the most recent N rows
    prob_threshold : go long if P(up) >= this, short if <= 1 - this, else flat
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")

    feat = build_features(df)
    cols = [c for c in feat.columns if c != "label"]
    X = feat[cols].to_numpy()
    y = feat["label"].to_numpy()

    rows = np.flatnonzero(~np.isnan(X).any(axis=1))  # rows where every feature exists
    prob = np.full(len(df), np.nan)
    importances = []

    for b in range(min_train_size, len(rows), retrain_every):
        block = rows[b:b + retrain_every]
        train = rows[rows < block[0]]
        train = train[~np.isnan(y[train])]
        if train_window:
            train = train[-train_window:]

        if len(np.unique(y[train])) < 2:
            prob[block] = y[train].mean()
            continue

        clf = MODELS[model](random_state)
        clf.fit(X[train], y[train])
        prob[block] = clf.predict_proba(X[block])[:, 1]
        imp = _importance(clf)
        if imp is not None:
            importances.append(imp / imp.sum())

    prob_s = pd.Series(prob, index=df.index)
    signal = pd.Series(0.0, index=df.index)
    signal[prob_s >= prob_threshold] = 1.0
    signal[prob_s <= 1 - prob_threshold] = -1.0

    # score only rows that were predicted AND whose outcome is known
    scored = ~np.isnan(prob) & ~np.isnan(y)
    p, actual = prob[scored], y[scored].astype(int)
    pred = (p > 0.5).astype(int)
    n = len(actual)
    if n:
        correct = int((pred == actual).sum())
        majority_rate = max(actual.mean(), 1 - actual.mean())
        p_value = binomtest(correct, n, majority_rate, alternative="greater").pvalue
        auc = roc_auc_score(actual, p) if len(np.unique(actual)) == 2 else np.nan
        accuracy = correct / n
    else:
        majority_rate = p_value = auc = accuracy = np.nan

    return MLResult(
        signal=signal,
        predictions=pd.DataFrame({"prob_up": prob_s, "label": feat["label"]}).dropna(subset=["prob_up"]),
        accuracy=accuracy,
        baseline=majority_rate,
        p_value=p_value,
        auc=auc,
        n_predictions=n,
        feature_importance=(
            pd.Series(np.mean(importances, axis=0), index=cols).sort_values(ascending=False)
            if importances else None
        ),
    )
