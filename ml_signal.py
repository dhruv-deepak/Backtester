"""
ml_signal.py — ML-based trading signal via walk-forward classification.

The core idea being tested: can a classifier predict next-day direction
well enough, out-of-sample, to beat simple rule-based strategies net of
costs? Most student projects that attempt this make one of two mistakes:

  1. Train/test split with shuffling — this leaks future information into
     training (a model trained partly on 2023 data and tested on 2022 data
     is cheating, even if the split LOOKS like 70/30).
  2. Train once on the first chunk, evaluate on everything after — this is
     closer to legitimate, but ignores that market conditions drift over
     time (regime change), so a model trained once in 2015 may be stale by
     2023.

This module avoids both: it retrains periodically on an EXPANDING window
of only past data (walk-forward), so at any prediction point, the model
has only ever seen data strictly before that point.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from features import build_features

FEATURE_COLUMNS = None  # populated dynamically, excludes 'label'


def walk_forward_ml_signal(
    df: pd.DataFrame,
    min_train_size: int = 252,
    retrain_every: int = 20,
    prob_threshold: float = 0.55,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Generate trading signals using a walk-forward-retrained classifier.

    Parameters
    ----------
    min_train_size  : minimum number of rows before the first prediction
                       (need enough history to train a sane model)
    retrain_every    : retrain the model every N trading days (rather than
                        daily, for speed and to avoid overfitting to noise)
    prob_threshold   : only take a position if predicted P(up) is at least
                        this far from 0.5 — avoids forcing a trade on
                        every single day when the model is unsure
    n_estimators     : trees in the Random Forest

    Returns
    -------
    dict with:
        'signal'          : pd.Series of positions (+1 / 0 / -1), same
                             index as df, ready to feed into engine.Backtester
        'oos_accuracy'     : out-of-sample directional accuracy
        'oos_baseline'     : accuracy of always predicting the majority
                              class (the bar the model needs to beat)
        'n_predictions'    : number of out-of-sample predictions made
    """
    feat = build_features(df)
    feature_cols = [c for c in feat.columns if c != "label"]

    # Drop rows where features/label aren't computable (early window, last row)
    valid = feat.dropna()
    valid_index = valid.index

    signal = pd.Series(0, index=df.index, dtype=float)
    predictions = []
    actuals = []

    model = None
    scaler = None

    n = len(valid)
    for i in range(min_train_size, n):
        # Retrain only every `retrain_every` steps — reuse the model otherwise
        if model is None or (i - min_train_size) % retrain_every == 0:
            train = valid.iloc[:i]  # strictly past data only — expanding window
            X_train = train[feature_cols].values
            y_train = train["label"].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,          # shallow trees — resist overfitting on noisy financial data
                min_samples_leaf=20,  # require enough samples per leaf, same reason
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)

        # Predict for the CURRENT row using the model trained on data strictly before it
        row = valid.iloc[[i]]
        X_row = scaler.transform(row[feature_cols].values)
        prob_up = model.predict_proba(X_row)[0][1]

        pred_label = int(prob_up > 0.5)
        actual_label = int(row["label"].values[0])
        predictions.append(pred_label)
        actuals.append(actual_label)

        # Convert probability to a position, with a confidence threshold
        idx = valid_index[i]
        if prob_up >= prob_threshold:
            signal.loc[idx] = 1
        elif prob_up <= (1 - prob_threshold):
            signal.loc[idx] = -1
        else:
            signal.loc[idx] = 0  # not confident enough — stay flat

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    oos_accuracy = (predictions == actuals).mean() if len(actuals) > 0 else np.nan
    majority_class = pd.Series(actuals).mode()[0] if len(actuals) > 0 else np.nan
    oos_baseline = (actuals == majority_class).mean() if len(actuals) > 0 else np.nan

    return {
        "signal": signal,
        "oos_accuracy": oos_accuracy,
        "oos_baseline": oos_baseline,
        "n_predictions": len(actuals),
    }
