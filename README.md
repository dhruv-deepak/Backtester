# Algorithmic Trading Backtester

A from-scratch backtesting engine for evaluating systematic trading strategies,
built to avoid the common correctness pitfalls (lookahead bias, ignoring
transaction costs) that make naive backtests misleadingly optimistic.


## What it does

- Loads historical price data (via `yfinance`) or generates synthetic
  Geometric Brownian Motion price series for offline testing
- Implements three rule-based strategies: Moving Average Crossover,
  Time-Series Momentum, and Bollinger-Band Mean Reversion
- Implements an **ML-based strategy**: a Random Forest classifier trained
  to predict next-day direction, retrained on a walk-forward expanding
  window so it never sees future data at any prediction point
- Runs each strategy through a backtest engine that:
  - Shifts signals forward by one day to prevent lookahead bias
  - Charges transaction costs (in basis points) on every position change
  - Compounds returns into a realistic equity curve
- Computes standard performance metrics: CAGR, annualized volatility,
  Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, win rate
- Reports the ML classifier's **out-of-sample accuracy against a
  majority-class baseline** — the honest question isn't "does it make
  money" but "does it actually predict better than guessing the more
  common direction," and the code reports both explicitly
- Outputs a comparison table across strategies plus a benchmark
  (buy-and-hold), and plots equity curves

## The ML strategy, specifically

`ml_signal.py` implements walk-forward classification:

1. Features are engineered from price/volume only (lagged returns,
   moving-average ratios, rolling volatility, RSI, volume ratio, Bollinger
   z-score) — see `features.py`
2. The model (Random Forest, intentionally shallow — `max_depth=5`,
   `min_samples_leaf=20` — to resist overfitting on noisy financial data)
   is retrained every `retrain_every` days on an **expanding window of only
   past data**
3. At each prediction point, the model has strictly never seen the row
   it's predicting, or anything after it
4. A confidence threshold (`prob_threshold`) means the strategy only takes
   a position when the model's predicted probability is meaningfully off
   0.5 — otherwise it stays flat rather than forcing a low-conviction trade

This design deliberately avoids the two most common mistakes in
"ML for trading" student projects: shuffled train/test splits (which leak
future information) and single static train-once models (which go stale
as market regimes shift).

## Why these design choices matter

Backtesting is notoriously easy to get *wrong* in ways that look right:

- **Lookahead bias**: if you use today's closing price to decide today's
  trade, you're cheating — you couldn't have known the close until the
  market closed. This engine lags signals by one day before applying them.
- **Ignoring costs**: a strategy that trades often can look great gross of
  costs and lose money net of costs. This engine charges costs on every
  position change, so turnover is penalized realistically.
- **Sharpe vs. Sortino vs. Calmar**: Sharpe penalizes all volatility
  (including upside surprises), Sortino only penalizes downside volatility,
  and Calmar measures return relative to worst-case drawdown. Reporting
  all three (rather than just Sharpe) gives a fuller risk picture.

## Usage

```bash
pip install -r requirements.txt

# Rule-based strategies only, synthetic data (no internet needed)
python main.py

# Rule-based strategies on real historical data
python main.py --ticker AAPL --start 2019-01-01 --end 2024-01-01
python main.py --ticker RELIANCE.NS --start 2020-01-01 --end 2024-01-01

# Include the ML classifier strategy (slower — retrains periodically)
python main_ml.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
python main_ml.py  # synthetic data, no internet needed

# ML strategy tuning knobs
python main_ml.py --ticker AAPL --start 2018-01-01 --end 2024-01-01 \
  --min_train_size 252 --retrain_every 20 --prob_threshold 0.55
```

## Project structure

```
backtester/
├── data.py        # price data loading (real + synthetic)
├── strategies.py  # rule-based signal generators
├── features.py     # feature engineering for the ML strategy
├── ml_signal.py    # walk-forward ML classifier signal
├── engine.py       # core backtest loop
├── metrics.py      # performance metrics
├── main.py         # orchestration + reporting (rule-based strategies)
├── main_ml.py       # orchestration + reporting (includes ML classifier)
└── requirements.txt
```

## Possible extensions

- Multi-asset portfolio backtesting with correlation-aware allocation
- Slippage modeling in addition to transaction costs
- Parameter sensitivity analysis (heatmaps of Sharpe across parameter grids)
- Statistical significance testing on Sharpe ratio (is it distinguishable
  from zero given the sample size?)
- Feature importance analysis on the ML classifier — which features
  actually drive predictions, and does that make economic sense?
- Try simpler models (logistic regression) alongside the Random Forest as
  a sanity check — added model complexity should earn its keep
