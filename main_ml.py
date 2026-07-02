"""
main_ml.py — Compare the ML-based walk-forward strategy against the
rule-based strategies and buy-and-hold, using the same backtest engine.

Usage:
    python main_ml.py                          # synthetic data (offline)
    python main_ml.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
"""

import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

from data import load_price_data
from strategies import STRATEGY_REGISTRY
from engine import Backtester
from metrics import summarize
from ml_signal import walk_forward_ml_signal


def main():
    parser = argparse.ArgumentParser(description="Compare ML signal vs rule-based strategies.")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cost_bps", type=float, default=5.0)
    parser.add_argument("--min_train_size", type=int, default=252)
    parser.add_argument("--retrain_every", type=int, default=20)
    parser.add_argument("--prob_threshold", type=float, default=0.55)
    args = parser.parse_args()

    price_df = load_price_data(ticker=args.ticker, start=args.start, end=args.end)
    print(f"Loaded {len(price_df)} rows of price data.\n")

    bt = Backtester(price_df, initial_capital=args.capital, cost_bps=args.cost_bps)

    results = {}
    summary_rows = {}

    # Benchmark
    bh_result = bt.run_buy_and_hold()
    results["Buy & Hold"] = bh_result
    summary_rows["Buy & Hold"] = summarize(bh_result["returns"])

    # Rule-based strategies
    for name, strategy_fn in STRATEGY_REGISTRY.items():
        signal = strategy_fn(price_df)
        result = bt.run(signal)
        results[name] = result
        summary_rows[name] = summarize(result["returns"])

    # ML strategy (walk-forward)
    print("Training walk-forward ML classifier (this retrains periodically, may take a moment)...")
    t0 = time.time()
    ml_result_raw = walk_forward_ml_signal(
        price_df,
        min_train_size=args.min_train_size,
        retrain_every=args.retrain_every,
        prob_threshold=args.prob_threshold,
    )
    print(f"Done in {time.time() - t0:.1f}s\n")

    ml_backtest = bt.run(ml_result_raw["signal"])
    results["ML Classifier (walk-forward)"] = ml_backtest
    summary_rows["ML Classifier (walk-forward)"] = summarize(ml_backtest["returns"])

    # --- Report classifier honesty check FIRST — this is the important number ---
    print("=" * 100)
    print("ML CLASSIFIER — OUT-OF-SAMPLE DIRECTIONAL ACCURACY (walk-forward, never sees future data)")
    print("=" * 100)
    print(f"  Model accuracy:      {ml_result_raw['oos_accuracy']*100:.2f}%")
    print(f"  Majority-class baseline: {ml_result_raw['oos_baseline']*100:.2f}%  (accuracy from always guessing the more common direction)")
    edge = (ml_result_raw['oos_accuracy'] - ml_result_raw['oos_baseline']) * 100
    print(f"  Edge over baseline:  {edge:+.2f} percentage points")
    print(f"  Predictions made:    {ml_result_raw['n_predictions']}")
    if edge <= 0:
        print("  -> The model shows NO genuine skill over the naive baseline on this data.")
    elif edge < 2:
        print("  -> Marginal edge — likely within noise; interpret Sharpe/returns with skepticism.")
    else:
        print("  -> Meaningful edge over baseline.")
    print("=" * 100)
    print()

    # --- Strategy comparison table ---
    summary_df = pd.DataFrame(summary_rows).T
    summary_df["CAGR"] = (summary_df["CAGR"] * 100).round(2).astype(str) + "%"
    summary_df["Annualized Volatility"] = (summary_df["Annualized Volatility"] * 100).round(2).astype(str) + "%"
    summary_df["Max Drawdown"] = (summary_df["Max Drawdown"] * 100).round(2).astype(str) + "%"
    summary_df["Win Rate"] = (summary_df["Win Rate"] * 100).round(1).astype(str) + "%"
    summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].round(2)
    summary_df["Sortino Ratio"] = summary_df["Sortino Ratio"].round(2)
    summary_df["Calmar Ratio"] = summary_df["Calmar Ratio"].round(2)

    print("=" * 100)
    print("STRATEGY PERFORMANCE COMPARISON (including ML classifier)")
    print("=" * 100)
    print(summary_df.to_string())
    print("=" * 100)

    # --- Plot ---
    plt.figure(figsize=(11, 6))
    for name, result in results.items():
        style = "--" if "ML" in name else "-"
        plt.plot(result["equity_curve"], label=name, linewidth=1.8 if "ML" in name else 1.3, linestyle=style)
    plt.title("Strategy Equity Curves (including walk-forward ML classifier)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curves_ml.png", dpi=150)
    print("\nSaved plot to equity_curves_ml.png")


if __name__ == "__main__":
    main()
