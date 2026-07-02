"""
main.py — Run all strategies against a price series, compare performance,
and produce a summary table + equity curve plot.

Usage:
    python main.py                          # uses synthetic data (offline)
    python main.py --ticker AAPL --start 2019-01-01 --end 2024-01-01
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from data import load_price_data
from strategies import STRATEGY_REGISTRY
from engine import Backtester
from metrics import summarize


def main():
    parser = argparse.ArgumentParser(description="Run backtests across strategies.")
    parser.add_argument("--ticker", type=str, default=None, help="e.g. AAPL, RELIANCE.NS")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cost_bps", type=float, default=5.0, help="transaction cost in bps")
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

    # Strategies
    for name, strategy_fn in STRATEGY_REGISTRY.items():
        signal = strategy_fn(price_df)
        result = bt.run(signal)
        results[name] = result
        summary_rows[name] = summarize(result["returns"])

    # --- Print comparison table ---
    summary_df = pd.DataFrame(summary_rows).T
    summary_df["CAGR"] = (summary_df["CAGR"] * 100).round(2).astype(str) + "%"
    summary_df["Annualized Volatility"] = (summary_df["Annualized Volatility"] * 100).round(2).astype(str) + "%"
    summary_df["Max Drawdown"] = (summary_df["Max Drawdown"] * 100).round(2).astype(str) + "%"
    summary_df["Win Rate"] = (summary_df["Win Rate"] * 100).round(1).astype(str) + "%"
    summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].round(2)
    summary_df["Sortino Ratio"] = summary_df["Sortino Ratio"].round(2)
    summary_df["Calmar Ratio"] = summary_df["Calmar Ratio"].round(2)

    print("=" * 100)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("=" * 100)
    print(summary_df.to_string())
    print("=" * 100)

    # --- Plot equity curves ---
    plt.figure(figsize=(11, 6))
    for name, result in results.items():
        plt.plot(result["equity_curve"], label=name, linewidth=1.5)
    plt.title("Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curves.png", dpi=150)
    print("\nSaved plot to equity_curves.png")


if __name__ == "__main__":
    main()
