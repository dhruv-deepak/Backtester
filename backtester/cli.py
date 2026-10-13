"""
cli.py — Command-line interface.

    python main.py compare  [data options] [--ml] [--vol_target 0.15]
    python main.py optimize --strategy ma [data options]

Run `python main.py <command> -h` for all options.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import pandas as pd

from .data import load_price_data
from .engine import Backtester, CostModel
from .metrics import summarize
from .ml import MODELS, walk_forward_ml
from .optimize import grid_search, walk_forward_optimize
from .report import format_table, plot_heatmap, plot_tearsheet, plot_walk_forward, section
from .stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio, robustness_report
from .strategies import BUY_AND_HOLD, STRATEGIES, volatility_target
from .validation import LookaheadError, check_no_lookahead


def _common_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    d = p.add_argument_group("data")
    d.add_argument("--ticker", help="Yahoo symbol, e.g. AAPL, SPY, RELIANCE.NS")
    d.add_argument("--start", help="YYYY-MM-DD")
    d.add_argument("--end", help="YYYY-MM-DD")
    d.add_argument("--csv", help="local OHLCV CSV instead of Yahoo")
    d.add_argument("--days", type=int, default=2000, help="synthetic: number of days")
    d.add_argument("--trend", type=float, default=0.0,
                   help="synthetic: hidden-trend strength; 0 = random walk, e.g. 0.1 = planted edge")
    d.add_argument("--seed", type=int, default=42, help="synthetic: RNG seed")

    e = p.add_argument_group("execution & costs")
    e.add_argument("--capital", type=float, default=100_000.0)
    e.add_argument("--execution", choices=["next_open", "close"], default="next_open")
    e.add_argument("--commission_bps", type=float, default=1.0)
    e.add_argument("--spread_bps", type=float, default=2.0, help="half spread, bps")
    e.add_argument("--vol_slippage", type=float, default=0.02, help="slippage as a fraction of daily vol")
    e.add_argument("--borrow_bps", type=float, default=50.0, help="annual short borrow fee")
    e.add_argument("--vol_target", type=float, default=None,
                   help="scale positions to this annual vol, e.g. 0.15 (default: off)")
    e.add_argument("--max_leverage", type=float, default=2.0)

    p.add_argument("--out", default="output", help="directory for charts")
    return p


def _setup(args):
    prices, desc = load_price_data(
        ticker=args.ticker, start=args.start, end=args.end, csv=args.csv,
        days=args.days, trend=args.trend, seed=args.seed,
    )
    costs = CostModel(
        commission_bps=args.commission_bps, half_spread_bps=args.spread_bps,
        vol_slippage=args.vol_slippage, borrow_bps_annual=args.borrow_bps,
    )
    bt = Backtester(prices, args.capital, costs, args.execution)
    overlay = (partial(volatility_target, target_vol=args.vol_target, max_leverage=args.max_leverage)
               if args.vol_target else None)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    sizing = f"vol target {args.vol_target:.0%}" if args.vol_target else "unit size"
    print(f"\nData: {desc} | {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} bars)")
    print(f"Fills: {args.execution} | costs: {args.commission_bps}bp comm + {args.spread_bps}bp half-spread "
          f"+ {args.vol_slippage} x daily vol | sizing: {sizing}\n")
    return prices, bt, overlay, out


def _with_overlay(fn, overlay):
    return (lambda df: overlay(fn(df), df)) if overlay else fn


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

def cmd_compare(args) -> None:
    prices, bt, overlay, out = _setup(args)
    close_bt = Backtester(prices, args.capital, bt.costs, "close")

    signals = {}
    for key, strat in STRATEGIES.items():
        fn = _with_overlay(strat.signal, overlay)
        try:
            check_no_lookahead(fn, prices)
        except LookaheadError as e:
            raise SystemExit(f"Lookahead audit FAILED for {strat.name}: {e}")
        signals[strat.name] = fn(prices)
    print(f"Lookahead audit: all {len(signals)} rule-based strategies passed (truncation-invariance check)\n")

    ml = None
    if args.ml:
        print(f"Training walk-forward {args.model} classifier...", end=" ", flush=True)
        t0 = time.time()
        ml = walk_forward_ml(prices, model=args.model, min_train_size=args.min_train,
                             retrain_every=args.retrain_every, prob_threshold=args.prob_threshold)
        print(f"done in {time.time() - t0:.1f}s\n")
        sig = ml.signal
        signals[f"ML ({args.model})"] = overlay(sig, prices) if overlay else sig

    signals = {BUY_AND_HOLD.name: _with_overlay(BUY_AND_HOLD.signal, overlay)(prices), **signals}
    results = [bt.run(sig, name) for name, sig in signals.items()]

    perf = pd.DataFrame({r.name: summarize(r) for r in results}).T
    section("PERFORMANCE (net of all costs)", format_table(perf))

    # every strategy shown counts as a trial for the deflated Sharpe
    rob = robustness_report(results, trial_returns=[r.returns for r in results[1:]])
    rob["Sharpe @close fill"] = [summarize(close_bt.run(signals[n], n))["Sharpe"] for n in rob.index]
    section(
        "IS IT REAL?  (PSR = P(true Sharpe > 0);  DSR = same, but the bar is the best Sharpe "
        f"{rob.attrs['n_trials']} worthless strategies would show by luck: {rob.attrs['sr0_annual']:.2f})",
        format_table(rob) + "\n\nRule of thumb: PSR/DSR > 0.95 is evidence of skill. A CI that includes 0 means "
        "the data can't distinguish the strategy from no edge.\n'Sharpe @close fill' shows how much a result "
        "leans on the unrealistic assumption of trading at the decision bar's close.",
    )

    if ml is not None:
        lines = [
            f"Out-of-sample predictions: {ml.n_predictions}",
            f"Accuracy:                  {ml.accuracy:.2%}",
            f"Majority-class baseline:   {ml.baseline:.2%}   (always guess the more common direction)",
            f"Binomial test p-value:     {ml.p_value:.3f}   (H0: no better than baseline)",
            f"AUC:                       {ml.auc:.3f}   (0.5 = no ranking skill)",
        ]
        verdict = ("Statistically significant skill (p < 0.05)." if ml.p_value < 0.05 else
                   "NOT significant: the accuracy is consistent with guessing. Treat its P&L as noise.")
        lines.append(f"-> {verdict}")
        if ml.feature_importance is not None:
            top = ", ".join(f"{k} ({v:.0%})" for k, v in ml.feature_importance.head(5).items())
            lines.append(f"Top features: {top}")
        section(f"ML CLASSIFIER ({args.model}, walk-forward, retrained every {args.retrain_every}d)",
                "\n".join(lines))

    path = out / "compare.png"
    plot_tearsheet(results, path, "Strategy comparison (net of costs)")
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------
# optimize
# --------------------------------------------------------------------------

def cmd_optimize(args) -> None:
    prices, bt, overlay, out = _setup(args)
    strat = STRATEGIES[args.strategy]
    params = list(strat.grid)

    # 1. full-sample grid: sensitivity + selection-bias-adjusted significance of the winner
    table, results = grid_search(strat, prices, bt, transform=overlay)
    best_i = int(table["Sharpe"].idxmax())
    best = results[best_i]
    trial_returns = [r.returns for r in results]
    psr = probabilistic_sharpe_ratio(best.returns)
    dsr = deflated_sharpe_ratio(best.returns, [r.mean() / r.std() for r in trial_returns])

    top = table.sort_values("Sharpe", ascending=False).head(8).reset_index(drop=True)
    body = format_table(top) + (
        f"\n\nCombos tried: {len(table)} | positive Sharpe: {(table['Sharpe'] > 0).mean():.0%} "
        f"| median Sharpe: {table['Sharpe'].median():.2f}"
        f"\nBest in-sample: {best.name}  Sharpe {table['Sharpe'].iloc[best_i]:.2f}"
        f"\n  PSR (ignores that we searched):  {psr:.3f}"
        f"\n  DSR (corrects for {len(table)} trials):    {dsr:.3f}"
        + ("\n  -> survives the multiple-testing correction." if dsr > 0.95 else
           "\n  -> does NOT survive the multiple-testing correction: the best combo is plausibly luck.")
    )
    section(f"PARAMETER GRID: {strat.name} (full sample = in-sample)", body)
    plot_heatmap(table, params, out / f"heatmap_{args.strategy}.png", f"{strat.name}: Sharpe by parameters")

    # 2. walk-forward: what you would actually have earned re-optimising in real time
    wf = walk_forward_optimize(strat, prices, bt, args.train_days, args.test_days, args.anchored, overlay)
    bench = bt.run(BUY_AND_HOLD.signal(prices), BUY_AND_HOLD.name).slice(wf.oos.returns.index[0], None)
    perf = pd.DataFrame({r.name: summarize(r) for r in [bench, wf.oos]}).T
    cols = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Trades", "Win Rate", "Cost Drag/yr"]
    body = (
        format_table(wf.folds.set_index("test_start")) + "\n\n" + format_table(perf[cols])
        + f"\n\nWalk-forward efficiency (OOS Sharpe / avg IS Sharpe): {wf.efficiency:.2f}"
        + "   (~0.5+ is healthy; near 0 or negative = in-sample results were mostly overfitting; '-' = no in-sample edge to compare against)"
        + f"\nOOS PSR (SR>0): {probabilistic_sharpe_ratio(wf.oos.returns):.3f}"
    )
    mode = "anchored" if args.anchored else "rolling"
    section(f"WALK-FORWARD ({mode} {args.train_days}d train / {args.test_days}d test, out-of-sample only)", body)
    plot_walk_forward(wf.oos, bench, wf.folds, out / f"walkforward_{args.strategy}.png",
                      f"{strat.name}: walk-forward out-of-sample equity")
    print(f"Charts saved to {out / f'heatmap_{args.strategy}.png'} and {out / f'walkforward_{args.strategy}.png'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="backtester", description="Backtest trading strategies, honestly.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = _common_args()

    c = sub.add_parser("compare", parents=[common], help="compare all strategies vs buy & hold")
    c.add_argument("--ml", action="store_true", help="include the walk-forward ML classifier")
    c.add_argument("--model", choices=list(MODELS), default="logreg")
    c.add_argument("--min_train", type=int, default=504)
    c.add_argument("--retrain_every", type=int, default=21)
    c.add_argument("--prob_threshold", type=float, default=0.53)
    c.set_defaults(func=cmd_compare)

    o = sub.add_parser("optimize", parents=[common], help="grid search + walk-forward for one strategy")
    o.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    o.add_argument("--train_days", type=int, default=756)
    o.add_argument("--test_days", type=int, default=126)
    o.add_argument("--anchored", action="store_true")
    o.set_defaults(func=cmd_optimize)

    args = parser.parse_args(argv)
    args.func(args)
