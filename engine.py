"""
engine.py — The core backtest loop.

This is the part that matters most in an interview: it's easy to write a
backtester that LOOKS right but silently cheats (uses today's close to
decide today's trade, ignores transaction costs, etc.). This engine is
explicit about avoiding that.

Key design decisions:
  1. Signals are shifted forward by 1 day before being applied — you can
     only act on information *after* it's known, and the resulting
     position only earns the *next* day's return.
  2. Transaction costs are charged whenever the position changes size,
     proportional to the value traded (in basis points).
  3. Returns are computed on a fixed initial capital basis, then compounded,
     so you can see the actual equity curve, not just returns.
"""

import pandas as pd


class Backtester:
    def __init__(self, price_df: pd.DataFrame, initial_capital: float = 100_000.0, cost_bps: float = 5.0):
        """
        price_df        : DataFrame with a 'Close' column, indexed by date
        initial_capital : starting portfolio value
        cost_bps        : transaction cost in basis points (1 bps = 0.01%)
                           charged on the value of every position CHANGE
        """
        self.price_df = price_df
        self.initial_capital = initial_capital
        self.cost_bps = cost_bps

    def run(self, signal: pd.Series) -> dict:
        """
        Run the backtest for a given signal series (+1 / 0 / -1 positions).
        Returns a dict with:
            'returns'       : daily strategy returns (net of costs)
            'equity_curve'  : cumulative portfolio value over time
            'positions'     : the actual (lagged) positions held each day
            'turnover'      : total position changes (useful sanity check)
        """
        signal = signal.reindex(self.price_df.index).fillna(0)

        # --- avoid lookahead: you decide on day t's close, you're positioned from day t+1 ---
        position = signal.shift(1).fillna(0)

        # daily asset returns
        asset_returns = self.price_df["Close"].pct_change().fillna(0)

        # gross strategy return = position * next day's asset return
        gross_returns = position * asset_returns

        # transaction costs: charged when position changes, proportional to size of change
        position_change = position.diff().abs().fillna(position.abs())
        cost_rate = self.cost_bps / 10_000
        costs = position_change * cost_rate

        net_returns = gross_returns - costs

        equity_curve = self.initial_capital * (1 + net_returns).cumprod()

        return {
            "returns": net_returns,
            "equity_curve": equity_curve,
            "positions": position,
            "turnover": position_change.sum(),
        }

    def run_buy_and_hold(self) -> dict:
        """Benchmark: buy on day 1, hold throughout. No signal needed."""
        always_long = pd.Series(1, index=self.price_df.index)
        return self.run(always_long)
