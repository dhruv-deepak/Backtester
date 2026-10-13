"""
engine.py — The core backtest loop (vectorised).

Timing model. A strategy's signal at row t is computed from data up to the
close of day t. You can't trade at a price you used to make the decision, so
the engine supports two fill assumptions:

  execution="next_open"  (default, realistic)
      Decide after the close of t, fill at the OPEN of t+1. On day t+1 the
      overnight gap (Close_t -> Open_t+1) is earned by the OLD position and
      the intraday move (Open_t+1 -> Close_t+1) by the NEW one.

  execution="close"      (optimistic)
      Fill at the same close the signal was computed from — what most simple
      backtesters implicitly assume. Useful to measure how much of a
      strategy's return depends on an unrealistic fill.

Costs per unit of position traded (as a fraction of equity):
  commission + half the bid/ask spread + slippage proportional to recent
  volatility (markets move more against you when they're volatile).
Carry costs per day held:
  short borrow fee on short exposure, financing on leverage above 1x.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0      # per unit traded
    half_spread_bps: float = 2.0     # paid crossing the spread, per unit traded
    vol_slippage: float = 0.02       # extra cost = vol_slippage * daily vol, per unit traded
    borrow_bps_annual: float = 50.0  # fee for holding short positions
    financing_bps_annual: float = 300.0  # cost of leverage above 1x

    def unit_cost(self, prices: pd.DataFrame) -> pd.Series:
        """Cost of trading one unit of equity on each day, using only info known before the fill."""
        daily_vol = prices["Close"].pct_change().rolling(20, min_periods=2).std()
        # the fill on day t happens before t's close, so use vol known at t-1
        known_vol = daily_vol.shift(1).fillna(0.0)
        fixed = (self.commission_bps + self.half_spread_bps) / 1e4
        return fixed + self.vol_slippage * known_vol


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series          # daily net returns
    gross_returns: pd.Series    # before costs
    positions: pd.Series        # position held during each day (after fills)
    costs: pd.Series            # trading + carry costs as a return drag
    trades: pd.DataFrame        # one row per round trip
    initial_capital: float = 100_000.0
    meta: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return self.initial_capital * (1 + self.returns).cumprod()

    def slice(self, start=None, end=None) -> "BacktestResult":
        """Restrict to a date window (e.g. out-of-sample period only)."""
        s = slice(start, end)
        returns = self.returns.loc[s]
        t = self.trades
        # keep every trade that overlaps the window
        trades = t[(t["exit"] >= returns.index[0]) & (t["entry"] <= returns.index[-1])] if len(t) else t
        return BacktestResult(
            self.name, returns, self.gross_returns.loc[s], self.positions.loc[s],
            self.costs.loc[s], trades, self.initial_capital, dict(self.meta),
        )


class Backtester:
    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        costs: CostModel | None = None,
        execution: str = "next_open",
    ):
        if execution not in ("next_open", "close"):
            raise ValueError("execution must be 'next_open' or 'close'")
        self.prices = prices
        self.initial_capital = initial_capital
        self.costs = costs or CostModel()
        self.execution = execution
        self._unit_cost = self.costs.unit_cost(prices)

    def run(self, signal: pd.Series, name: str = "strategy") -> BacktestResult:
        p = self.prices
        signal = signal.reindex(p.index).fillna(0.0).astype(float)

        held = signal.shift(1).fillna(0.0)  # position during day t = decision made at close t-1
        prev = held.shift(1).fillna(0.0)    # position during day t-1
        traded = (held - prev).abs()

        trade_cost = traded * self._unit_cost
        carry = (
            held.clip(upper=0).abs() * self.costs.borrow_bps_annual
            + (held.abs() - 1).clip(lower=0) * self.costs.financing_bps_annual
        ) / 1e4 / TRADING_DAYS

        if self.execution == "close":
            asset_ret = p["Close"].pct_change().fillna(0.0)
            gross = held * asset_ret
            net = (1 - trade_cost) * (1 + gross) - 1 - carry
        else:
            gap = (p["Open"] / p["Close"].shift(1) - 1).fillna(0.0)
            intraday = p["Close"] / p["Open"] - 1
            # old position rides the overnight gap, we pay costs at the open, new position rides the day
            gross = (1 + prev * gap) * (1 + held * intraday) - 1
            net = (1 + prev * gap - trade_cost) * (1 + held * intraday) - 1 - carry

        return BacktestResult(
            name=name,
            returns=net,
            gross_returns=gross,
            positions=held,
            costs=gross - net,
            trades=extract_trades(held, net),
            initial_capital=self.initial_capital,
            meta={"execution": self.execution},
        )


def extract_trades(positions: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """
    Split the position series into round trips: maximal runs of the same
    direction. Each trade's return compounds its daily net returns, plus the
    exit day when the next state is flat (that's when the exit cost — and,
    with next-open fills, the final overnight gap — lands).
    """
    side = np.sign(positions)
    run_id = (side != side.shift()).cumsum()
    idx = positions.index
    rows = []
    for _, run in side.groupby(run_id):
        direction = run.iloc[0]
        if direction == 0:
            continue
        start, end = idx.get_loc(run.index[0]), idx.get_loc(run.index[-1])
        stop = end + 1
        if stop < len(idx) and side.iloc[stop] == 0:
            stop += 1  # include the exit day
        trade_ret = float((1 + returns.iloc[start:stop]).prod() - 1)
        rows.append({
            "entry": run.index[0],
            "exit": idx[min(stop, len(idx)) - 1],
            "direction": "long" if direction > 0 else "short",
            "days": end - start + 1,
            "return": trade_ret,
        })
    return pd.DataFrame(rows, columns=["entry", "exit", "direction", "days", "return"])
