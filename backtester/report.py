"""
report.py — Console tables and charts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown_series

TRADING_DAYS = 252
LINE = "=" * 110

PCT = {"CAGR", "Volatility", "Max Drawdown", "Exposure", "Win Rate", "Cost Drag/yr"}
INT = {"Max DD Days", "Trades", "Avg Hold (days)"}


def format_table(df: pd.DataFrame) -> str:
    out = df.copy().astype(object)
    for col in df.columns:
        for i, v in df[col].items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.at[i, col] = "-"
            elif col in PCT:
                out.at[i, col] = f"{v * 100:.1f}%"
            elif col in INT:
                out.at[i, col] = f"{v:.0f}"
            elif isinstance(v, (float, np.floating)):
                out.at[i, col] = "inf" if np.isinf(v) else f"{v:.2f}"
    return out.to_string()


def section(title: str, body: str) -> None:
    print(LINE)
    print(title)
    print(LINE)
    print(body)
    print()


def plot_tearsheet(results, path: Path, title: str) -> None:
    """Equity (log scale), drawdown, and rolling 6-month Sharpe for each result."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    for res in results:
        bench = res.name == "Buy & Hold"
        style = dict(lw=1.2 if not bench else 1.6, color="black" if bench else None,
                     ls="--" if bench else "-", label=res.name)
        axes[0].plot(res.equity, **style)
        axes[1].plot(drawdown_series(res.returns) * 100, **style)
        r = res.returns
        roll = r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(TRADING_DAYS)
        axes[2].plot(roll, **style)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Equity ($, log)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].set_ylabel("Rolling 6m Sharpe")
    axes[2].axhline(0, color="grey", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, params: list[str], path: Path, title: str) -> None:
    """Sharpe across a parameter grid (first two params; best over any others)."""
    if len(params) == 1:
        pivot = table.set_index(params[0])[["Sharpe"]].T
    else:
        pivot = table.pivot_table(index=params[0], columns=params[1], values="Sharpe", aggfunc="max")

    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 3, 0.8 * pivot.shape[0] + 2))
    lim = np.nanmax(np.abs(pivot.to_numpy())) or 1
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_xlabel(params[1] if len(params) > 1 else params[0])
    if len(params) > 1:
        ax.set_ylabel(params[0])
    for (i, j), v in np.ndenumerate(pivot.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Sharpe")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_walk_forward(oos, benchmark, folds: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(oos.equity / oos.equity.iloc[0], label=oos.name)
    ax.plot(benchmark.equity / benchmark.equity.iloc[0], label="Buy & Hold", color="black", ls="--")
    for d in folds["test_start"]:
        ax.axvline(pd.Timestamp(d), color="grey", lw=0.5, alpha=0.5)
    ax.set_ylabel("Growth of $1 (out-of-sample only)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
