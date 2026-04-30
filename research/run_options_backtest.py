"""Backtest the options strategy on the same windows as run_recent_v3.

Identical entry signals (v3_mtf model, p ≥ 0.55), but each signal is
executed as a 0DTE ATM SPY option (call for long, put for short) priced
via Black-Scholes with VIX as IV. Exit logic mirrors live/options.py.

Two windows for comparison:
  • Apr 2026 (last month — fully OOS)
  • Nov 2025 → Apr 2026 (last 6 months, partly in-sample)

Compare these against research/run_recent_v3 (stock version) to see
whether the options leverage is worth the structurally lower hit rate.

Run:
    python -m research.run_options_backtest
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from utils.config import load as load_cfg
from research.backtest_compare import walk_with_blocking
from research.backtest_original import build_frame
from research.v3_combinations import V3Bundle, get_v3_signals_and_bars
from research.option_simulator import OptionsSimConfig, make_options_simulator


WINDOWS = [
    ("Last month (Apr 2026, OOS)",            "2026-04-01", "2026-04-29"),
    ("Last 6 months (Nov 2025 → Apr 2026)",    "2025-11-01", "2026-04-29"),
]


def _pull_vix(start: str, end: str) -> pd.Series:
    """Pull the VIX series we'll feed the simulator. Comes from the same
    assembled frame the model uses, so we know it's aligned to bars."""
    cfg = load_cfg("v1")
    out = build_frame(start, end, cfg)
    if out.empty or "vix_close" not in out.columns:
        return pd.Series(dtype=float)
    return out["vix_close"].dropna()


def _summarise(label: str, trades: list[dict]) -> None:
    if not trades:
        print(f"  {label}: no trades.")
        return
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date

    n = len(df)
    win_pct = (df["pnl_dollars"] > 0).mean() * 100
    tot_dollars = df["pnl_dollars"].sum()
    avg_per_trade = df["pnl_dollars"].mean()
    n_days = df["et_date"].nunique()

    daily = df.groupby("et_date")["pnl_dollars"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    cum = daily.sort_index().cumsum()
    running_peak = cum.cummax()
    drawdown = (cum - running_peak)
    max_dd = drawdown.min() if len(drawdown) else 0.0

    print(f"  {label}")
    print(f"  {'-'*80}")
    print(f"  trades              {n}")
    print(f"  win%                {win_pct:.1f}%")
    print(f"  total $             {tot_dollars:+,.0f}")
    print(f"  avg $/trade         {avg_per_trade:+,.2f}")
    print(f"  trading days        {n_days}")
    print(f"  avg trades/day      {n / n_days:.1f}")
    print(f"  daily-Sharpe (ann)  {sharpe:+.2f}")
    print(f"  max DD              {max_dd:+,.0f}")

    # exit-reason breakdown — diagnostic for the strategy
    reasons = df["reason"].value_counts()
    print(f"  exit reasons:")
    for r, count in reasons.items():
        avg = df[df["reason"] == r]["pnl_dollars"].mean()
        print(f"    {r:<16} n={count:<4} avg ${avg:+,.0f}")

    # in-sample / OOS split (model retrained 2026-01-27)
    train_cutoff = pd.Timestamp("2026-01-27", tz="UTC")
    in_s = df[df["entry_ts"] < train_cutoff]
    oos = df[df["entry_ts"] >= train_cutoff]
    if len(in_s) > 0 and len(oos) > 0:
        print()
        print(f"  IN-SAMPLE (before 2026-01-27):")
        print(f"    {len(in_s)} trades, win {(in_s['pnl_dollars']>0).mean()*100:.1f}%, "
              f"${in_s['pnl_dollars'].sum():+,.0f}")
        print(f"  OOS (after 2026-01-27):")
        print(f"    {len(oos)} trades, win {(oos['pnl_dollars']>0).mean()*100:.1f}%, "
              f"${oos['pnl_dollars'].sum():+,.0f}")


def main():
    print("=" * 100)
    print(" OPTIONS backtest — v3_mtf entries, 0DTE ATM SPY, BS-priced via VIX")
    print("=" * 100)

    bundle = V3Bundle("mtf")

    for label, start, end in WINDOWS:
        print()
        print("=" * 100)
        print(f" {label}: {start} → {end}")
        print("=" * 100)

        signals, bars, _ = get_v3_signals_and_bars(
            start, end, bundle, add_vol=False, add_mtf=True)

        if not signals or bars.empty:
            print("  no signals or bars.")
            continue

        vix = _pull_vix(start, end)
        if vix.empty:
            print("  no VIX data.")
            continue

        cfg = OptionsSimConfig(vix_series=vix)
        sim = make_options_simulator(cfg)
        trades = walk_with_blocking(signals, bars, sim)

        _summarise(label, trades)

        if trades:
            out_dir = Path(__file__).parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            safe = f"{start}_{end}".replace("-", "")
            pd.DataFrame(trades).to_csv(out_dir / f"options_{safe}.csv", index=False)


if __name__ == "__main__":
    main()
