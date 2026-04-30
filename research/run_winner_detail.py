"""Detailed report on the winning options config from the sweep.

Tests the same variant on three windows and splits in-sample / OOS to
catch overfit. Compares apples-to-apples to the live stock strategy
on the same windows.

Run:
    python -m research.run_winner_detail
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


# Winner from the sweep
WINNER = {
    "dte": 7,
    "conviction_min": 0.65,
    "theta_protect_mins": 0,
    "stop_pct": 0.50,
    "target_pct": 1.00,
}

# Test on three windows: full 6mo, recent 3mo (post-retrain OOS only), holdout April
WINDOWS = [
    ("Last 6 months",       "2025-11-01", "2026-04-29"),
    ("Post-retrain OOS",    "2026-01-27", "2026-04-29"),
    ("Apr 2026 only",       "2026-04-01", "2026-04-29"),
    ("Pre-retrain OOS",     "2025-11-01", "2026-01-26"),  # before model retrain
]


def _pull_vix(start, end):
    cfg = load_cfg("v1")
    out = build_frame(start, end, cfg)
    return out["vix_close"].dropna() if not out.empty else pd.Series(dtype=float)


def detail(label, trades):
    if not trades:
        print(f"  {label:<26}  no trades")
        return
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("et_date")["pnl_dollars"].sum()
    cum = daily.sort_index().cumsum()
    dd = cum - cum.cummax()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    n = len(df)
    print(f"  {label:<26}  n={n:>3}  win={float((df['pnl_dollars']>0).mean()*100):>5.1f}%  "
          f"$={df['pnl_dollars'].sum():+9,.0f}  $/tr={df['pnl_dollars'].mean():+6,.0f}  "
          f"Sharpe={sharpe:+5.2f}  DD=${dd.min() if len(dd) else 0:+,.0f}  "
          f"days={df['et_date'].nunique()}")
    # exit reasons
    reasons = df["reason"].value_counts()
    s = "    reasons: " + ", ".join(f"{r}={c}" for r, c in reasons.items())
    print(s)


def main():
    print("=" * 100)
    print(f" WINNER detail: dte={WINNER['dte']} conviction>={WINNER['conviction_min']} "
          f"stop={WINNER['stop_pct']:.0%} target={WINNER['target_pct']:.0%}")
    print("=" * 100)

    bundle = V3Bundle("mtf")

    for label, start, end in WINDOWS:
        print()
        print("-" * 100)
        print(f" {label}: {start} → {end}")
        print("-" * 100)
        signals, bars, _ = get_v3_signals_and_bars(
            start, end, bundle, add_vol=False, add_mtf=True)
        if not signals or bars.empty:
            print("  (no data)")
            continue
        vix = _pull_vix(start, end)
        cfg = OptionsSimConfig(vix_series=vix, **WINNER)
        sim = make_options_simulator(cfg)
        trades = walk_with_blocking(signals, bars, sim)
        detail(label, trades)

        # save trades
        if trades:
            out_dir = Path(__file__).parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            safe = f"{start}_{end}".replace("-", "")
            pd.DataFrame(trades).to_csv(out_dir / f"winner_{safe}.csv", index=False)


if __name__ == "__main__":
    main()
