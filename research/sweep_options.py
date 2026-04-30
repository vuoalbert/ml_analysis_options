"""Grid sweep over DTE × conviction × theta_protect for the options strategy.

Loads signals + bars + VIX once per window, then loops over the grid.
Prints a leaderboard sorted by total $ P&L on the 6-month window.

Run:
    python -m research.sweep_options
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
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


# 6-month window: same as the headline run
WINDOW = ("Last 6 months", "2025-11-01", "2026-04-29")
# Validation: make sure best variant doesn't blow up on the held-out month
HOLDOUT = ("Apr 2026 OOS", "2026-04-01", "2026-04-29")


def _pull_vix(start: str, end: str) -> pd.Series:
    cfg = load_cfg("v1")
    out = build_frame(start, end, cfg)
    if out.empty or "vix_close" not in out.columns:
        return pd.Series(dtype=float)
    return out["vix_close"].dropna()


def summarise_trades(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_pct": 0.0, "total_$": 0.0, "avg_$": 0.0,
                "sharpe_d": 0.0, "max_dd_$": 0.0, "n_days": 0}
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date

    daily = df.groupby("et_date")["pnl_dollars"].sum()
    cum = daily.sort_index().cumsum()
    drawdown = cum - cum.cummax()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    return {
        "n":         len(df),
        "win_pct":   float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$":   float(df["pnl_dollars"].sum()),
        "avg_$":     float(df["pnl_dollars"].mean()),
        "sharpe_d":  float(sharpe),
        "max_dd_$":  float(drawdown.min() if len(drawdown) else 0.0),
        "n_days":    int(df["et_date"].nunique()),
    }


def run_variant(signals, bars, vix, **kwargs) -> dict:
    cfg = OptionsSimConfig(vix_series=vix, **kwargs)
    sim = make_options_simulator(cfg)
    trades = walk_with_blocking(signals, bars, sim)
    return summarise_trades(trades)


# ---------- the grid ----------

GRID = []

# Phase 1: DTE comparison at default settings
for dte in [0, 1, 3, 7]:
    GRID.append({"dte": dte, "conviction_min": 0.55, "theta_protect_mins": 60 if dte == 0 else 0,
                 "tag": f"dte={dte} conv=0.55 theta=default"})

# Phase 2: conviction sweep at DTE=7 (likely winner from theta math)
for conv in [0.60, 0.65, 0.70]:
    GRID.append({"dte": 7, "conviction_min": conv, "theta_protect_mins": 0,
                 "tag": f"dte=7 conv={conv:.2f}"})

# Phase 3: theta_protect sweep at DTE=0 (rescue attempt for 0DTE)
for tp in [0, 30, 90]:
    GRID.append({"dte": 0, "conviction_min": 0.55, "theta_protect_mins": tp,
                 "tag": f"dte=0 conv=0.55 theta={tp}"})

# Phase 4: combined — high conviction + 7DTE
for dte in [3, 7]:
    for conv in [0.60, 0.65]:
        GRID.append({"dte": dte, "conviction_min": conv, "theta_protect_mins": 0,
                     "tag": f"combined dte={dte} conv={conv:.2f}"})

# Phase 5: bigger stops/targets for longer-dated options
for dte in [3, 7]:
    GRID.append({"dte": dte, "conviction_min": 0.55, "theta_protect_mins": 0,
                 "stop_pct": 0.30, "target_pct": 0.60,
                 "tag": f"narrow dte={dte} stop=30 tgt=60"})


def main():
    print("=" * 100)
    print(" Options strategy grid sweep")
    print(f" Window: {WINDOW[0]} ({WINDOW[1]} → {WINDOW[2]})")
    print("=" * 100)

    bundle = V3Bundle("mtf")
    signals, bars, _ = get_v3_signals_and_bars(
        WINDOW[1], WINDOW[2], bundle, add_vol=False, add_mtf=True)
    vix = _pull_vix(WINDOW[1], WINDOW[2])

    if not signals or bars.empty or vix.empty:
        print("data load failed.")
        return

    print(f" {len(signals)} candidate signals loaded.\n")

    # ---------- run grid ----------
    results = []
    for i, params in enumerate(GRID):
        tag = params.pop("tag")
        kwargs = {k: v for k, v in params.items() if k != "tag"}
        m = run_variant(signals, bars, vix, **kwargs)
        m["tag"] = tag
        m.update(params)   # keep parameters in the result row
        results.append(m)
        print(f"  [{i+1:>2}/{len(GRID)}] {tag:<40} "
              f"n={m['n']:>4}  win={m['win_pct']:>5.1f}%  "
              f"$={m['total_$']:+9,.0f}  Sharpe={m['sharpe_d']:+5.2f}  "
              f"DD=${m['max_dd_$']:+,.0f}")

    # ---------- leaderboard ----------
    print()
    print("=" * 100)
    print(" Leaderboard (sorted by total $)")
    print("=" * 100)
    leaderboard = sorted(results, key=lambda r: r["total_$"], reverse=True)
    print(f"  {'rank':<5}{'tag':<42}{'n':>5}{'win%':>7}{'$':>10}{'$/tr':>7}"
          f"{'Sharpe':>9}{'DD$':>9}")
    for i, r in enumerate(leaderboard, 1):
        print(f"  {i:<5}{r['tag']:<42}{r['n']:>5}{r['win_pct']:>6.1f}%"
              f"{r['total_$']:>+10,.0f}{r['avg_$']:>+7,.0f}"
              f"{r['sharpe_d']:>+9.2f}{r['max_dd_$']:>+9,.0f}")

    # ---------- holdout sanity check on top 3 ----------
    print()
    print("=" * 100)
    print(f" Holdout sanity check — top 3 on {HOLDOUT[0]} ({HOLDOUT[1]} → {HOLDOUT[2]})")
    print("=" * 100)
    h_signals, h_bars, _ = get_v3_signals_and_bars(
        HOLDOUT[1], HOLDOUT[2], bundle, add_vol=False, add_mtf=True)
    h_vix = _pull_vix(HOLDOUT[1], HOLDOUT[2])

    print(f"  {'tag':<42}{'n':>5}{'win%':>7}{'$':>10}{'$/tr':>7}{'Sharpe':>9}")
    for r in leaderboard[:3]:
        kwargs = {k: r[k] for k in ("dte", "conviction_min", "theta_protect_mins")
                  if k in r}
        if "stop_pct" in r: kwargs["stop_pct"] = r["stop_pct"]
        if "target_pct" in r: kwargs["target_pct"] = r["target_pct"]
        m = run_variant(h_signals, h_bars, h_vix, **kwargs)
        print(f"  {r['tag']:<42}{m['n']:>5}{m['win_pct']:>6.1f}%"
              f"{m['total_$']:>+10,.0f}{m['avg_$']:>+7,.0f}{m['sharpe_d']:>+9.2f}")

    # ---------- save raw ----------
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(leaderboard).to_csv(out_dir / "options_sweep.csv", index=False)


if __name__ == "__main__":
    main()
