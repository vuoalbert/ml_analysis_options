"""Phase 3-4: stress-test IV dynamics with worst-case β + multi-window OOS.

Goal: validate that the multi=3 winner survives:
  (a) pessimistic IV beta (calls crush hard, puts modest expansion)
  (b) multiple non-overlapping windows including pre-retrain history

If both hold, we lock in the config.

Run:
    python -m research.sweep_v4_stress_oos
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
from research.backtest_original import build_frame
from research.v3_combinations import V3Bundle, get_v3_signals_and_bars
from research.option_simulator import OptionsSimConfig, make_options_simulator
from research.multi_walker import walk_multi


# Realistic deployable winner — modest leverage, beats stocks
WINNER = {
    "dte": 7,
    "conviction_min": 0.55,
    "theta_protect_mins": 0,
    "risk_pct": 0.02,
    "max_qty": 100,
    "use_iv_dynamics": True,
    "iv_beta_call": 5.0,
    "iv_beta_put": 8.0,
}
WINNER_MAX_CONCURRENT = 3


def _pull_vix(start, end):
    cfg = load_cfg("v1")
    out = build_frame(start, end, cfg)
    return out["vix_close"].dropna() if not out.empty else pd.Series(dtype=float)


def summarise(trades):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("et_date")["pnl_dollars"].sum()
    cum = daily.sort_index().cumsum()
    dd = cum - cum.cummax()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    return {
        "n":         len(df),
        "win_pct":   float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$":   float(df["pnl_dollars"].sum()),
        "avg_$":     float(df["pnl_dollars"].mean()),
        "sharpe_d":  float(sharpe),
        "max_dd_$":  float(dd.min() if len(dd) else 0.0),
    }


def run(signals, bars, vix, *, max_concurrent=3, **kwargs):
    cfg = OptionsSimConfig(vix_series=vix, **kwargs)
    sim = make_options_simulator(cfg)
    trades = walk_multi(signals, bars, sim, max_concurrent=max_concurrent)
    return summarise(trades)


def fmt(m):
    if m is None:
        return "(no trades)"
    return (f"n={m['n']:>4}  win={m['win_pct']:>5.1f}%  $={m['total_$']:>+9,.0f}  "
            f"$/tr={m['avg_$']:>+5,.0f}  Sh={m['sharpe_d']:>+5.2f}  DD=${m['max_dd_$']:>+,.0f}")


# ---------- (a) IV stress test ----------

IV_STRESS = [
    {"label": "Realistic     β_call=5,  β_put=8",   "iv_beta_call": 5.0,  "iv_beta_put": 8.0},
    {"label": "Symmetric     β_call=8,  β_put=8",   "iv_beta_call": 8.0,  "iv_beta_put": 8.0},
    {"label": "Pessimistic   β_call=10, β_put=4",   "iv_beta_call": 10.0, "iv_beta_put": 4.0},
    {"label": "Disaster      β_call=15, β_put=2",   "iv_beta_call": 15.0, "iv_beta_put": 2.0},
    {"label": "No IV dynamics (constant IV)",        "use_iv_dynamics": False},
]

# ---------- (b) Multi-window OOS test ----------
WINDOWS = [
    ("Pre-retrain OOS (Nov-Jan)",      "2025-11-01", "2026-01-26"),
    ("Post-retrain OOS (Jan-Apr)",     "2026-01-27", "2026-04-29"),
    ("Apr 2026 only (full holdout)",   "2026-04-01", "2026-04-29"),
    ("Last 6 months (combined)",       "2025-11-01", "2026-04-29"),
    ("Older slice — Q3 2025",          "2025-07-01", "2025-09-30"),
    ("Older slice — H1 2025",          "2025-01-01", "2025-06-30"),
    ("Older slice — H2 2024",          "2024-07-01", "2024-12-31"),
    ("Older slice — H1 2024",          "2024-01-01", "2024-06-30"),
]


def main():
    print("=" * 110)
    print(" PHASE 3-4: stress test + multi-window OOS")
    print(f" Winner config: 7DTE, p≥{WINNER['conviction_min']}, "
          f"risk={WINNER['risk_pct']:.0%}, multi={WINNER_MAX_CONCURRENT}")
    print("=" * 110)

    bundle = V3Bundle("mtf")

    # ---------- (a) IV stress test on 6mo window ----------
    print()
    print("─" * 110)
    print(" (a) IV stress test — 6mo window (Nov 2025 → Apr 2026)")
    print(f"     Stock baseline: +$41,455")
    print("─" * 110)
    signals, bars, _ = get_v3_signals_and_bars(
        "2025-11-01", "2026-04-29", bundle, add_vol=False, add_mtf=True)
    vix = _pull_vix("2025-11-01", "2026-04-29")

    stock_baseline = 41455
    for stress in IV_STRESS:
        kwargs = dict(WINNER)
        # apply stress overrides
        for k, v in stress.items():
            if k != "label":
                kwargs[k] = v
        m = run(signals, bars, vix, max_concurrent=WINNER_MAX_CONCURRENT, **kwargs)
        beats = "✓" if m and m["total_$"] > stock_baseline else " "
        ratio = (m["total_$"] / stock_baseline) if m else 0
        print(f"  {stress['label']:<42}  {fmt(m)}    {beats} {ratio:>4.1f}× stocks")

    # ---------- (b) Multi-window OOS ----------
    print()
    print("─" * 110)
    print(" (b) Multi-window OOS — fixed config (realistic β), test on each window separately")
    print("─" * 110)

    for label, start, end in WINDOWS:
        try:
            sig, b, _ = get_v3_signals_and_bars(start, end, bundle, add_vol=False, add_mtf=True)
            v = _pull_vix(start, end)
            if not sig or b.empty:
                print(f"  {label:<32}  (no data — likely outside training history)")
                continue
            m = run(sig, b, v, max_concurrent=WINNER_MAX_CONCURRENT, **WINNER)
            print(f"  {label:<32}  {fmt(m)}")
        except Exception as e:
            print(f"  {label:<32}  error: {e}")


if __name__ == "__main__":
    main()
