"""Multi-month, multi-N comparison — runs Quick Compare on each month.

Validates the strategy across each post-retrain OOS month, plus pre-retrain
months for context. Same simulator, just iterates over month windows.

Run:
    python -m research.run_monthly_compare
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from utils.config import load as load_cfg
from research.v3_combinations import V3Bundle, get_v3_signals_and_bars
from research.option_simulator import OptionsSimConfig, make_options_simulator
from research.multi_walker import walk_multi


# Each month — broad enough to span the full post-retrain OOS period
MONTHS = [
    ("Dec 2025",     "2025-12-01", "2025-12-31"),
    ("Jan 2026",     "2026-01-01", "2026-01-31"),
    ("Feb 2026",     "2026-02-01", "2026-02-28"),
    ("Mar 2026",     "2026-03-01", "2026-03-31"),
    ("Apr 2026",     "2026-04-01", "2026-04-30"),
]
LEVELS = [3, 5, 10, 15, 20]


def _vix_from_cache(end_str):
    """Bypass build_frame for VIX — read directly from parquet cache."""
    from data_pull import cache as _cache
    cache_root = _cache.cache_dir()
    end_ts = pd.Timestamp(end_str, tz="UTC")
    for vp in sorted(cache_root.glob("yf_VIX_*.parquet"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty and df.index.max() >= end_ts - pd.Timedelta(days=10):
            return (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")
    return pd.Series(dtype=float)


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
        "sharpe_d":  float(sharpe),
        "max_dd_$":  float(dd.min() if len(dd) else 0.0),
    }


BASE = {
    "dte": 7,
    "conviction_min": 0.55,
    "theta_protect_mins": 0,
    "risk_pct": 0.02,
    "max_qty": 100,
    "use_iv_dynamics": True,
    "iv_beta_call": 5.0,
    "iv_beta_put": 8.0,
}


def main():
    print("=" * 110)
    print(" MONTHLY COMPARE — multi-N strategy across 5 months (Dec 2025 → Apr 2026)")
    print(" All months post-Jan-27 retrain are pure OOS for the v3_mtf model")
    print("=" * 110)

    bundle = V3Bundle("mtf")

    all_results = {}
    for label, start, end in MONTHS:
        print(f"\n{label} ({start} → {end})")
        t0 = time.time()
        try:
            sig, bars, _ = get_v3_signals_and_bars(start, end, bundle, add_vol=False, add_mtf=True)
            vix = _vix_from_cache(end)
            print(f"  loaded {len(sig)} signals in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        if not sig:
            print("  no signals")
            continue

        all_results[label] = {}
        print(f"  {'multi':<8}{'n':>5}{'win%':>7}{'total $':>11}{'Sh':>7}{'DD$':>9}")
        print(f"  {'-'*54}")
        for k in LEVELS:
            cfg = OptionsSimConfig(vix_series=vix, **BASE)
            sim = make_options_simulator(cfg)
            trades = walk_multi(sig, bars, sim, max_concurrent=k)
            m = summarise(trades)
            if m:
                print(f"  multi={k:<3}{m['n']:>5}{m['win_pct']:>6.1f}%"
                      f"{m['total_$']:>+11,.0f}{m['sharpe_d']:>+7.2f}"
                      f"{m['max_dd_$']:>+9,.0f}")
                all_results[label][k] = m

    # Summary table
    print()
    print("=" * 110)
    print(" SUMMARY TABLE — total $ per month per multi-N")
    print("=" * 110)
    header = f"  {'month':<14}" + "".join(f"{'multi=' + str(k):>14}" for k in LEVELS)
    print(header)
    print("  " + "-" * (14 + 14 * len(LEVELS)))
    for label, _, _ in MONTHS:
        if label not in all_results:
            print(f"  {label:<14}  (no data)")
            continue
        row = f"  {label:<14}"
        for k in LEVELS:
            r = all_results[label].get(k)
            if r:
                row += f"{r['total_$']:>+14,.0f}"
            else:
                row += f"{'—':>14}"
        print(row)

    # Per-multi totals
    print()
    print(f"  {'TOTAL':<14}", end="")
    for k in LEVELS:
        total = sum(all_results.get(lbl, {}).get(k, {}).get("total_$", 0)
                    for lbl, _, _ in MONTHS)
        print(f"{total:>+14,.0f}", end="")
    print()


if __name__ == "__main__":
    main()
