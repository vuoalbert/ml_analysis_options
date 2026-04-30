"""Multi-N comparison across 2020-2023 pre-training era.

The v3_mtf model was trained on data 2023-04-01+. Everything before that
is data the model has never seen — strongest test of generalization.

We use 6-month windows (more reliable loading than single-month for the
older data range). 5 windows × 5 multi-N levels = 25 walks.

Run:
    python -m research.run_historical_compare
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


WINDOWS = [
    ("2020 H2 + Q1 21",  "2020-09-01", "2021-03-31"),
    ("Q2-Q3 2021",        "2021-04-01", "2021-09-30"),
    ("Q4 21 + Q1 22",     "2021-10-01", "2022-03-31"),
    ("Q2-Q3 2022 (bear)", "2022-04-01", "2022-09-30"),
    ("Q4 22 + Q1 23",     "2022-10-01", "2023-03-31"),
]
LEVELS = [3, 5, 10, 15, 20]

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


def _vix_from_cache(end_str):
    from data_pull import cache as _cache
    cache_root = _cache.cache_dir()
    end_ts = pd.Timestamp(end_str, tz="UTC")
    # Pick a parquet that includes our end_ts in its range
    candidates = []
    for vp in cache_root.glob("yf_VIX_*.parquet"):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty and df.index.min() <= end_ts <= df.index.max() + pd.Timedelta(days=10):
            candidates.append((vp.stat().st_mtime, df))
    if not candidates:
        # Fallback: most recent available
        for vp in sorted(cache_root.glob("yf_VIX_*.parquet"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            df = pd.read_parquet(vp)
            if df.index.dtype != "datetime64[ns, UTC]":
                df.index = df.index.astype("datetime64[ns, UTC]")
            if not df.empty:
                return (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")
        return pd.Series(dtype=float)
    candidates.sort(key=lambda x: x[0], reverse=True)
    df = candidates[0][1]
    return (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")


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


def main():
    print("=" * 110)
    print(" HISTORICAL COMPARE — multi-N strategy across 2020-2023 pre-training era")
    print(" Model trained on 2023-04-01+. Everything below is FULLY out-of-sample.")
    print("=" * 110)

    bundle = V3Bundle("mtf")

    all_results = {}
    for label, start, end in WINDOWS:
        print(f"\n{label} ({start} → {end})")
        t0 = time.time()
        try:
            sig, bars, _ = get_v3_signals_and_bars(start, end, bundle, add_vol=False, add_mtf=True)
            vix = _vix_from_cache(end)
            print(f"  loaded {len(sig)} signals, vix len={len(vix)} in {time.time()-t0:.1f}s")
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
    print(" SUMMARY — total $ per 6mo window per multi-N (pre-training)")
    print("=" * 110)
    print(f"  {'window':<22}" + "".join(f"{'multi=' + str(k):>14}" for k in LEVELS))
    print("  " + "-" * (22 + 14 * len(LEVELS)))
    for label, _, _ in WINDOWS:
        if label not in all_results:
            print(f"  {label:<22}  (no data)")
            continue
        row = f"  {label:<22}"
        for k in LEVELS:
            r = all_results[label].get(k)
            row += f"{r['total_$']:>+14,.0f}" if r else f"{'—':>14}"
        print(row)
    # Per-multi totals
    print(f"  {'TOTAL (30 months)':<22}", end="")
    for k in LEVELS:
        total = sum(all_results.get(lbl, {}).get(k, {}).get("total_$", 0)
                    for lbl, _, _ in WINDOWS)
        print(f"{total:>+14,.0f}", end="")
    print()


if __name__ == "__main__":
    main()
