"""Quick multi=3/10/20 comparison on the recent 6mo window.

Single window, three concurrency levels, IV dynamics on, realistic costs.
This is the minimal end-to-end validation that didn't complete before
due to perf bugs we've now patched.

Run:
    python -m research.run_quick_compare
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
from research.backtest_original import build_frame
from research.v3_combinations import V3Bundle, get_v3_signals_and_bars
from research.option_simulator import OptionsSimConfig, make_options_simulator
from research.multi_walker import walk_multi


WINDOW = ("March 2026", "2026-03-01", "2026-03-31")
STOCK_BENCHMARK = None     # not specifically known for March; we'll just compare across multi-N


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


def main():
    print("=" * 100)
    print(f" QUICK MULTI COMPARE — {WINDOW[0]} ({WINDOW[1]} → {WINDOW[2]})")
    if STOCK_BENCHMARK is not None:
        print(f" Target: beat +${STOCK_BENCHMARK:,.0f}")
    print("=" * 100)

    bundle = V3Bundle("mtf")

    t0 = time.time()
    print(f"\n[t={time.time()-t0:.1f}s] loading window...")
    sig, bars, _ = get_v3_signals_and_bars(
        WINDOW[1], WINDOW[2], bundle, add_vol=False, add_mtf=True)
    print(f"[t={time.time()-t0:.1f}s] got {len(sig)} signals, bars {bars.shape}")

    print(f"[t={time.time()-t0:.1f}s] loading VIX directly from cache (skip build_frame)...")
    # Pull VIX directly from cache — find the most recent yf_VIX parquet
    # covering our window. Bypasses the build_frame-induced slowdown.
    from data_pull import cache as _cache
    cache_root = _cache.cache_dir()
    vix_files = sorted(cache_root.glob("yf_VIX_*.parquet"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    vix = pd.Series(dtype=float)
    for vp in vix_files:
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        # Check if this file covers our window
        end_ts = pd.Timestamp(WINDOW[2], tz="UTC")
        if not df.empty and df.index.max() >= end_ts - pd.Timedelta(days=10):
            if "close" in df.columns:
                vix = df["close"].rename("vix_close")
            else:
                vix = df.iloc[:, 0]
            break
    print(f"[t={time.time()-t0:.1f}s] vix len={len(vix)}")

    BASE = {
        "vix_series": vix,
        "dte": 7,
        "conviction_min": 0.55,
        "theta_protect_mins": 0,
        "risk_pct": 0.02,
        "max_qty": 100,
        "use_iv_dynamics": True,
        "iv_beta_call": 5.0,
        "iv_beta_put": 8.0,
    }

    print()
    print(f"  {'multi':<8}{'n':>6}{'win%':>7}{'total $':>12}{'$/tr':>7}{'Sharpe':>9}{'DD$':>10}{'beats?':>8}")
    print("  " + "-" * 72)
    for k in [3, 5, 10, 15, 20]:
        cfg = OptionsSimConfig(**BASE)
        sim = make_options_simulator(cfg)
        t1 = time.time()
        trades = walk_multi(sig, bars, sim, max_concurrent=k)
        m = summarise(trades)
        beat = "✓" if (STOCK_BENCHMARK is not None and m and m["total_$"] > STOCK_BENCHMARK) else " "
        if m:
            print(f"  multi={k:<3}{m['n']:>6}{m['win_pct']:>6.1f}%"
                  f"{m['total_$']:>+12,.0f}{m['avg_$']:>+7,.0f}"
                  f"{m['sharpe_d']:>+9.2f}{m['max_dd_$']:>+10,.0f}    {beat}"
                  f"   [walk {time.time()-t1:.1f}s]")
        else:
            print(f"  multi={k:<3}  no trades")

    # also realistic friction
    print()
    print(" REALISTIC FRICTION (100 bps each way):")
    print(f"  {'multi':<8}{'n':>6}{'win%':>7}{'total $':>12}{'$/tr':>7}{'Sharpe':>9}{'DD$':>10}{'beats?':>8}")
    print("  " + "-" * 72)
    for k in [3, 5, 10, 15, 20]:
        kwargs = dict(BASE)
        kwargs["entry_cost_bps"] = 100.0
        kwargs["exit_cost_bps"] = 100.0
        cfg = OptionsSimConfig(**kwargs)
        sim = make_options_simulator(cfg)
        trades = walk_multi(sig, bars, sim, max_concurrent=k)
        m = summarise(trades)
        beat = "✓" if (STOCK_BENCHMARK is not None and m and m["total_$"] > STOCK_BENCHMARK) else " "
        if m:
            print(f"  multi={k:<3}{m['n']:>6}{m['win_pct']:>6.1f}%"
                  f"{m['total_$']:>+12,.0f}{m['avg_$']:>+7,.0f}"
                  f"{m['sharpe_d']:>+9.2f}{m['max_dd_$']:>+10,.0f}    {beat}")


if __name__ == "__main__":
    main()
