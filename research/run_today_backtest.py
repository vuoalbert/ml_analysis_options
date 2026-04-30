"""Backtest today's session — load enough lookback for features, filter to today.

Run:
    python -m research.run_today_backtest
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

from research.v3_combinations import V3Bundle, get_v3_signals_and_bars
from research.option_simulator import OptionsSimConfig, make_options_simulator
from research.multi_walker import walk_multi


# Use 1 month lookback so features are populated; we'll filter signals to today.
LOOKBACK_START = "2026-04-01"
WINDOW_END = "2026-05-01"           # next day to make Alpaca range non-zero
TODAY = pd.Timestamp("2026-04-30", tz="UTC")
LEVELS = [3, 5, 10, 15, 20]


def _vix_from_cache():
    from data_pull import cache as _cache
    cache_root = _cache.cache_dir()
    for vp in sorted(cache_root.glob("yf_VIX_*.parquet"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty:
            return (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")
    return pd.Series(dtype=float)


def summarise(trades):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    daily = df.groupby(df["entry_ts"].dt.date)["pnl_dollars"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if (len(daily) > 1 and daily.std() > 0) else 0.0
    return {
        "n":         len(df),
        "win_pct":   float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$":   float(df["pnl_dollars"].sum()),
        "avg_$":     float(df["pnl_dollars"].mean()),
        "sharpe_d":  float(sharpe),
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
    print("=" * 100)
    print(f" TODAY'S BACKTEST — Apr 30 2026")
    print(f" Lookback for features: {LOOKBACK_START} → {WINDOW_END}")
    print(f" Filter to entries on: {TODAY.date()}")
    print("=" * 100)

    bundle = V3Bundle("mtf")

    t0 = time.time()
    print(f"\n[t={time.time()-t0:.1f}s] loading...")
    sig, bars, _ = get_v3_signals_and_bars(LOOKBACK_START, WINDOW_END, bundle,
                                           add_vol=False, add_mtf=True)
    vix = _vix_from_cache()
    print(f"[t={time.time()-t0:.1f}s] {len(sig)} total signals, {bars.shape[0]} bars, vix len={len(vix)}")

    # Filter signals to those that fired TODAY
    today_sigs = [s for s in sig if pd.Timestamp(s["ts"]).date() == TODAY.date()]
    print(f"[t={time.time()-t0:.1f}s] {len(today_sigs)} signals fired today")

    if not today_sigs:
        # Most likely: today is mid-session and signals haven't been generated yet
        # OR not enough trading data has elapsed today
        print("\n  No signals today (yet). Possible reasons:")
        print("    - Bar data hasn't accumulated enough today (RTH started 13:30 UTC)")
        print("    - Model predictions all below threshold (no high-conviction signals)")
        return

    print()
    print(f"  {'multi':<8}{'n':>5}{'win%':>7}{'total $':>11}{'$/tr':>7}{'Sharpe':>9}")
    print("  " + "-" * 50)
    for k in LEVELS:
        cfg = OptionsSimConfig(vix_series=vix, **BASE)
        sim = make_options_simulator(cfg)
        trades = walk_multi(today_sigs, bars, sim, max_concurrent=k)
        m = summarise(trades)
        if m:
            print(f"  multi={k:<3}{m['n']:>5}{m['win_pct']:>6.1f}%"
                  f"{m['total_$']:>+11,.0f}{m['avg_$']:>+7,.0f}"
                  f"{m['sharpe_d']:>+9.2f}")
        else:
            print(f"  multi={k:<3}  (no trades — all signals blocked)")

    # Show first few signals so user knows what we're trading
    print()
    print(" Today's signals (first 10):")
    for s in today_sigs[:10]:
        ts_et = pd.Timestamp(s["ts"]).tz_convert("America/New_York")
        side = "LONG" if s["side"] == "long" else "SHORT"
        p = s["p_up"] if s["side"] == "long" else s["p_dn"]
        print(f"   {ts_et.strftime('%H:%M ET')}  {side}  p={p:.3f}")


if __name__ == "__main__":
    main()
