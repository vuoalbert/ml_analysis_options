"""Research Phase 1: test each individually-tunable idea vs baseline.

Tests on the 6-month recent window (Nov 2025 → Apr 2026) which is
representative and fast to load.

Run:
    python -m research.run_research_phase1
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


WINDOW = ("Last 6 months", "2025-11-01", "2026-04-29")
MAX_CONCURRENT = 15

# Baseline (current live config)
BASELINE = {
    "dte": 7,
    "conviction_min": 0.55,
    "theta_protect_mins": 0,
    "risk_pct": 0.02,
    "max_qty": 100,
    "use_iv_dynamics": True,
    "iv_beta_call": 5.0,
    "iv_beta_put": 8.0,
    "moneyness": "atm",
}

# Each "idea" is a delta to apply on top of baseline
IDEAS = [
    {"name": "BASELINE (multi=15 live)", "delta": {}},
    {"name": "4. Conviction-tiered (p≥0.55, scale to 4%)",
     "delta": {"conviction_lo": 0.55, "conviction_hi": 0.75,
               "risk_pct": 0.005, "risk_pct_max": 0.04}},
    {"name": "5. VIX > 14 filter",      "delta": {"vix_min": 14.0}},
    {"name": "5b. VIX > 18 filter",     "delta": {"vix_min": 18.0}},
    {"name": "6. Skip midday 11-14 ET", "delta": {"skip_hours_et": (15, 16, 17, 18)}},  # UTC
    {"name": "7. ITM 0.5% offset",      "delta": {"moneyness": "itm", "itm_offset_pct": 0.005}},
    {"name": "7b. ITM 1% offset",       "delta": {"moneyness": "itm", "itm_offset_pct": 0.01}},
    {"name": "Theta protection 30 min", "delta": {"theta_protect_mins": 30}},
    {"name": "Theta protection 60 min", "delta": {"theta_protect_mins": 60}},
    {"name": "Tight stop 30%",          "delta": {"stop_pct": 0.30}},
    {"name": "Wider stop 70%",          "delta": {"stop_pct": 0.70}},
    {"name": "Tight target 50%",        "delta": {"target_pct": 0.50}},
    {"name": "Wider target 200%",       "delta": {"target_pct": 2.00}},
]


def summarise(trades, baseline_total=0):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("et_date")["pnl_dollars"].sum()
    cum = daily.sort_index().cumsum()
    dd = cum - cum.cummax()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if (len(daily) > 1 and daily.std() > 0) else 0.0
    losing_trades = df[df["pnl_dollars"] < 0]
    return {
        "n":             len(df),
        "win_pct":       float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$":       float(df["pnl_dollars"].sum()),
        "sharpe":        float(sharpe),
        "max_dd_$":      float(dd.min() if len(dd) else 0.0),
        "gross_loss_$":  float(losing_trades["pnl_dollars"].sum()),
        "vs_baseline":   float(df["pnl_dollars"].sum() - baseline_total),
    }


def main():
    print("=" * 110)
    print(" RESEARCH PHASE 1 — each idea individually vs baseline (6mo window)")
    print(f" Window: {WINDOW[1]} → {WINDOW[2]}, multi={MAX_CONCURRENT}")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    print("\nloading window...", flush=True)
    t0 = time.time()
    sig, bars, _ = get_v3_signals_and_bars(WINDOW[1], WINDOW[2], bundle,
                                           add_vol=False, add_mtf=True)
    # Pull VIX from cache
    from data_pull import cache as _cache
    vix_files = sorted(_cache.cache_dir().glob("yf_VIX_*.parquet"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    vix = pd.Series(dtype=float)
    for vp in vix_files:
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty and df.index.max() >= pd.Timestamp(WINDOW[2], tz="UTC") - pd.Timedelta(days=10):
            vix = (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")
            break
    print(f"  loaded {len(sig)} signals, {bars.shape[0]} bars in {time.time()-t0:.0f}s")

    # Run baseline first
    print(f"\n{'idea':<42}{'n':>6}{'win%':>7}{'total $':>11}{'gross loss':>13}"
          f"{'maxDD':>9}{'Sharpe':>8}{'vs base':>11}")
    print("  " + "-"*100)

    baseline_total = 0
    results = []
    for i, idea in enumerate(IDEAS):
        kwargs = dict(BASELINE)
        kwargs.update(idea["delta"])
        cfg = OptionsSimConfig(vix_series=vix, **kwargs)
        sim = make_options_simulator(cfg)
        trades = walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)
        m = summarise(trades, baseline_total)
        if i == 0:
            baseline_total = m["total_$"]
            m["vs_baseline"] = 0
        if m:
            print(f"  {idea['name']:<40}{m['n']:>6}{m['win_pct']:>6.1f}%"
                  f"{m['total_$']:>+11,.0f}{m['gross_loss_$']:>+13,.0f}"
                  f"{m['max_dd_$']:>+9,.0f}{m['sharpe']:>+8.2f}"
                  f"{m['vs_baseline']:>+11,.0f}")
            results.append({**m, "name": idea["name"]})

    # Save and rank
    print()
    print("=" * 110)
    print(" Ranked by improvement over baseline:")
    print("=" * 110)
    ranked = sorted(results[1:], key=lambda r: r["vs_baseline"], reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"  {i:>2}. {r['name']:<42}  Δ vs base = {r['vs_baseline']:>+11,.0f}")

    out = Path(__file__).parent / "outputs" / "research_phase1.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
