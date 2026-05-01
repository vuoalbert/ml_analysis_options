"""Phase 5c: multi-DTE ensemble — split signals across different expiration buckets.

Idea: instead of always 7DTE, split each signal into 3 trades at different DTEs.
Diversifies time exposure. Some trades benefit from quick expiry, some from longer.

We test by allocating signals proportionally across 1DTE, 3DTE, 7DTE, 14DTE.

Run:
    python -m research.run_research_phase5c
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


WINDOW = ("April 2026", "2026-04-01", "2026-04-30")
MAX_CONCURRENT = 15

WINNER_BASE = {
    "conviction_min": 0.55, "theta_protect_mins": 0,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "itm", "itm_offset_pct": 0.025, "target_pct": 0.30,
}

# DTE variants — single DTE
SINGLE_DTE = [1, 3, 7, 14, 21, 30]

# Ensemble: alternate across signals
ENSEMBLES = [
    {"name": "Multi-DTE [1,3,7]",     "dtes": [1, 3, 7]},
    {"name": "Multi-DTE [3,7,14]",    "dtes": [3, 7, 14]},
    {"name": "Multi-DTE [1,7,14]",    "dtes": [1, 7, 14]},
    {"name": "Multi-DTE [7,14,30]",   "dtes": [7, 14, 30]},
    {"name": "Multi-DTE [3,7,14,30]", "dtes": [3, 7, 14, 30]},
]


def _vix_from_cache(end_str):
    from data_pull import cache as _cache
    end_ts = pd.Timestamp(end_str, tz="UTC")
    for vp in sorted(_cache.cache_dir().glob("yf_VIX_*.parquet"),
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
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if (len(daily) > 1 and daily.std() > 0) else 0.0
    return {
        "n": len(df),
        "win_pct": float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$": float(df["pnl_dollars"].sum()),
        "sharpe": float(sharpe),
        "max_dd_$": float(dd.min() if len(dd) else 0.0),
    }


def run_dte_walk(sig, bars, vix, dte: int):
    cfg = OptionsSimConfig(vix_series=vix, dte=dte, **WINNER_BASE)
    sim = make_options_simulator(cfg)
    return walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)


def run_ensemble_walk(sig, bars, vix, dtes: list[int]):
    """Round-robin signals across DTE variants. Each ensemble walk uses
    proportional max_concurrent so total deployed = single-DTE level."""
    sub_cap = max(1, MAX_CONCURRENT // len(dtes))
    all_trades = []
    for i, dte in enumerate(dtes):
        # Take every Nth signal for this DTE
        subset = [s for j, s in enumerate(sig) if j % len(dtes) == i]
        cfg = OptionsSimConfig(vix_series=vix, dte=dte, **WINNER_BASE)
        sim = make_options_simulator(cfg)
        trades = walk_multi(subset, bars, sim, max_concurrent=sub_cap)
        for t in trades:
            t["dte"] = dte
        all_trades.extend(trades)
    return all_trades


def main():
    print("=" * 110)
    print(" PHASE 5c — DTE sweep + multi-DTE ensemble vs Phase 4 winner (DTE=7)")
    print(f" Window: {WINDOW[1]} → {WINDOW[2]}")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    print("\nloading...", flush=True)
    t0 = time.time()
    sig, bars, _ = get_v3_signals_and_bars(WINDOW[1], WINDOW[2], bundle,
                                           add_vol=False, add_mtf=True)
    vix = _vix_from_cache(WINDOW[2])
    print(f"  loaded {len(sig)} signals in {time.time()-t0:.0f}s")

    # Single-DTE sweep
    print()
    print(f"  {'variant':<35}{'n':>5}{'win%':>7}{'total $':>11}{'maxDD':>9}{'Sh':>7}{'vs DTE7':>11}")
    print("  " + "-"*84)

    base_total = 0
    results = []
    for dte in SINGLE_DTE:
        trades = run_dte_walk(sig, bars, vix, dte)
        m = summarise(trades) or {"n": 0, "total_$": 0, "max_dd_$": 0, "sharpe": 0, "win_pct": 0}
        m["name"] = f"Single DTE={dte}"
        if dte == 7:
            base_total = m["total_$"]
        m["vs_winner"] = m["total_$"] - base_total
        marker = "★" if dte != 7 and m["total_$"] > base_total else " "
        if dte == 7:
            marker = "→"
        print(f"  {marker} {m['name']:<33}{m['n']:>5}{m['win_pct']:>6.1f}%"
              f"{m['total_$']:>+11,.0f}{m['max_dd_$']:>+9,.0f}"
              f"{m['sharpe']:>+7.2f}{m['vs_winner']:>+11,.0f}")
        results.append(m)

    # Ensembles
    print()
    for v in ENSEMBLES:
        trades = run_ensemble_walk(sig, bars, vix, v["dtes"])
        m = summarise(trades) or {"n": 0, "total_$": 0, "max_dd_$": 0, "sharpe": 0, "win_pct": 0}
        m["name"] = v["name"]
        m["vs_winner"] = m["total_$"] - base_total
        marker = "★" if m["total_$"] > base_total else " "
        print(f"  {marker} {m['name']:<33}{m['n']:>5}{m['win_pct']:>6.1f}%"
              f"{m['total_$']:>+11,.0f}{m['max_dd_$']:>+9,.0f}"
              f"{m['sharpe']:>+7.2f}{m['vs_winner']:>+11,.0f}")
        results.append(m)

    # Conclusion
    winners = [r for r in results if r["vs_winner"] > 0 and r["name"] != "Single DTE=7"]
    print()
    print("=" * 110)
    if winners:
        print(" Variants beating DTE=7 winner:")
        for r in sorted(winners, key=lambda r: r["vs_winner"], reverse=True):
            print(f"  ★ {r['name']:<35}  Δ = {r['vs_winner']:>+10,.0f}  "
                  f"DD={r['max_dd_$']:>+,.0f}  Sh={r['sharpe']:>+.2f}")
    else:
        print(" None — DTE=7 remains optimal on April.")

    out = Path(__file__).parent / "outputs" / "research_phase5c.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
