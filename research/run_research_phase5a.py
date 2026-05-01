"""Phase 5a: test the new ideas — trailing stop, cancel-on-flip, per-symbol cap.

Uses the Phase 4 winner config (ITM 2.5% + target 30%) as the new baseline
and tests each new idea on top of it.

Run:
    python -m research.run_research_phase5a
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

# Phase 4 winner is the new baseline
WINNER_BASE = {
    "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "itm", "itm_offset_pct": 0.025, "target_pct": 0.30,
}

# Each test: (name, sim_config_delta, walker_kwargs)
TESTS = [
    # Phase 4 winner (new baseline)
    {"name": "WINNER (ITM 2.5% target 30%)", "delta": {}, "walker": {}},

    # Trailing stop variants
    {"name": "+ trailing 30% lock 50%",
     "delta": {"trailing_threshold_pct": 0.30, "trail_lock_frac": 0.50}, "walker": {}},
    {"name": "+ trailing 20% lock 50%",
     "delta": {"trailing_threshold_pct": 0.20, "trail_lock_frac": 0.50}, "walker": {}},
    {"name": "+ trailing 15% lock 33%",
     "delta": {"trailing_threshold_pct": 0.15, "trail_lock_frac": 0.33}, "walker": {}},
    {"name": "+ trailing 50% lock 50%",
     "delta": {"trailing_threshold_pct": 0.50, "trail_lock_frac": 0.50}, "walker": {}},

    # Cancel-on-flip
    {"name": "+ cancel-on-flip",
     "delta": {}, "walker": {"cancel_on_flip": True}},

    # Per-symbol cap
    {"name": "+ max_per_symbol=3",
     "delta": {}, "walker": {"max_per_symbol": 3}},
    {"name": "+ max_per_symbol=5",
     "delta": {}, "walker": {"max_per_symbol": 5}},

    # Combinations
    {"name": "+ trailing 20% + cancel-on-flip",
     "delta": {"trailing_threshold_pct": 0.20, "trail_lock_frac": 0.50},
     "walker": {"cancel_on_flip": True}},
    {"name": "+ trailing 20% + max_per_symbol=3",
     "delta": {"trailing_threshold_pct": 0.20, "trail_lock_frac": 0.50},
     "walker": {"max_per_symbol": 3}},
    {"name": "+ all three (trail + flip + cap=3)",
     "delta": {"trailing_threshold_pct": 0.20, "trail_lock_frac": 0.50},
     "walker": {"cancel_on_flip": True, "max_per_symbol": 3}},
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
    losing = df[df["pnl_dollars"] < 0]
    return {
        "n": len(df),
        "win_pct": float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$": float(df["pnl_dollars"].sum()),
        "sharpe": float(sharpe),
        "max_dd_$": float(dd.min() if len(dd) else 0.0),
        "gross_loss_$": float(losing["pnl_dollars"].sum()),
    }


def main():
    print("=" * 110)
    print(" PHASE 5a — trailing stop / cancel-on-flip / per-symbol cap on top of Phase 4 winner")
    print(f" Window: {WINDOW[1]} → {WINDOW[2]}")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    print("\nloading...", flush=True)
    t0 = time.time()
    sig, bars, _ = get_v3_signals_and_bars(WINDOW[1], WINDOW[2], bundle,
                                           add_vol=False, add_mtf=True)
    vix = _vix_from_cache(WINDOW[2])
    print(f"  loaded {len(sig)} signals in {time.time()-t0:.0f}s")

    print()
    print(f"  {'idea':<45}{'n':>5}{'win%':>6}{'total $':>10}"
          f"{'maxDD':>9}{'Sharpe':>8}{'vs winner':>11}")
    print("  " + "-"*94)

    winner_total = 0
    results = []
    for t in TESTS:
        kwargs = dict(WINNER_BASE); kwargs.update(t["delta"])
        cfg = OptionsSimConfig(vix_series=vix, **kwargs)
        sim = make_options_simulator(cfg)
        walker_kw = {"max_concurrent": MAX_CONCURRENT}
        walker_kw.update(t["walker"])
        trades = walk_multi(sig, bars, sim, **walker_kw)
        m = summarise(trades) or {"n": 0, "total_$": 0, "max_dd_$": 0,
                                   "sharpe": 0, "win_pct": 0, "gross_loss_$": 0}
        if "WINNER" in t["name"]:
            winner_total = m["total_$"]
        m["vs_winner"] = m["total_$"] - winner_total
        m["name"] = t["name"]
        print(f"  {t['name']:<45}{m['n']:>5}{m['win_pct']:>5.1f}%"
              f"{m['total_$']:>+10,.0f}{m['max_dd_$']:>+9,.0f}"
              f"{m['sharpe']:>+8.2f}{m['vs_winner']:>+11,.0f}")
        results.append(m)

    # Rank
    print()
    print("=" * 110)
    print(" Top variants beating the WINNER:")
    print("=" * 110)
    ranked = sorted([r for r in results if "WINNER" not in r["name"]],
                    key=lambda r: r["vs_winner"], reverse=True)
    for r in ranked:
        marker = "★" if r["vs_winner"] > 0 else "✗"
        print(f"  {marker}  {r['name']:<45}  Δ = {r['vs_winner']:>+10,.0f}  "
              f"DD={r['max_dd_$']:>+,.0f}  Sh={r['sharpe']:>+.2f}")

    out = Path(__file__).parent / "outputs" / "research_phase5a.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
