"""Phase 5b: vertical debit + credit spreads vs Phase 4 winner.

Run:
    python -m research.run_research_phase5b
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
from research.option_simulator_spreads import SpreadConfig, make_spread_simulator
from research.multi_walker import walk_multi


WINDOW = ("April 2026", "2026-04-01", "2026-04-30")
MAX_CONCURRENT = 15

WINNER_BASE = {
    "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "itm", "itm_offset_pct": 0.025, "target_pct": 0.30,
}

# Spread variants
SPREAD_VARIANTS = [
    {"name": "DEBIT  spread 1% width",  "spread_kind": "debit",  "leg_offset_pct": 0.01},
    {"name": "DEBIT  spread 2% width",  "spread_kind": "debit",  "leg_offset_pct": 0.02},
    {"name": "DEBIT  spread 3% width",  "spread_kind": "debit",  "leg_offset_pct": 0.03},
    {"name": "DEBIT  spread 5% width",  "spread_kind": "debit",  "leg_offset_pct": 0.05},
    {"name": "CREDIT spread 1% width",  "spread_kind": "credit", "leg_offset_pct": 0.01},
    {"name": "CREDIT spread 2% width",  "spread_kind": "credit", "leg_offset_pct": 0.02},
    {"name": "CREDIT spread 3% width",  "spread_kind": "credit", "leg_offset_pct": 0.03},
    {"name": "CREDIT spread 5% width",  "spread_kind": "credit", "leg_offset_pct": 0.05},
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


def main():
    print("=" * 110)
    print(" PHASE 5b — vertical debit + credit spreads vs Phase 4 winner (naked ITM 2.5% calls/puts)")
    print(f" Window: {WINDOW[1]} → {WINDOW[2]}")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    print("\nloading...", flush=True)
    t0 = time.time()
    sig, bars, _ = get_v3_signals_and_bars(WINDOW[1], WINDOW[2], bundle,
                                           add_vol=False, add_mtf=True)
    vix = _vix_from_cache(WINDOW[2])
    print(f"  loaded {len(sig)} signals in {time.time()-t0:.0f}s")

    # Run winner first
    winner_cfg = OptionsSimConfig(vix_series=vix, **WINNER_BASE)
    winner_sim = make_options_simulator(winner_cfg)
    winner_trades = walk_multi(sig, bars, winner_sim, max_concurrent=MAX_CONCURRENT)
    w = summarise(winner_trades)
    print(f"\nWINNER (ITM 2.5% naked + target 30%):  n={w['n']}  win={w['win_pct']:.1f}%  "
          f"$={w['total_$']:+,.0f}  DD={w['max_dd_$']:+,.0f}  Sh={w['sharpe']:+.2f}")

    print()
    print(f"  {'spread variant':<35}{'n':>5}{'win%':>7}{'total $':>11}"
          f"{'maxDD':>9}{'Sh':>7}{'vs winner':>11}")
    print("  " + "-"*84)

    results = [{"name": "WINNER (naked)", **w, "vs_winner": 0}]
    for v in SPREAD_VARIANTS:
        cfg = SpreadConfig(
            vix_series=vix,
            spread_kind=v["spread_kind"],
            leg_offset_pct=v["leg_offset_pct"],
            dte=WINNER_BASE["dte"],
            conviction_min=WINNER_BASE["conviction_min"],
            risk_pct=WINNER_BASE["risk_pct"],
            max_qty=WINNER_BASE["max_qty"],
            use_iv_dynamics=WINNER_BASE["use_iv_dynamics"],
            iv_beta_call=WINNER_BASE["iv_beta_call"],
            iv_beta_put=WINNER_BASE["iv_beta_put"],
            target_pct=WINNER_BASE["target_pct"],
        )
        sim = make_spread_simulator(cfg)
        trades = walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)
        m = summarise(trades) or {"n": 0, "total_$": 0, "max_dd_$": 0,
                                   "sharpe": 0, "win_pct": 0}
        m["vs_winner"] = m["total_$"] - w["total_$"]
        m["name"] = v["name"]
        marker = "★" if m["vs_winner"] > 0 else " "
        print(f"  {marker} {v['name']:<33}{m['n']:>5}{m['win_pct']:>6.1f}%"
              f"{m['total_$']:>+11,.0f}{m['max_dd_$']:>+9,.0f}"
              f"{m['sharpe']:>+7.2f}{m['vs_winner']:>+11,.0f}")
        results.append(m)

    print()
    print("=" * 110)
    print(" Variants beating the winner:")
    print("=" * 110)
    winners = [r for r in results if r["name"] != "WINNER (naked)" and r["vs_winner"] > 0]
    if winners:
        for r in sorted(winners, key=lambda r: r["vs_winner"], reverse=True):
            print(f"  ★ {r['name']:<35}  Δ = {r['vs_winner']:>+10,.0f}  "
                  f"DD={r['max_dd_$']:>+,.0f}  Sh={r['sharpe']:>+.2f}")
    else:
        print("  None — naked ITM 2.5% beats all spread variants on April.")
        print("  Spreads cap upside; the strategy's edge needs uncapped winners.")

    out = Path(__file__).parent / "outputs" / "research_phase5b.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
