"""Phase 2: combine the top winners from Phase 1 + sweep within them.

Top winners: ITM offset, tight target, VIX filter, tight stop.
Test all pairwise + a few triples + sweep ITM offset for sweet spot.

Run:
    python -m research.run_research_phase2
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


WINDOW = ("April 2026", "2026-04-01", "2026-04-30")    # smaller window to avoid the cold-cache slow path
MAX_CONCURRENT = 15

BASELINE = {
    "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "atm",
}

# Phase 2 combinations — pairs of top winners + ITM sweep
COMBOS = [
    # ITM offset sweep (find sweet spot)
    {"name": "ITM 0.25%",            "delta": {"moneyness": "itm", "itm_offset_pct": 0.0025}},
    {"name": "ITM 0.5%",             "delta": {"moneyness": "itm", "itm_offset_pct": 0.005}},
    {"name": "ITM 1.0%",             "delta": {"moneyness": "itm", "itm_offset_pct": 0.01}},
    {"name": "ITM 1.5%",             "delta": {"moneyness": "itm", "itm_offset_pct": 0.015}},
    {"name": "ITM 2.0%",             "delta": {"moneyness": "itm", "itm_offset_pct": 0.02}},

    # Target sweep
    {"name": "target 30%",           "delta": {"target_pct": 0.30}},
    {"name": "target 50%",           "delta": {"target_pct": 0.50}},
    {"name": "target 75%",           "delta": {"target_pct": 0.75}},

    # PAIRS — ITM 1% + target sweep
    {"name": "ITM 1% + target 30%",  "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "target_pct": 0.30}},
    {"name": "ITM 1% + target 50%",  "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "target_pct": 0.50}},
    {"name": "ITM 1% + target 75%",  "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "target_pct": 0.75}},

    # PAIRS — ITM 1.5% + target sweep
    {"name": "ITM 1.5% + target 30%","delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.30}},
    {"name": "ITM 1.5% + target 50%","delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.50}},
    {"name": "ITM 1.5% + target 75%","delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.75}},

    # PAIRS — top winners with VIX filter
    {"name": "ITM 1% + VIX>14",      "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "vix_min": 14.0}},
    {"name": "ITM 1% + tight stop 30%","delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "stop_pct": 0.30}},

    # TRIPLES — combine top 3
    {"name": "ITM 1% + target 50% + VIX>14",
        "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "target_pct": 0.50, "vix_min": 14.0}},
    {"name": "ITM 1% + target 50% + stop 30%",
        "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "target_pct": 0.50, "stop_pct": 0.30}},
    {"name": "ITM 1.5% + target 50% + VIX>14",
        "delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.50, "vix_min": 14.0}},
    {"name": "ITM 1.5% + target 50% + stop 30%",
        "delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.50, "stop_pct": 0.30}},

    # QUADRUPLE — top 4 ideas
    {"name": "ITM 1% + target 50% + stop 30% + VIX>14",
        "delta": {"moneyness": "itm", "itm_offset_pct": 0.01, "target_pct": 0.50,
                  "stop_pct": 0.30, "vix_min": 14.0}},
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
    losing = df[df["pnl_dollars"] < 0]
    return {
        "n": len(df),
        "win_pct": float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$": float(df["pnl_dollars"].sum()),
        "sharpe": float(sharpe),
        "max_dd_$": float(dd.min() if len(dd) else 0.0),
        "gross_loss_$": float(losing["pnl_dollars"].sum()),
        "vs_baseline": float(df["pnl_dollars"].sum() - baseline_total),
    }


def main():
    print("=" * 110)
    print(" RESEARCH PHASE 2 — combinations of top winners (6mo window)")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    print("\nloading window...", flush=True)
    t0 = time.time()
    sig, bars, _ = get_v3_signals_and_bars(WINDOW[1], WINDOW[2], bundle,
                                           add_vol=False, add_mtf=True)
    from data_pull import cache as _cache
    vix = pd.Series(dtype=float)
    for vp in sorted(_cache.cache_dir().glob("yf_VIX_*.parquet"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty and df.index.max() >= pd.Timestamp(WINDOW[2], tz="UTC") - pd.Timedelta(days=10):
            vix = (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")
            break
    print(f"  loaded {len(sig)} signals in {time.time()-t0:.0f}s")

    # Baseline
    cfg = OptionsSimConfig(vix_series=vix, **BASELINE)
    sim = make_options_simulator(cfg)
    base_trades = walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)
    base = summarise(base_trades)
    print(f"\nBASELINE: n={base['n']} win={base['win_pct']:.1f}% $={base['total_$']:+,.0f} "
          f"DD=${base['max_dd_$']:+,.0f} Sh={base['sharpe']:+.2f}")
    baseline_total = base["total_$"]

    print()
    print(f"  {'idea':<48}{'n':>5}{'win%':>6}{'total $':>11}"
          f"{'gross loss':>13}{'maxDD':>9}{'Sh':>7}{'vs base':>11}")
    print("  " + "-"*107)

    results = [{"name": "BASELINE", **base, "vs_baseline": 0}]
    for c in COMBOS:
        kwargs = dict(BASELINE); kwargs.update(c["delta"])
        cfg = OptionsSimConfig(vix_series=vix, **kwargs)
        sim = make_options_simulator(cfg)
        trades = walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)
        m = summarise(trades, baseline_total)
        if m:
            print(f"  {c['name']:<48}{m['n']:>5}{m['win_pct']:>5.1f}%"
                  f"{m['total_$']:>+11,.0f}{m['gross_loss_$']:>+13,.0f}"
                  f"{m['max_dd_$']:>+9,.0f}{m['sharpe']:>+7.2f}"
                  f"{m['vs_baseline']:>+11,.0f}")
            results.append({"name": c["name"], **m})

    # Rank
    print()
    print("=" * 110)
    print(" Ranked top 10:")
    print("=" * 110)
    ranked = sorted(results[1:], key=lambda r: r["vs_baseline"], reverse=True)[:10]
    for i, r in enumerate(ranked, 1):
        beats_baseline_dd = abs(r["max_dd_$"]) <= abs(base["max_dd_$"])
        beats_sharpe = r["sharpe"] > base["sharpe"]
        marker = "✓✓" if (beats_baseline_dd and beats_sharpe) else "✓ " if beats_baseline_dd or beats_sharpe else "  "
        print(f"  {i:>2}. {r['name']:<48}  {marker}  $={r['total_$']:>+11,.0f}  "
              f"DD={r['max_dd_$']:>+,.0f}  Sh={r['sharpe']:>+.2f}")

    # Pareto-better than baseline (better total $ AND DD AND Sharpe)
    print()
    print("=" * 110)
    print(" PARETO-DOMINANT vs baseline (better total$, DD, AND Sharpe):")
    print("=" * 110)
    pareto = [r for r in results[1:]
              if r["total_$"] > base["total_$"]
              and abs(r["max_dd_$"]) <= abs(base["max_dd_$"])
              and r["sharpe"] >= base["sharpe"]]
    if pareto:
        for r in pareto:
            print(f"  ★ {r['name']}")
            print(f"     total: {r['total_$']:>+11,.0f}  vs base: {r['vs_baseline']:>+11,.0f}")
            print(f"     DD:    {r['max_dd_$']:>+11,.0f}  vs base: {abs(r['max_dd_$']) - abs(base['max_dd_$']):>+11,.0f}")
            print(f"     Sharpe:{r['sharpe']:>+11.2f}  vs base: {r['sharpe'] - base['sharpe']:>+.2f}")
    else:
        print("  None. All winners trade higher $ for worse DD or Sharpe.")

    out = Path(__file__).parent / "outputs" / "research_phase2.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
