"""Phase 3: validate top Phase 2 winners across multiple months.

Top winners on April:
  1. ITM 1.5% + target 30%  (+$22,627 vs base)
  2. ITM 1% + target 30%    (+$18,486)
  3. ITM 2.0%                (+$18,371)
  4. ITM 1.5%                (+$14,027)
  5. ITM 1% + target 50%     (+$9,422)

Test each across 5 monthly windows (Dec 2025 → Apr 2026) and one
pre-training month (2022-05, the bear-market peak month for baseline).

Run:
    python -m research.run_research_phase3
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


MONTHS = [
    ("Dec 2025",  "2025-12-01", "2025-12-31"),
    ("Jan 2026",  "2026-01-01", "2026-01-31"),
    ("Feb 2026",  "2026-02-01", "2026-02-28"),
    ("Mar 2026",  "2026-03-01", "2026-03-31"),
    ("Apr 2026",  "2026-04-01", "2026-04-30"),
    # Pre-training stress test
    ("May 2022 (bear, OOS)",  "2022-05-01", "2022-05-31"),
    ("Sep 2022 (deep bear, OOS)", "2022-09-01", "2022-09-30"),
]
MAX_CONCURRENT = 15

BASELINE = {
    "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "atm",
}

VARIANTS = [
    {"name": "BASELINE",                 "delta": {}},
    {"name": "ITM 1.5% + target 30%",    "delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.30}},
    {"name": "ITM 1% + target 30%",      "delta": {"moneyness": "itm", "itm_offset_pct": 0.01,  "target_pct": 0.30}},
    {"name": "ITM 2%",                   "delta": {"moneyness": "itm", "itm_offset_pct": 0.02}},
    {"name": "ITM 1.5%",                 "delta": {"moneyness": "itm", "itm_offset_pct": 0.015}},
    {"name": "ITM 1.5% + target 50%",    "delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.50}},
]


def _vix_from_cache(end_str):
    from data_pull import cache as _cache
    end_ts = pd.Timestamp(end_str, tz="UTC")
    candidates = []
    for vp in _cache.cache_dir().glob("yf_VIX_*.parquet"):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty and df.index.min() <= end_ts <= df.index.max() + pd.Timedelta(days=10):
            candidates.append((vp.stat().st_mtime, df))
    if not candidates:
        # fallback
        for vp in sorted(_cache.cache_dir().glob("yf_VIX_*.parquet"),
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
    print("=" * 130)
    print(" PHASE 3 — multi-month validation of top winners")
    print("=" * 130)

    bundle = V3Bundle("mtf")

    # results[month][variant_name] = metrics
    results: dict = {}

    for label, start, end in MONTHS:
        print(f"\n[{label}] {start} → {end}", flush=True)
        t0 = time.time()
        try:
            sig, bars, _ = get_v3_signals_and_bars(start, end, bundle,
                                                   add_vol=False, add_mtf=True)
            vix = _vix_from_cache(end)
            print(f"  loaded {len(sig)} signals in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        if not sig:
            continue

        results[label] = {}
        for v in VARIANTS:
            kwargs = dict(BASELINE); kwargs.update(v["delta"])
            cfg = OptionsSimConfig(vix_series=vix, **kwargs)
            sim = make_options_simulator(cfg)
            trades = walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)
            m = summarise(trades) or {"n": 0, "total_$": 0, "max_dd_$": 0, "sharpe": 0, "gross_loss_$": 0, "win_pct": 0}
            results[label][v["name"]] = m
            print(f"    {v['name']:<28} n={m['n']:>4}  win={m['win_pct']:>5.1f}%  "
                  f"$={m['total_$']:>+9,.0f}  DD=${m['max_dd_$']:>+,.0f}  "
                  f"Sh={m['sharpe']:>+5.2f}", flush=True)

    # Summary table — total $ per variant per month
    print()
    print("=" * 130)
    print(" SUMMARY — Total $ per variant × month")
    print("=" * 130)
    var_names = [v["name"] for v in VARIANTS]
    print(f"  {'month':<28}" + "".join(f"{n[:18]:>20}" for n in var_names))
    print("  " + "-"*(28 + 20*len(var_names)))
    totals = {n: 0 for n in var_names}
    for label, _, _ in MONTHS:
        if label not in results:
            continue
        row = f"  {label:<28}"
        for n in var_names:
            r = results[label].get(n)
            if r:
                row += f"{r['total_$']:>+20,.0f}"
                totals[n] += r["total_$"]
            else:
                row += f"{'—':>20}"
        print(row)

    print("  " + "-"*(28 + 20*len(var_names)))
    print(f"  {'TOTAL':<28}" + "".join(f"{totals[n]:>+20,.0f}" for n in var_names))

    # Pareto check — does any variant dominate baseline on every month?
    print()
    print("=" * 130)
    print(" PARETO check — variants beating baseline in EVERY month")
    print("=" * 130)
    base_per_month = {label: results[label].get("BASELINE", {}).get("total_$", 0) for label, _, _ in MONTHS if label in results}
    for v in VARIANTS:
        if v["name"] == "BASELINE":
            continue
        per_month = {label: results[label].get(v["name"], {}).get("total_$", 0)
                     for label in base_per_month}
        beats_all = all(per_month[m] > base_per_month[m] for m in base_per_month)
        marker = "★ DOMINATES BASELINE" if beats_all else "(falls short on some month)"
        print(f"  {v['name']:<28}  {marker}")

    out = Path(__file__).parent / "outputs" / "research_phase3.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for label, _, _ in MONTHS:
        if label not in results:
            continue
        for n in var_names:
            r = results[label].get(n)
            if r:
                rows.append({"month": label, "variant": n, **r})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
