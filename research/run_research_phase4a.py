"""Phase 4a: validate winner (ITM 1.5% + target 30%) across more pre-training months.

Phase 3 confirmed the variant dominates on 5 recent + 2 bear pre-training months.
4a checks: does it hold up across non-bear pre-training regimes too?

Test on 12 individual pre-training months covering:
  - 2020 H2 recovery (Sept-Dec)
  - 2021 H1 (calm bull)
  - 2021 H2 (mid-year mixed)
  - 2022 H1 (peak + bear start)
  - 2023 H1 (banking crisis recovery)

If winner Pareto-dominates baseline in ≥80% of these, we ship.

Run:
    python -m research.run_research_phase4a
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


# 12 pre-training months across diverse regimes
MONTHS = [
    ("Sep 2020 — recovery",      "2020-09-01", "2020-09-30"),
    ("Nov 2020 — election",      "2020-11-01", "2020-11-30"),
    ("Feb 2021 — bull continue", "2021-02-01", "2021-02-28"),
    ("May 2021 — calm grind",    "2021-05-01", "2021-05-31"),
    ("Aug 2021 — chop",          "2021-08-01", "2021-08-31"),
    ("Nov 2021 — peak/correct",  "2021-11-01", "2021-11-30"),
    ("Jan 2022 — bear start",    "2022-01-01", "2022-01-31"),
    ("Apr 2022 — bear continue", "2022-04-01", "2022-04-30"),
    ("Jul 2022 — bear bounce",   "2022-07-01", "2022-07-31"),
    ("Oct 2022 — capitulation",  "2022-10-01", "2022-10-31"),
    ("Feb 2023 — recovery",      "2023-02-01", "2023-02-28"),
    ("Mar 2023 — SVB",           "2023-03-01", "2023-03-31"),
]

MAX_CONCURRENT = 15

BASELINE = {
    "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "atm",
}

# Top winners from Phase 3 — focus on the 2 Pareto-dominators + 1 challenger
VARIANTS = [
    {"name": "BASELINE",                 "delta": {}},
    {"name": "ITM 1.5% + target 30%",    "delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.30}},
    {"name": "ITM 1.5% + target 50%",    "delta": {"moneyness": "itm", "itm_offset_pct": 0.015, "target_pct": 0.50}},
    {"name": "ITM 2% + target 30%",      "delta": {"moneyness": "itm", "itm_offset_pct": 0.020, "target_pct": 0.30}},
    {"name": "ITM 2.5% + target 30%",    "delta": {"moneyness": "itm", "itm_offset_pct": 0.025, "target_pct": 0.30}},
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
    return {
        "n": len(df),
        "win_pct": float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$": float(df["pnl_dollars"].sum()),
        "sharpe": float(sharpe),
        "max_dd_$": float(dd.min() if len(dd) else 0.0),
    }


def main():
    print("=" * 130)
    print(" PHASE 4a — pre-training validation across 12 monthly windows")
    print(" Question: does ITM 1.5% + target 30% dominate baseline in non-bear regimes too?")
    print("=" * 130)

    bundle = V3Bundle("mtf")
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
            m = summarise(trades) or {"n": 0, "total_$": 0, "max_dd_$": 0, "sharpe": 0, "win_pct": 0}
            results[label][v["name"]] = m
            print(f"    {v['name']:<28} n={m['n']:>4}  win={m['win_pct']:>5.1f}%  "
                  f"$={m['total_$']:>+9,.0f}  DD=${m['max_dd_$']:>+,.0f}  "
                  f"Sh={m['sharpe']:>+5.2f}", flush=True)

    # Summary table
    print()
    print("=" * 130)
    print(" SUMMARY — Total $ per variant × month")
    print("=" * 130)
    var_names = [v["name"] for v in VARIANTS]
    print(f"  {'month':<28}" + "".join(f"{n[:18]:>20}" for n in var_names))
    print("  " + "-"*(28 + 20*len(var_names)))
    totals = {n: 0 for n in var_names}
    win_count = {n: 0 for n in var_names if n != "BASELINE"}
    for label, _, _ in MONTHS:
        if label not in results:
            continue
        row = f"  {label:<28}"
        base = results[label].get("BASELINE", {}).get("total_$", 0)
        for n in var_names:
            r = results[label].get(n)
            if r:
                row += f"{r['total_$']:>+20,.0f}"
                totals[n] += r["total_$"]
                if n != "BASELINE" and r["total_$"] > base:
                    win_count[n] += 1
            else:
                row += f"{'—':>20}"
        print(row)

    print("  " + "-"*(28 + 20*len(var_names)))
    print(f"  {'TOTAL':<28}" + "".join(f"{totals[n]:>+20,.0f}" for n in var_names))

    # Pareto check
    print()
    print("=" * 130)
    print(" Win-rate vs baseline across 12 months:")
    print("=" * 130)
    n_months = len([1 for label, _, _ in MONTHS if label in results])
    for n in var_names:
        if n == "BASELINE":
            continue
        wr = win_count[n] / n_months * 100
        marker = "★" if wr >= 80 else "✓" if wr >= 50 else "✗"
        print(f"  {marker}  {n:<28}  beats baseline in {win_count[n]}/{n_months} months ({wr:.0f}%)")

    out = Path(__file__).parent / "outputs" / "research_phase4a.csv"
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
