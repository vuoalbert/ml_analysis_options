"""Comprehensive month-by-month backtest from Sep 2020 → Apr 2026.

Runs the deployed multi=15 / 7DTE config on each month independently.
For each month outputs: trades, win%, total $, $/trade, max drawdown,
daily-Sharpe (annualized), avg hold, days traded.

Approach: load 1-year chunks (with 1mo feature lookback), then slice
signals by month within each chunk. Avoids the per-window load overhead
that single-month loads would incur.

Run:
    python -m research.run_all_months
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


# Yearly chunks that align with cache structure. Each chunk includes
# extra lookback at start so first-month features are populated.
CHUNKS = [
    # (load_start, load_end, label) — full sweep with the new gross_losses metric
    ("2020-08-01", "2021-04-30", "2020-Sep–2021-Apr"),
    ("2021-04-01", "2022-04-30", "2021-Apr–2022-Apr"),
    ("2022-04-01", "2023-04-30", "2022-Apr–2023-Apr"),
    ("2023-04-01", "2024-04-30", "2023-Apr–2024-Apr"),
    ("2024-04-01", "2025-04-30", "2024-Apr–2025-Apr"),
    ("2025-04-01", "2026-04-30", "2025-Apr–2026-Apr"),
]

DEPLOY_CONFIG = {
    "dte": 7,
    "conviction_min": 0.55,
    "theta_protect_mins": 0,
    "risk_pct": 0.02,
    "max_qty": 100,
    "use_iv_dynamics": True,
    "iv_beta_call": 5.0,
    "iv_beta_put": 8.0,
}
MAX_CONCURRENT = 15


def _vix_from_cache():
    from data_pull import cache as _cache
    cache_root = _cache.cache_dir()
    # Stitch all VIX caches into one Series spanning the full history
    frames = []
    for vp in cache_root.glob("yf_VIX_*.parquet"):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        frames.append(df["close"] if "close" in df.columns else df.iloc[:, 0])
    if not frames:
        return pd.Series(dtype=float)
    full = pd.concat(frames).rename("vix_close")
    return full[~full.index.duplicated(keep="first")].sort_index()


def month_metrics(trades, label):
    if not trades:
        return {"month": label, "n": 0, "win_pct": 0, "total_$": 0,
                "avg_$": 0, "sharpe": 0, "max_dd_$": 0, "gross_losses_$": 0,
                "gross_wins_$": 0, "days": 0,
                "avg_hold": 0, "best_$": 0, "worst_$": 0}
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("et_date")["pnl_dollars"].sum()
    cum = daily.sort_index().cumsum()
    dd = cum - cum.cummax()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if (len(daily) > 1 and daily.std() > 0) else 0.0
    losing_trades = df[df["pnl_dollars"] < 0]
    winning_trades = df[df["pnl_dollars"] > 0]
    return {
        "month":    label,
        "n":        len(df),
        "win_pct":  float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$":  float(df["pnl_dollars"].sum()),
        "avg_$":    float(df["pnl_dollars"].mean()),
        "sharpe":   float(sharpe),
        "max_dd_$": float(dd.min() if len(dd) else 0.0),
        # gross_losses_$ = sum of all losing-trade P&L (negative number).
        # This is what "total monthly drawdown" usually means: the cumulative
        # dollar loss across every trade that lost, before being offset by winners.
        "gross_losses_$": float(losing_trades["pnl_dollars"].sum()),
        "gross_wins_$":   float(winning_trades["pnl_dollars"].sum()),
        "days":     int(df["et_date"].nunique()),
        "avg_hold": float(df["hold_min"].mean()) if "hold_min" in df.columns else 0,
        "best_$":   float(df["pnl_dollars"].max()),
        "worst_$":  float(df["pnl_dollars"].min()),
    }


def main():
    print("=" * 130)
    print(" MONTH-BY-MONTH BACKTEST — multi=15 / 7DTE / p≥0.55 / risk 2% / IV dynamics")
    print(" Period: Sep 2020 → Apr 2026 (68 months)")
    print(" Note: model trained 2023-04+. Sep 2020 → Mar 2023 is FULLY out-of-sample.")
    print("=" * 130)

    bundle = V3Bundle("mtf")
    vix = _vix_from_cache()
    print(f" VIX series: {len(vix)} obs from {vix.index.min()} to {vix.index.max()}")

    all_results = []

    for start, end, chunk_label in CHUNKS:
        print(f"\n[chunk {chunk_label}] loading {start} → {end} ...", flush=True)
        t0 = time.time()
        try:
            sig, bars, _ = get_v3_signals_and_bars(start, end, bundle,
                                                   add_vol=False, add_mtf=True)
            print(f"  got {len(sig)} signals, {bars.shape[0]} bars in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  load failed: {e}")
            continue

        if not sig:
            continue

        # Group signals by month and walk each separately
        signal_df = pd.DataFrame(sig)
        signal_df["ts"] = pd.to_datetime(signal_df["ts"])
        signal_df["month"] = signal_df["ts"].dt.tz_convert("America/New_York").dt.to_period("M")

        for month_period in sorted(signal_df["month"].unique()):
            month_label = str(month_period)
            month_sigs = signal_df[signal_df["month"] == month_period].to_dict("records")
            if not month_sigs:
                continue

            cfg = OptionsSimConfig(vix_series=vix, **DEPLOY_CONFIG)
            sim = make_options_simulator(cfg)
            trades = walk_multi(month_sigs, bars, sim, max_concurrent=MAX_CONCURRENT)
            m = month_metrics(trades, month_label)
            all_results.append(m)
            print(f"  {month_label}: n={m['n']:>4}  win={m['win_pct']:>5.1f}%  "
                  f"$={m['total_$']:>+9,.0f}  Sh={m['sharpe']:>+5.2f}  "
                  f"DD=${m['max_dd_$']:>+,.0f}", flush=True)

    # ---------- Final table ----------
    print()
    print("=" * 130)
    print(" SUMMARY TABLE — multi=15 / 7DTE per-month results")
    print("=" * 130)
    print(f"  {'month':<10}{'trades':>7}{'win%':>6}{'profit $':>11}"
          f"{'gross losses':>14}{'max DD':>10}{'Sharpe':>8}{'days':>5}")
    print("  " + "-"*80)
    cumul_dollars = 0
    cumul_trades = 0
    cumul_losses = 0
    for r in all_results:
        cumul_dollars += r["total_$"]
        cumul_trades += r["n"]
        cumul_losses += r.get("gross_losses_$", 0)
        print(f"  {r['month']:<10}{r['n']:>7}{r['win_pct']:>5.1f}%"
              f"{r['total_$']:>+11,.0f}"
              f"{r['gross_losses_$']:>+14,.0f}"
              f"{r['max_dd_$']:>+10,.0f}"
              f"{r['sharpe']:>+8.2f}{r['days']:>5}")

    print("  " + "-"*120)
    win_months = sum(1 for r in all_results if r["total_$"] > 0)
    print(f"  TOTAL    {cumul_trades:>8}     -    {cumul_dollars:>+11,.0f}")
    print(f"  Months profitable: {win_months}/{len(all_results)} ({win_months/max(len(all_results),1)*100:.0f}%)")
    print(f"  Best month:  {max(all_results, key=lambda r: r['total_$'])['month']}  "
          f"+${max(r['total_$'] for r in all_results):,.0f}")
    print(f"  Worst month: {min(all_results, key=lambda r: r['total_$'])['month']}  "
          f"${min(r['total_$'] for r in all_results):+,.0f}")
    print(f"  Avg month:   ${cumul_dollars/max(len(all_results),1):+,.0f}")

    # Save CSV
    out_csv = Path(__file__).parent / "outputs" / "all_months_multi15.csv"
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\n  CSV saved: {out_csv}")


if __name__ == "__main__":
    main()
