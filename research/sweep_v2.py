"""Phase-1+ sweep: scale-up + multi-position + conviction-weighted + ITM.

Goal: beat the stock strategy's +$41,455 / 6mo. Test cheap scaling levers
on top of the validated 7DTE p≥0.65 winner.

Run:
    python -m research.sweep_v2
"""
from __future__ import annotations

import sys
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


WINDOW = ("Last 6 months", "2025-11-01", "2026-04-29")
HOLDOUT = ("Apr 2026 OOS", "2026-04-01", "2026-04-29")
STOCK_BENCHMARK = 41455.0
MAX_QTY_HI = 100  # raise the contract cap to let scaling work


def _pull_vix(start, end):
    cfg = load_cfg("v1")
    out = build_frame(start, end, cfg)
    return out["vix_close"].dropna() if not out.empty else pd.Series(dtype=float)


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
        "n_days":    int(df["et_date"].nunique()),
    }


def run(signals, bars, vix, *, max_concurrent=1, **kwargs):
    cfg = OptionsSimConfig(vix_series=vix, max_qty=MAX_QTY_HI, **kwargs)
    sim = make_options_simulator(cfg)
    trades = walk_multi(signals, bars, sim, max_concurrent=max_concurrent)
    return summarise(trades)


def fmt_row(rank, tag, m):
    if m is None:
        return f"  {rank:<4}{tag:<54}  no trades"
    return (f"  {rank:<4}{tag:<54}n={m['n']:>4}  win={m['win_pct']:>5.1f}%  "
            f"$={m['total_$']:+10,.0f}  $/tr={m['avg_$']:>+5,.0f}  "
            f"Sh={m['sharpe_d']:>+5.2f}  DD=${m['max_dd_$']:>+,.0f}")


# ---------- The grid: ordered phases ----------

GRID = []

# --- Group A: pure scaling on the best 7DTE config ---
for risk in [0.01, 0.02, 0.03, 0.04, 0.05]:
    GRID.append({"tag": f"A scale risk={risk:.2%}    7DTE p≥0.65",
                 "dte": 7, "conviction_min": 0.65,
                 "theta_protect_mins": 0, "risk_pct": risk})

# --- Group B: conviction-weighted sizing (scale risk with p) ---
for risk_max in [0.02, 0.04, 0.06, 0.08]:
    GRID.append({"tag": f"B conv-weighted 0.5%→{risk_max:.0%}    7DTE p≥0.55",
                 "dte": 7, "conviction_min": 0.55,
                 "theta_protect_mins": 0,
                 "risk_pct": 0.005, "risk_pct_max": risk_max,
                 "conviction_lo": 0.55, "conviction_hi": 0.75})

# --- Group C: multi-position (allow up to N concurrent) ---
for max_conc in [3, 5, 10, 20]:
    GRID.append({"tag": f"C multi={max_conc} pos    7DTE p≥0.55 risk=2%",
                 "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
                 "risk_pct": 0.02, "max_concurrent": max_conc})

# --- Group D: ITM (delta ≈ 0.7) for higher $/trade ---
for risk in [0.01, 0.02, 0.03]:
    GRID.append({"tag": f"D ITM 0.5% strike risk={risk:.0%}    7DTE p≥0.65",
                 "dte": 7, "conviction_min": 0.65, "theta_protect_mins": 0,
                 "moneyness": "itm", "itm_offset_pct": 0.005,
                 "risk_pct": risk})

# --- Group E: combined — multi + conv-weighted + ITM ---
GRID.append({"tag": "E combined: multi=10 + ITM + conv-weight",
             "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
             "moneyness": "itm", "itm_offset_pct": 0.005,
             "risk_pct": 0.005, "risk_pct_max": 0.04,
             "conviction_lo": 0.55, "conviction_hi": 0.75,
             "max_concurrent": 10})
GRID.append({"tag": "E combined: multi=20 + ITM + conv-weight aggressive",
             "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
             "moneyness": "itm", "itm_offset_pct": 0.005,
             "risk_pct": 0.01, "risk_pct_max": 0.06,
             "conviction_lo": 0.55, "conviction_hi": 0.75,
             "max_concurrent": 20})
GRID.append({"tag": "E combined: multi=20 + ATM + conv-weight aggressive",
             "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
             "risk_pct": 0.01, "risk_pct_max": 0.06,
             "conviction_lo": 0.55, "conviction_hi": 0.75,
             "max_concurrent": 20})


def main():
    print("=" * 110)
    print(" PHASE 1 SWEEP — scale-up + multi-position + conviction-weighted + ITM")
    print(f" Target: beat stock baseline of +${STOCK_BENCHMARK:,.0f}")
    print(f" Window: {WINDOW[1]} → {WINDOW[2]}")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    signals, bars, _ = get_v3_signals_and_bars(
        WINDOW[1], WINDOW[2], bundle, add_vol=False, add_mtf=True)
    vix = _pull_vix(WINDOW[1], WINDOW[2])
    print(f" {len(signals)} candidate signals\n")

    results = []
    for i, params in enumerate(GRID):
        tag = params.pop("tag")
        max_conc = params.pop("max_concurrent", 1)
        m = run(signals, bars, vix, max_concurrent=max_conc, **params)
        if m is not None:
            m["tag"] = tag
            m["params"] = {**params, "max_concurrent": max_conc}
            results.append(m)
        print(fmt_row(f"{i+1}", tag, m))

    print()
    print("=" * 110)
    print(" Leaderboard (sorted by total $)")
    print("=" * 110)
    results.sort(key=lambda r: r["total_$"], reverse=True)
    for i, r in enumerate(results, 1):
        beats = "✓" if r["total_$"] > STOCK_BENCHMARK else " "
        print(f"  {beats} {fmt_row(i, r['tag'], r)}")

    best = results[0]
    print()
    if best["total_$"] > STOCK_BENCHMARK:
        print(f"BEAT TARGET — best is {best['tag']} at +${best['total_$']:,.0f}")
    else:
        gap = STOCK_BENCHMARK - best["total_$"]
        ratio = best["total_$"] / STOCK_BENCHMARK
        print(f"Best so far: {best['tag']} at +${best['total_$']:,.0f}  "
              f"({ratio:.0%} of target, gap of -${gap:,.0f})")

    # holdout sanity
    print()
    print("=" * 110)
    print(f" Holdout: {HOLDOUT[1]} → {HOLDOUT[2]} (top 5)")
    print("=" * 110)
    h_signals, h_bars, _ = get_v3_signals_and_bars(
        HOLDOUT[1], HOLDOUT[2], bundle, add_vol=False, add_mtf=True)
    h_vix = _pull_vix(HOLDOUT[1], HOLDOUT[2])
    for r in results[:5]:
        params = dict(r["params"])
        max_conc = params.pop("max_concurrent", 1)
        m = run(h_signals, h_bars, h_vix, max_concurrent=max_conc, **params)
        print(fmt_row("h", r["tag"], m))

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame([{**r, "params": str(r["params"])} for r in results]).to_csv(
        out_dir / "options_sweep_v2.csv", index=False)


if __name__ == "__main__":
    main()
