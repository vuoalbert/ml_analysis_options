"""Re-run the top variants with IV dynamics ON to see how much the
unrealistic backtest numbers haircut.

Run:
    python -m research.sweep_v3_realism
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
STOCK_BENCHMARK = 41455.0


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
    }


def run_pair(signals, bars, vix, *, max_concurrent, **kwargs):
    """Run with IV dynamics OFF and ON, return both summaries."""
    out = {}
    for label, dyn in [("CONST_IV", False), ("DYN_IV", True)]:
        cfg = OptionsSimConfig(vix_series=vix, max_qty=100,
                               use_iv_dynamics=dyn, **kwargs)
        sim = make_options_simulator(cfg)
        trades = walk_multi(signals, bars, sim, max_concurrent=max_concurrent)
        out[label] = summarise(trades)
    return out


VARIANTS = [
    {"tag": "Original 7DTE p≥0.65 (single position)",
     "max_concurrent": 1,
     "dte": 7, "conviction_min": 0.65, "theta_protect_mins": 0, "risk_pct": 0.01},
    {"tag": "C multi=3 risk=2%       (realistic winner)",
     "max_concurrent": 3,
     "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0, "risk_pct": 0.02},
    {"tag": "C multi=5 risk=2%",
     "max_concurrent": 5,
     "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0, "risk_pct": 0.02},
    {"tag": "E multi=5 + ITM + conv-weight",
     "max_concurrent": 5,
     "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0,
     "moneyness": "itm", "itm_offset_pct": 0.005,
     "risk_pct": 0.005, "risk_pct_max": 0.04,
     "conviction_lo": 0.55, "conviction_hi": 0.75},
    {"tag": "C multi=10 risk=2%",
     "max_concurrent": 10,
     "dte": 7, "conviction_min": 0.55, "theta_protect_mins": 0, "risk_pct": 0.02},
]


def main():
    print("=" * 110)
    print(" REALISM CHECK: re-running top variants with IV dynamics ON")
    print(f" Window: {WINDOW[1]} → {WINDOW[2]}    Target: beat +${STOCK_BENCHMARK:,.0f}")
    print("=" * 110)

    bundle = V3Bundle("mtf")
    signals, bars, _ = get_v3_signals_and_bars(
        WINDOW[1], WINDOW[2], bundle, add_vol=False, add_mtf=True)
    vix = _pull_vix(WINDOW[1], WINDOW[2])
    print(f" {len(signals)} candidate signals\n")

    print(f"  {'variant':<55}{'CONST IV $':>14}{'DYN IV $':>14}{'haircut':>11}{'beats':>8}")
    print("  " + "-"*102)
    for v in VARIANTS:
        tag = v.pop("tag")
        max_conc = v.pop("max_concurrent")
        out = run_pair(signals, bars, vix, max_concurrent=max_conc, **v)
        c, d = out["CONST_IV"], out["DYN_IV"]
        if c is None or d is None:
            print(f"  {tag:<55} (no trades)")
            continue
        haircut = (1 - d["total_$"] / c["total_$"]) * 100 if c["total_$"] > 0 else 0
        beats = "✓" if d["total_$"] > STOCK_BENCHMARK else " "
        print(f"  {tag:<55}{c['total_$']:>+14,.0f}{d['total_$']:>+14,.0f}"
              f"{haircut:>10.1f}%  {beats}")
        # extra detail line
        print(f"    {'':<53}n={d['n']:<5} win={d['win_pct']:.1f}%  "
              f"DD=${d['max_dd_$']:+,.0f}  Sh={d['sharpe_d']:+.2f}")


if __name__ == "__main__":
    main()
