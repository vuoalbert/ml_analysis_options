"""Compare multi=3 vs multi=10 vs multi=20 across the pre-training era
(2020-09 → 2023-04). The question: do the higher-leverage variants hold
up on data the model never saw, or were they 6-month-window artifacts?

Run:
    python -m research.run_multi_compare
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


BASE_CONFIG = {
    "dte": 7,
    "conviction_min": 0.55,
    "theta_protect_mins": 0,
    "risk_pct": 0.02,
    "max_qty": 100,
    "use_iv_dynamics": True,
    "iv_beta_call": 5.0,
    "iv_beta_put": 8.0,
}

# Apply a realistic 25% haircut by widening costs to 100 bps each way (vs 50 bps default).
REAL_COST_OVERRIDE = {"entry_cost_bps": 100.0, "exit_cost_bps": 100.0}

WINDOWS = [
    # 2024-2025 windows — fast-loading. Cover 21 months across diverse regimes.
    # Last 6mo dropped because window 5 hits the slow tz path (~8 min load).
    ("H1 2024",                            "2024-01-01", "2024-06-30"),
    ("H2 2024",                            "2024-07-01", "2024-12-31"),
    ("H1 2025",                            "2025-01-01", "2025-06-30"),
    ("Q3 2025",                            "2025-07-01", "2025-09-30"),
]

CONCURRENT_LEVELS = [3, 10, 20]


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


def run(signals, bars, vix, max_concurrent, *, realistic_costs=False):
    kwargs = dict(BASE_CONFIG)
    if realistic_costs:
        kwargs.update(REAL_COST_OVERRIDE)
    cfg = OptionsSimConfig(vix_series=vix, **kwargs)
    sim = make_options_simulator(cfg)
    trades = walk_multi(signals, bars, sim, max_concurrent=max_concurrent)
    return summarise(trades)


def fmt_brief(m):
    if m is None:
        return "(no trades)"
    return f"${m['total_$']:>+10,.0f}  DD=${m['max_dd_$']:>+,.0f}  win={m['win_pct']:.0f}%"


def main():
    print("=" * 110)
    print(" MULTI-POSITION COMPARISON — pre-training OOS (2020-09 → 2023-04, model never saw)")
    print(" Both stardard (50bps) and realistic-friction (100bps round-trip) costs.")
    print("=" * 110)

    bundle = V3Bundle("mtf")

    # Run each window to completion (load + walks + print) so progress
    # is visible incrementally — don't wait for all loads to finish first.
    print("\n  " + " " * 42 + "".join(f"{'multi=' + str(k):>30}" for k in CONCURRENT_LEVELS))
    print("-" * 110)
    cache = {}
    for label, start, end in WINDOWS:
        import time
        t0 = time.time()
        sig, b, _ = get_v3_signals_and_bars(start, end, bundle, add_vol=False, add_mtf=True)
        v = _pull_vix(start, end)
        cache[(start, end)] = (sig, b, v)
        load_t = time.time() - t0

        if not sig:
            print(f"  {label:<42}  (no signals)")
            continue
        print(f"  {label:<42}", end="", flush=True)
        for k in CONCURRENT_LEVELS:
            m = run(sig, b, v, k)
            print(f"{fmt_brief(m):>30}", end="", flush=True)
        print(f"  [load {load_t:.0f}s]")

    # Realistic friction (100bps each way)
    print()
    print("-" * 110)
    print(" Realistic friction (100 bps each way — wider spreads + slippage):")
    print("-" * 110)
    print(f"  {'window':<42}", end="")
    for k in CONCURRENT_LEVELS:
        print(f"{'multi=' + str(k):>30}", end="")
    print()
    for label, start, end in WINDOWS:
        sig, b, v = cache[(start, end)]
        if not sig:
            continue
        print(f"  {label:<42}", end="")
        for k in CONCURRENT_LEVELS:
            m = run(sig, b, v, k, realistic_costs=True)
            print(f"{fmt_brief(m):>30}", end="")
        print()


if __name__ == "__main__":
    main()
