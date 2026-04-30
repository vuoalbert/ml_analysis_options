"""Pure pre-training OOS backtest — 2020 through Q1 2023.

The v3_mtf model was trained on data 2023-04-01 onwards. Everything before
that is data the model has never seen. This is the strongest possible test
of generalization.

We run the deployable winner config (7DTE, p≥0.55, risk=2%, multi=3, IV
dynamics on) across 5 non-overlapping 6mo slices spanning post-COVID recovery,
2021 bull market, 2022 bear, and Q1 2023 banking crisis.

Run:
    python -m research.run_pretraining_options
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


# Deployable winner — locked in v1.yaml
WINNER = {
    "dte": 7,
    "conviction_min": 0.55,
    "theta_protect_mins": 0,
    "risk_pct": 0.02,
    "max_qty": 100,
    "use_iv_dynamics": True,
    "iv_beta_call": 5.0,
    "iv_beta_put": 8.0,
}
WINNER_MAX_CONCURRENT = 3

WINDOWS = [
    ("Post-COVID recovery   (2020 H2 + Q1 2021)", "2020-09-01", "2021-03-31"),
    ("Mid-2021 grind higher (Q2-Q3 2021)",        "2021-04-01", "2021-09-30"),
    ("2021 peak + bear start (Q4 21 + Q1 22)",    "2021-10-01", "2022-03-31"),
    ("Deep 2022 bear        (Q2-Q3 2022)",        "2022-04-01", "2022-09-30"),
    ("Capitulation + SVB    (Q4 22 + Q1 23)",     "2022-10-01", "2023-03-31"),
    ("Full 30 months        (2020-09 → 2023-04)", "2020-09-01", "2023-03-31"),
]


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


def run(signals, bars, vix):
    cfg = OptionsSimConfig(vix_series=vix, **WINNER)
    sim = make_options_simulator(cfg)
    trades = walk_multi(signals, bars, sim, max_concurrent=WINNER_MAX_CONCURRENT)
    return summarise(trades), trades


def fmt(m):
    if m is None:
        return "(no trades)"
    return (f"n={m['n']:>4}  win={m['win_pct']:>5.1f}%  $={m['total_$']:>+9,.0f}  "
            f"$/tr={m['avg_$']:>+5,.0f}  Sh={m['sharpe_d']:>+5.2f}  DD=${m['max_dd_$']:>+,.0f}  "
            f"days={m['n_days']:>3}")


def main():
    print("=" * 110)
    print(" PURE PRE-TRAINING OOS — model trained on 2023-04-01+, this tests 2020-09 → 2023-04")
    print(f" Winner config: 7DTE, p≥{WINNER['conviction_min']}, "
          f"risk={WINNER['risk_pct']:.0%}, multi={WINNER_MAX_CONCURRENT}, IV dynamics ON")
    print("=" * 110)

    bundle = V3Bundle("mtf")

    summaries = []
    for label, start, end in WINDOWS:
        try:
            print()
            print(f"  {label}")
            print(f"  {start} → {end}")
            sig, b, _ = get_v3_signals_and_bars(start, end, bundle, add_vol=False, add_mtf=True)
            v = _pull_vix(start, end)
            if not sig or b.empty:
                print("    (no data)")
                continue
            m, trades = run(sig, b, v)
            print(f"    {fmt(m)}")
            summaries.append((label, start, end, m, trades))
        except Exception as e:
            print(f"    error: {e}")

    print()
    print("=" * 110)
    print(" Summary table")
    print("=" * 110)
    print(f"  {'window':<46}{'trades':>7}{'win%':>7}{'total $':>12}{'$/tr':>7}"
          f"{'Sharpe':>9}{'DD$':>10}")
    for label, start, end, m, _ in summaries:
        if m:
            print(f"  {label:<46}{m['n']:>7}{m['win_pct']:>6.1f}%"
                  f"{m['total_$']:>+12,.0f}{m['avg_$']:>+7,.0f}"
                  f"{m['sharpe_d']:>+9.2f}{m['max_dd_$']:>+10,.0f}")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    for label, start, end, m, trades in summaries:
        if trades:
            safe = f"pretraining_{start}_{end}".replace("-", "")
            pd.DataFrame(trades).to_csv(out_dir / f"{safe}.csv", index=False)


if __name__ == "__main__":
    main()
