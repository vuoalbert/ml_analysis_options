"""V2 combined research model — three improvements stacked vs current live.

Components:
  1. Better entry model         (research_v2_entry: full 3yr, all 61 features)
  2. Joint exit prediction      (research_v2_exit: stop/target/hold from features)
  3. Time-of-day filter         (skip entries 15:30-16:00 ET; optional 13:30-14:30)

Compares three rolling builds against current live across 5 windows:

  A. Current live              — research_3yr_top30 + K=1.5 + 1:2 R/R + conviction + cap=2.0
  B. V2 entry only             — drop in research_v2_entry, otherwise live
  C. V2 entry + V2 exits       — also use predicted stop/target/hold per trade
  D. V2 entry + V2 exits + TOD — also skip 15:30-16:00 ET

Run:
    python -m research.run_v2_combined
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from utils.config import load as load_cfg
from features.build import build as build_features
from model.artifact import load as load_artifact
from research.backtest_original import build_frame
from research.vol_scaled_exits import (
    compute_exit_plan, simulate_one_trade_no_horizon,
    simulate_one_trade, ExitPlan,
)
from research.backtest_compare import (
    walk_with_blocking, summarize, vol_trade_no_horizon_conviction,
    EQUITY, COST_BPS,
)
from research.v2_predictors import load as load_v2_exits


# ---------- TOD filter ----------

def in_tod_filter(ts: pd.Timestamp, skip_late: bool = True,
                   skip_lunch: bool = False) -> bool:
    """Returns True if this entry timestamp should be SKIPPED."""
    et = ts.tz_convert("America/New_York")
    h, m = et.hour, et.minute
    minutes = h * 60 + m
    if skip_late and minutes >= 15 * 60 + 30:    # 15:30+
        return True
    if skip_lunch and 13 * 60 + 30 <= minutes < 14 * 60 + 30:  # 13:30-14:30
        return True
    return False


# ---------- predictions and bars (mirrors backtest_compare but pluggable artifact) ----------

def get_signals_features_bars(start: str, end: str, *, artifact_name: str, cfg_name: str = "v1"):
    c = load_cfg(cfg_name)
    art = load_artifact(artifact_name)
    sym = c["universe"]["symbol"].lower()
    thr_up = float(art.thresholds["up"])
    thr_dn = float(art.thresholds["down"])

    out = build_frame(start, end, c)
    if out.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    feats = build_features(out, c)
    for col in art.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[art.feature_cols]
    essential = [col for col in art.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    feats = feats[feats[essential].notna().all(axis=1)]
    if feats.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    proba = art.booster.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    pred["close"] = out.loc[pred.index, f"{sym}_close"]

    # Vectorized RTH mask (was per-element .apply — 100× slower)
    minutes_of_day = pred.index.hour * 60 + pred.index.minute
    rth_lo = 13 * 60 + 30 + c["risk"]["skip_first_minutes"]
    rth_hi = 20 * 60 - c["risk"]["skip_last_minutes"]
    pred = pred[(minutes_of_day >= rth_lo) & (minutes_of_day < rth_hi)]

    bars = out[[f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close"]].rename(columns={
        f"{sym}_open": "open", f"{sym}_high": "high",
        f"{sym}_low": "low", f"{sym}_close": "close",
    }).dropna()

    signals = []
    for ts, row in pred.iterrows():
        side = None
        if row["p_up"] >= thr_up:
            side = "long"
        elif row["p_down"] >= thr_dn:
            side = "short"
        if side is None:
            continue
        signals.append({"ts": ts, "side": side,
                         "p_up": float(row["p_up"]),
                         "p_dn": float(row["p_down"])})
    return signals, bars, feats


# ---------- variants ----------

def variant_A_live(bars, entry):
    """Current live: K=1.5 + 1:2 R/R + conviction + cap 2.0."""
    return vol_trade_no_horizon_conviction(
        bars, entry, K=1.5, rr=2.0,
        risk_pct_base=0.005, risk_pct_max=0.010,
        max_notional_frac=2.0,
    )


def make_variant_C(v2_exits, feats: pd.DataFrame, *,
                     min_stop_bps: float = 5.0, max_stop_bps: float = 100.0,
                     min_target_bps: float = 5.0, max_target_bps: float = 200.0,
                     conviction_lo: float = 0.55, conviction_hi: float = 0.70,
                     risk_pct_base: float = 0.005, risk_pct_max: float = 0.010,
                     max_notional_frac: float = 2.0,
                     tod_filter: bool = False, tod_skip_lunch: bool = False):
    """V2 entry + V2 exits + (optional) TOD filter."""
    def _variant(bars, entry):
        ts = entry["ts"]; side = entry["side"]
        if tod_filter and in_tod_filter(ts, skip_late=True, skip_lunch=tod_skip_lunch):
            return None
        if ts not in bars.index:
            idx = bars.index.searchsorted(ts)
            if idx >= len(bars):
                return None
            ts = bars.index[idx]
        ei = bars.index.get_loc(ts)
        if ei < 31:
            return None
        if ts not in feats.index:
            return None
        entry_price = float(bars.iloc[ei]["close"])

        # Predict stop, target, hold from features
        feat_row = feats.loc[ts].values.reshape(1, -1)
        stop_bps  = float(np.clip(v2_exits.stop_model.predict(feat_row)[0],
                                    min_stop_bps, max_stop_bps))
        target_bps = float(np.clip(v2_exits.target_model.predict(feat_row)[0],
                                     max(stop_bps * 1.0, min_target_bps), max_target_bps))
        hold_min  = int(np.clip(round(v2_exits.hold_model.predict(feat_row)[0]), 5, 390))

        # Conviction-weighted risk_pct
        p = entry["p_up"] if side == "long" else entry["p_dn"]
        s = max(0.0, min(1.0, (p - conviction_lo) / max(conviction_hi - conviction_lo, 1e-9)))
        risk_pct = risk_pct_base + (risk_pct_max - risk_pct_base) * s

        # Build a manual ExitPlan (we're bypassing the K-based stop calc)
        risk_dollars = EQUITY * risk_pct
        stop_distance_dollars = entry_price * (stop_bps / 1e4)
        if stop_distance_dollars <= 0:
            return None
        qty_risk = int(risk_dollars // stop_distance_dollars)
        qty_cap = int((EQUITY * max_notional_frac) // max(entry_price, 1e-9))
        qty = max(1, min(qty_risk, qty_cap))
        plan = ExitPlan(
            stop_bps=stop_bps, target_bps=target_bps, qty=qty,
            risk_dollars=risk_dollars, rv_bps=0.0,
            note=f"v2 stop={stop_bps:.1f} tgt={target_bps:.1f} hold={hold_min} qty={qty}",
        )

        # Exit: predicted stop/target with predicted hold as horizon cap
        out = simulate_one_trade(bars, ts, side, plan, horizon_min=hold_min)
        if out is None:
            return None
        net_bps = out.pnl_bps - COST_BPS
        return {
            "entry_ts": out.entry_ts, "exit_ts": out.exit_ts, "side": out.side,
            "qty": qty, "entry_price": out.entry_price, "exit_price": out.exit_price,
            "pnl_dollars": out.pnl_dollars,
            "pnl_bps_gross": out.pnl_bps,
            "pnl_bps_net": net_bps,
            "reason": out.reason,
            "stop_bps": stop_bps, "target_bps": target_bps,
            "rv_bps": 0.0,
            "p_entry": p, "risk_pct": risk_pct,
            "hold_min": int((out.exit_ts - out.entry_ts).total_seconds() / 60),
            "predicted_hold": hold_min,
        }
    return _variant


def make_variant_B(*, max_notional_frac: float = 2.0):
    """V2 entry but live exits."""
    def _variant(bars, entry):
        return vol_trade_no_horizon_conviction(
            bars, entry, K=1.5, rr=2.0,
            risk_pct_base=0.005, risk_pct_max=0.010,
            max_notional_frac=max_notional_frac,
        )
    return _variant


# ---------- main ----------

def metrics(name: str, trades: list[dict]) -> dict:
    if not trades:
        return {"name": name, "n": 0}
    df = pd.DataFrame(trades)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    n = len(df)
    wins = int((df["pnl_bps_net"] > 0).sum())
    daily = df.groupby("et_date")["pnl_dollars"].sum()
    sharpe = daily.mean()/daily.std() if len(daily) > 1 and daily.std() > 0 else float("nan")
    cum = daily.cumsum()
    maxdd = float((cum - cum.cummax()).min())
    return {
        "name": name, "n": n, "win": wins/n,
        "total_bps_net": float(df["pnl_bps_net"].sum()),
        "total_dollars": float(df["pnl_dollars"].sum()),
        "avg_bps": float(df["pnl_bps_net"].mean()),
        "avg_qty": float(df["qty"].mean()),
        "daily_sharpe": float(sharpe),
        "max_dd_dollars": maxdd,
        "n_days": int(len(daily)),
    }


def print_table(rows):
    print(f"  {'setup':<46} {'n':>4} {'win%':>6} {'tot_$':>10} "
          f"{'tot_bps':>9} {'sharpe':>7} {'maxdd_$':>9}")
    print("  " + "-" * 95)
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['name']:<46} no trades")
            continue
        print(f"  {r['name']:<46} {r['n']:>4d} {r['win']*100:>5.1f}% "
              f"{r['total_dollars']:>+10.1f} {r['total_bps_net']:>+9.1f} "
              f"{r['daily_sharpe']:>7.2f} {r['max_dd_dollars']:>+9.1f}")
    print()


WINDOWS = [
    ("Feb 2026 (post-train OOS)",       "2026-02-01", "2026-03-01"),
    ("Mar 2026 (holdout, OOS)",         "2026-03-01", "2026-04-01"),
    ("Apr 2026 (live month, OOS)",      "2026-04-01", "2026-04-29"),
    ("Last 2 months (Mar+Apr 2026)",    "2026-03-01", "2026-04-29"),
    ("2 years (May 2024 - Apr 2026)",   "2024-04-29", "2026-04-29"),
]


def main():
    print("=" * 110)
    print(" V2 combined research model vs current live")
    print("=" * 110)
    print()
    v2_exits = load_v2_exits()
    print(f" v2 exit bundle train window: {v2_exits.train_window}")
    print(f"   stop RMSE  {v2_exits.metrics['stop_rmse']:.2f} bps")
    print(f"   target RMSE {v2_exits.metrics['target_rmse']:.2f} bps")
    print(f"   hold RMSE  {v2_exits.metrics['hold_rmse']:.1f} min")
    print()

    for label, start, end in WINDOWS:
        print("=" * 110)
        print(f" {label}: {start} → {end}")
        print("=" * 110)

        # Setup A — current live
        sigs_a, bars_a, _ = get_signals_features_bars(
            start, end, artifact_name="latest", cfg_name="v1")
        a = walk_with_blocking(sigs_a, bars_a, variant_A_live)

        # Setup B/C/D — using v2_entry signals
        sigs_v2, bars_v2, feats_v2 = get_signals_features_bars(
            start, end, artifact_name="research_v2_entry", cfg_name="v1")

        b = walk_with_blocking(sigs_v2, bars_v2, make_variant_B(max_notional_frac=2.0))
        c = walk_with_blocking(sigs_v2, bars_v2,
                                make_variant_C(v2_exits, feats_v2,
                                                 max_notional_frac=2.0,
                                                 tod_filter=False))
        d = walk_with_blocking(sigs_v2, bars_v2,
                                make_variant_C(v2_exits, feats_v2,
                                                 max_notional_frac=2.0,
                                                 tod_filter=True, tod_skip_lunch=False))
        e = walk_with_blocking(sigs_v2, bars_v2,
                                make_variant_C(v2_exits, feats_v2,
                                                 max_notional_frac=2.0,
                                                 tod_filter=True, tod_skip_lunch=True))

        rows = [
            metrics("A. CURRENT LIVE",                                    a),
            metrics("B. v2 entry only (live exits + sizing)",             b),
            metrics("C. v2 entry + v2 exits",                             c),
            metrics("D. v2 entry + v2 exits + TOD filter (skip 15:30+)",  d),
            metrics("E. v2 entry + v2 exits + TOD (skip 15:30 & lunch)",  e),
        ]
        print_table(rows)

        # Save trades
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        safe = f"{start}_{end}".replace("-", "")
        for tag, tr in [("v2_A", a), ("v2_B", b), ("v2_C", c), ("v2_D", d), ("v2_E", e)]:
            if tr:
                pd.DataFrame(tr).to_csv(out_dir / f"{tag}_{safe}.csv", index=False)


if __name__ == "__main__":
    main()
