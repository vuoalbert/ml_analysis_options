"""Backtest the original ml_analysis model on a chosen historical window.

Replays the actual artifacts/latest model with the live exit policy
(10% notional, 15-min horizon, p ≥ 0.57). Goal: see if the holdout/walk-forward
numbers in meta.json hold up when we re-simulate them ourselves.

Usage:
    python -m research.backtest_original --start 2026-03-01 --end 2026-04-01
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from utils.config import load as load_cfg
from utils.calendar import NYSE, is_fomc_day, is_zero_dte, minutes_into_session
from features.build import build as build_features
from model.artifact import load as load_artifact
from data_pull import bars as bars_pull, yf_daily, fred
from data_pull.assemble import YF_DAILY_CROSS, ALPACA_CROSS, _daily_ffill, _align_minute


COST_BPS = 1.0   # matches v1.yaml: 0.5 half-spread + 0.5 slippage
EQUITY = 30_000.0


def build_frame(start: str, end: str, c: dict) -> pd.DataFrame:
    """Pull SPY + cross-asset minute bars + daily macro and assemble the frame
    the model expects. Mirrors the training pipeline (data_pull.assemble) but
    parameterised by the test window."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")

    sched = NYSE.schedule(start_date=s.date(), end_date=e.date())
    if sched.empty:
        return pd.DataFrame()
    import pandas_market_calendars as mcal
    minute_idx = mcal.date_range(sched, frequency="1min").tz_convert("UTC")

    sym = c["universe"]["symbol"]
    spy = bars_pull.pull(sym, start, end, use_cache=True)
    cols = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
    out = _align_minute(spy, minute_idx, cols, sym.lower())

    for s_sym in c["universe"]["cross_asset"]:
        if s_sym in ALPACA_CROSS:
            df = bars_pull.pull(s_sym, start, end, use_cache=True)
            if not df.empty:
                out = out.join(_align_minute(df, minute_idx, ["close", "volume"], s_sym.lower()))

    yf_start = (s - pd.Timedelta(days=120)).date().isoformat()
    yf_end = e.date().isoformat()
    for s_sym, prefix in YF_DAILY_CROSS.items():
        if s_sym in c["universe"]["cross_asset"]:
            try:
                df = yf_daily.pull(s_sym, yf_start, yf_end, use_cache=True)
                if not df.empty and "close" in df.columns:
                    ff = _daily_ffill(df[["close"]].rename(columns={"close": f"{prefix}_close"}),
                                      minute_idx, lag_days=0)
                    out = out.join(ff)
            except Exception:
                pass

    fred_df = fred.pull_many(c.get("fred_series", []), yf_start, yf_end)
    if not fred_df.empty:
        ff = _daily_ffill(fred_df, minute_idx, lag_days=1)
        ff.columns = [f"fred_{col}" for col in ff.columns]
        out = out.join(ff)

    out["evt_fomc_day"] = is_fomc_day(minute_idx).astype(int).values
    out["evt_zero_dte"] = is_zero_dte(minute_idx).astype(int).values
    out["session_min"] = minutes_into_session(minute_idx).values
    return out.dropna(subset=[f"{sym.lower()}_close"])


def run_backtest(start: str, end: str):
    print("=" * 92)
    print(f" Backtest: ml_analysis original model on {start} → {end}")
    print("=" * 92)
    print(f" Live policy: 10% notional, 15-min horizon, p_up/p_dn ≥ 0.57, costs {COST_BPS} bps/trade")
    print()

    c = load_cfg("v1")
    art = load_artifact("latest")
    thr_up = float(art.thresholds["up"])
    thr_dn = float(art.thresholds["down"])
    horizon = int(c["label"]["horizon_min"])
    sym = c["universe"]["symbol"]

    print("Building assembled frame…")
    out = build_frame(start, end, c)
    if out.empty:
        print("  no data, abort.")
        return
    print(f"  {len(out)} rows, {out.index.normalize().nunique()} trading days")

    print("Computing features + predictions…")
    feats = build_features(out, c)
    for col in art.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[art.feature_cols]
    essential = [col for col in art.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    feats = feats[feats[essential].notna().all(axis=1)]
    if feats.empty:
        print("  no usable feature rows.")
        return

    proba = art.booster.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    pred["close"] = out.loc[pred.index, f"{sym.lower()}_close"]
    print(f"  {len(pred)} prediction rows")
    print(f"  p_up:  max={pred['p_up'].max():.3f}  ≥0.57 rows={int((pred['p_up']>=thr_up).sum())}")
    print(f"  p_dn:  max={pred['p_down'].max():.3f}  ≥0.57 rows={int((pred['p_down']>=thr_dn).sum())}")
    print()

    # Filter to RTH + skip first/last N min (live policy)
    rth_mask = pred.index.to_series().apply(
        lambda t: 13 * 60 + 30 + c["risk"]["skip_first_minutes"]
                  <= t.hour * 60 + t.minute
                  < 20 * 60 - c["risk"]["skip_last_minutes"]
    )
    pred = pred[rth_mask.values]

    # SPY bars (high/low not needed for horizon-only baseline, but kept for symmetry)
    bars = out[[f"{sym.lower()}_open", f"{sym.lower()}_high", f"{sym.lower()}_low",
                 f"{sym.lower()}_close"]].rename(columns={
        f"{sym.lower()}_open": "open", f"{sym.lower()}_high": "high",
        f"{sym.lower()}_low": "low", f"{sym.lower()}_close": "close",
    }).dropna()

    # Walk forward enforcing 1-position-at-a-time + 15-min hold
    print("Walking forward…")
    trades = []
    blocked_until: pd.Timestamp | None = None
    for ts, row in pred.iterrows():
        if blocked_until is not None and ts < blocked_until:
            continue
        side = None
        if row["p_up"] >= thr_up:
            side = "long"
        elif row["p_down"] >= thr_dn:
            side = "short"
        if side is None:
            continue

        # Find entry bar in the bars frame
        if ts not in bars.index:
            idx = bars.index.searchsorted(ts)
            if idx >= len(bars):
                continue
            ts_aligned = bars.index[idx]
        else:
            ts_aligned = ts
        entry_idx = bars.index.get_loc(ts_aligned)
        entry_price = float(bars.iloc[entry_idx]["close"])

        # Exit at horizon timeout (live policy — no stops/targets)
        end_idx = min(entry_idx + horizon + 1, len(bars))
        window = bars.iloc[entry_idx + 1: end_idx]
        if window.empty:
            continue
        exit_price = float(window.iloc[-1]["close"])
        exit_ts = window.index[-1]

        sign = 1 if side == "long" else -1
        bps_gross = sign * (exit_price / entry_price - 1.0) * 1e4
        bps_net = bps_gross - COST_BPS
        qty = max(1, int((EQUITY * c["risk"]["max_position_notional_frac"]) // entry_price))
        pnl_d = sign * (exit_price - entry_price) * qty

        trades.append({
            "entry_ts": ts_aligned, "exit_ts": exit_ts, "side": side,
            "entry_price": entry_price, "exit_price": exit_price, "qty": qty,
            "pnl_dollars": pnl_d, "pnl_bps_gross": bps_gross, "pnl_bps_net": bps_net,
            "p_up": float(row["p_up"]), "p_dn": float(row["p_down"]),
            "hold_min": int((exit_ts - ts_aligned).total_seconds() / 60),
        })
        blocked_until = ts_aligned + pd.Timedelta(minutes=horizon)

    if not trades:
        print("  no trades fired.")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = int((df["pnl_bps_net"] > 0).sum())
    longs = int((df["side"] == "long").sum())
    shorts = int((df["side"] == "short").sum())
    total_bps = float(df["pnl_bps_net"].sum())
    total_dollars = float(df["pnl_dollars"].sum())
    avg_bps = float(df["pnl_bps_net"].mean())

    # Daily Sharpe (intraday daily — what meta.json reports)
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("et_date")["pnl_bps_net"].sum()
    if len(daily) > 1 and daily.std() > 0:
        daily_sharpe = float(daily.mean() / daily.std())
    else:
        daily_sharpe = float("nan")

    # Drawdown on cumulative bps
    cumbps = df["pnl_bps_net"].cumsum()
    max_dd_bps = float((cumbps - cumbps.cummax()).min())

    print()
    print("=" * 92)
    print(" Results")
    print("=" * 92)
    print(f"  trades             : {n}  ({longs} long, {shorts} short)")
    print(f"  hit rate           : {wins/n:.3f}  ({wins} winners)")
    print(f"  total P&L (bps net): {total_bps:+.1f}")
    print(f"  total P&L ($, 10%) : {total_dollars:+.2f} on $30k starting equity")
    print(f"  avg trade (bps net): {avg_bps:+.2f}")
    print(f"  active days        : {len(daily)}")
    print(f"  daily Sharpe       : {daily_sharpe:.2f}")
    print(f"  max drawdown (bps) : {max_dd_bps:+.1f}")
    print()

    # Compare to meta.json holdout if window matches
    print("Reference (from artifacts/latest/meta.json):")
    print(f"  holdout (Mar 2026) : 130 trades  57.7% hit  +210.7 bps net  Sharpe 2.18")
    print(f"  walk-forward avg   : 113 trades/fold  59.0% hit  +494 bps/fold  Sharpe 7.79")
    print()

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    safe = f"{start}_{end}".replace("-", "")
    df.to_csv(out_dir / f"original_model_{safe}.csv", index=False)
    print(f"Per-trade CSV: {out_dir}/original_model_{safe}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-04-01")
    args = ap.parse_args()
    run_backtest(args.start, args.end)


if __name__ == "__main__":
    main()
