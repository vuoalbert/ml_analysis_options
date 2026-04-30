"""Apples-to-apples: original model entries with two exit policies.

Same entries (p_up/p_dn ≥ 0.57, one-position-at-a-time, 15-min hold cap).
Two exit policies on top of the same entry list:

  • Baseline    — current live behaviour: 10% notional, exit at 15-min timeout
  • Vol-scaled  — stop = K × realized_vol(30m), target = 2 × stop,
                  risk-based qty (risk 0.5% of equity / stop_distance),
                  still capped at 15-min timeout

Backtest on:
  • March 2026  (the holdout month per artifacts/latest/meta.json)
  • April 2026  (the production live month)

Run:
    python -m research.backtest_compare
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
from features.build import build as build_features
from model.artifact import load as load_artifact
from research.backtest_original import build_frame
from research.vol_scaled_exits import (
    compute_exit_plan, simulate_one_trade,
    simulate_one_trade_no_horizon, simulate_one_trade_unlimited,
)


COST_BPS = 1.0
EQUITY = 30_000.0


def get_predictions_and_bars(start: str, end: str, artifact_name: str = "latest",
                              cfg_name: str = "v1"):
    """Build assembled frame, run model, return (signals_unblocked, bars).

    signals_unblocked: every minute where p_up≥thr or p_dn≥thr, in time order,
    with NO one-position-at-a-time blocking. Each variant applies its own
    blocking based on its actual hold time (the no-horizon variant naturally
    takes fewer trades because it holds longer).

    `artifact_name`: which artifact under artifacts/ to use for predictions.
    `cfg_name`: which configs/<name>.yaml — controls universe.symbol etc.
    """
    c = load_cfg(cfg_name)
    art = load_artifact(artifact_name)
    sym = c["universe"]["symbol"]
    thr_up = float(art.thresholds["up"])
    thr_dn = float(art.thresholds["down"])

    out = build_frame(start, end, c)
    if out.empty:
        return [], pd.DataFrame()

    feats = build_features(out, c)
    for col in art.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[art.feature_cols]
    essential = [col for col in art.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    feats = feats[feats[essential].notna().all(axis=1)]
    if feats.empty:
        return [], pd.DataFrame()

    proba = art.booster.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    pred["close"] = out.loc[pred.index, f"{sym.lower()}_close"]

    rth_mask = pred.index.to_series().apply(
        lambda t: 13 * 60 + 30 + c["risk"]["skip_first_minutes"]
                  <= t.hour * 60 + t.minute
                  < 20 * 60 - c["risk"]["skip_last_minutes"]
    )
    pred = pred[rth_mask.values]

    bars = out[[f"{sym.lower()}_open", f"{sym.lower()}_high", f"{sym.lower()}_low",
                 f"{sym.lower()}_close"]].rename(columns={
        f"{sym.lower()}_open": "open", f"{sym.lower()}_high": "high",
        f"{sym.lower()}_low": "low", f"{sym.lower()}_close": "close",
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
        signals.append({"ts": ts, "side": side, "p_up": float(row["p_up"]),
                         "p_dn": float(row["p_down"])})
    return signals, bars


def walk_with_blocking(signals: list, bars: pd.DataFrame, simulator) -> list[dict]:
    """Walk signals in time order; each variant's `simulator` returns the
    full trade dict (with exit_ts). The next signal must be at or after
    the previous trade's exit_ts (one-position-at-a-time)."""
    trades = []
    in_pos_until = None
    for s in signals:
        if in_pos_until is not None and s["ts"] < in_pos_until:
            continue
        trade = simulator(bars, s)
        if trade is None:
            continue
        trades.append(trade)
        in_pos_until = trade["exit_ts"]
    return trades


# -------- variants --------

def baseline_trade(bars, entry, horizon_min=15) -> dict | None:
    ts = entry["ts"]; side = entry["side"]
    if ts not in bars.index:
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars):
            return None
        ts = bars.index[idx]
    entry_idx = bars.index.get_loc(ts)
    entry_price = float(bars.iloc[entry_idx]["close"])
    qty = max(1, int((EQUITY * 0.10) // entry_price))
    end = min(entry_idx + horizon_min + 1, len(bars))
    window = bars.iloc[entry_idx + 1: end]
    if window.empty:
        return None
    exit_price = float(window.iloc[-1]["close"])
    sign = 1 if side == "long" else -1
    bps = sign * (exit_price / entry_price - 1.0) * 1e4
    pnl = sign * (exit_price - entry_price) * qty
    return {"entry_ts": ts, "exit_ts": window.index[-1], "side": side,
            "qty": qty, "entry_price": entry_price, "exit_price": exit_price,
            "pnl_dollars": pnl, "pnl_bps_gross": bps, "pnl_bps_net": bps - COST_BPS,
            "reason": "timeout",
            "hold_min": int((window.index[-1] - ts).total_seconds() / 60)}


def vol_trade_unlimited(bars, entry, K, risk_pct=0.005, rr=2.0) -> dict | None:
    """Vol-scaled stops/targets, NO horizon, NO end-of-day flat. Holds
    overnight / across days / weekends — exits only on stop, target, or
    end of backtest data."""
    ts = entry["ts"]; side = entry["side"]
    if ts not in bars.index:
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars):
            return None
        ts = bars.index[idx]
    entry_idx = bars.index.get_loc(ts)
    if entry_idx < 31:
        return None
    entry_price = float(bars.iloc[entry_idx]["close"])
    closes = bars.iloc[entry_idx - 30: entry_idx + 1]["close"].values
    plan = compute_exit_plan(bars_close=closes, entry_price=entry_price, equity=EQUITY,
                              side=side, K=K, rr_ratio=rr, risk_pct=risk_pct)
    if plan.qty <= 0:
        return None
    out = simulate_one_trade_unlimited(bars, ts, side, plan)
    if out is None:
        return None
    return {"entry_ts": out.entry_ts, "exit_ts": out.exit_ts, "side": out.side,
            "qty": out.qty, "entry_price": out.entry_price, "exit_price": out.exit_price,
            "pnl_dollars": out.pnl_dollars,
            "pnl_bps_gross": out.pnl_bps, "pnl_bps_net": out.pnl_bps - COST_BPS,
            "reason": out.reason, "stop_bps": out.stop_bps, "target_bps": out.target_bps,
            "rv_bps": out.rv_bps,
            "hold_min": int((out.exit_ts - out.entry_ts).total_seconds() / 60)}


def vol_trade_no_horizon_conviction(
    bars, entry, K,
    risk_pct_base: float = 0.005,
    risk_pct_max: float = 0.010,
    conviction_lo: float = 0.55,    # threshold (full risk_pct_base at this p)
    conviction_hi: float = 0.70,    # full risk_pct_max at this p or above
    rr: float = 2.0,
    max_notional_frac: float = 1.5,
) -> dict | None:
    """Vol-scaled exits + CONVICTION-WEIGHTED sizing.

    Risk-per-trade scales linearly with the entry probability:
        p = max(p_up, p_dn) at entry
        s = clip((p - conviction_lo) / (conviction_hi - conviction_lo), 0, 1)
        risk_pct = base + (max - base) × s

    Trades barely above threshold get smaller bets, high-conviction trades
    get up to 2× the base risk. Total deployed capital ≈ same; allocation
    is more concentrated on stronger signals.
    """
    ts = entry["ts"]; side = entry["side"]
    if ts not in bars.index:
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars):
            return None
        ts = bars.index[idx]
    entry_idx = bars.index.get_loc(ts)
    if entry_idx < 31:
        return None

    p = entry["p_up"] if side == "long" else entry["p_dn"]
    s = max(0.0, min(1.0, (p - conviction_lo) / (conviction_hi - conviction_lo)))
    risk_pct = risk_pct_base + (risk_pct_max - risk_pct_base) * s

    entry_price = float(bars.iloc[entry_idx]["close"])
    closes = bars.iloc[entry_idx - 30: entry_idx + 1]["close"].values
    plan = compute_exit_plan(
        bars_close=closes, entry_price=entry_price, equity=EQUITY,
        side=side, K=K, rr_ratio=rr, risk_pct=risk_pct,
        max_notional_frac=max_notional_frac,
    )
    if plan.qty <= 0:
        return None
    out = simulate_one_trade_no_horizon(bars, ts, side, plan)
    if out is None:
        return None
    return {
        "entry_ts": out.entry_ts, "exit_ts": out.exit_ts, "side": out.side,
        "qty": out.qty, "entry_price": out.entry_price,
        "exit_price": out.exit_price,
        "pnl_dollars": out.pnl_dollars,
        "pnl_bps_gross": out.pnl_bps,
        "pnl_bps_net": out.pnl_bps - COST_BPS,
        "reason": out.reason, "stop_bps": out.stop_bps,
        "target_bps": out.target_bps, "rv_bps": out.rv_bps,
        "p_entry": p, "risk_pct": risk_pct,
        "hold_min": int((out.exit_ts - out.entry_ts).total_seconds() / 60),
    }


def vol_trade_no_horizon(bars, entry, K, risk_pct=0.005, rr=2.0,
                           max_notional_frac=1.5) -> dict | None:
    """Vol-scaled stops/targets, NO 15-min horizon. Holds until stop or target
    fires, capped by end of trading day (5 min before close)."""
    ts = entry["ts"]; side = entry["side"]
    if ts not in bars.index:
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars):
            return None
        ts = bars.index[idx]
    entry_idx = bars.index.get_loc(ts)
    if entry_idx < 31:
        return None
    entry_price = float(bars.iloc[entry_idx]["close"])
    closes = bars.iloc[entry_idx - 30: entry_idx + 1]["close"].values
    plan = compute_exit_plan(bars_close=closes, entry_price=entry_price, equity=EQUITY,
                              side=side, K=K, rr_ratio=rr, risk_pct=risk_pct,
                              max_notional_frac=max_notional_frac)
    if plan.qty <= 0:
        return None
    out = simulate_one_trade_no_horizon(bars, ts, side, plan)
    if out is None:
        return None
    return {"entry_ts": out.entry_ts, "exit_ts": out.exit_ts, "side": out.side,
            "qty": out.qty, "entry_price": out.entry_price, "exit_price": out.exit_price,
            "pnl_dollars": out.pnl_dollars,
            "pnl_bps_gross": out.pnl_bps, "pnl_bps_net": out.pnl_bps - COST_BPS,
            "reason": out.reason, "stop_bps": out.stop_bps, "target_bps": out.target_bps,
            "rv_bps": out.rv_bps,
            "hold_min": int((out.exit_ts - out.entry_ts).total_seconds() / 60)}


def vol_trade(bars, entry, K, horizon_min=15, risk_pct=0.005, rr=2.0) -> dict | None:
    ts = entry["ts"]; side = entry["side"]
    if ts not in bars.index:
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars):
            return None
        ts = bars.index[idx]
    entry_idx = bars.index.get_loc(ts)
    if entry_idx < 31:
        return None
    entry_price = float(bars.iloc[entry_idx]["close"])
    closes = bars.iloc[entry_idx - 30: entry_idx + 1]["close"].values
    plan = compute_exit_plan(bars_close=closes, entry_price=entry_price, equity=EQUITY,
                              side=side, K=K, rr_ratio=rr, risk_pct=risk_pct)
    if plan.qty <= 0:
        return None
    out = simulate_one_trade(bars, ts, side, plan, horizon_min=horizon_min)
    if out is None:
        return None
    return {"entry_ts": out.entry_ts, "exit_ts": out.exit_ts, "side": out.side,
            "qty": out.qty, "entry_price": out.entry_price, "exit_price": out.exit_price,
            "pnl_dollars": out.pnl_dollars,
            "pnl_bps_gross": out.pnl_bps, "pnl_bps_net": out.pnl_bps - COST_BPS,
            "reason": out.reason, "stop_bps": out.stop_bps, "target_bps": out.target_bps,
            "rv_bps": out.rv_bps,
            "hold_min": int((out.exit_ts - out.entry_ts).total_seconds() / 60)}


# -------- summarise + print --------

def summarize(name: str, trades: list[dict]) -> dict:
    if not trades:
        return {"name": name, "n": 0}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = int((df["pnl_bps_net"] > 0).sum())
    df["et_date"] = df["entry_ts"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("et_date")["pnl_bps_net"].sum()
    daily_sharpe = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else float("nan")
    cumbps = df["pnl_bps_net"].cumsum()
    max_dd = float((cumbps - cumbps.cummax()).min())
    rec = {
        "name": name, "n": n,
        "win": wins / n,
        "total_bps_net": float(df["pnl_bps_net"].sum()),
        "total_dollars": float(df["pnl_dollars"].sum()),
        "avg_bps": float(df["pnl_bps_net"].mean()),
        "avg_qty": float(df["qty"].mean()),
        "avg_hold": float(df["hold_min"].mean()),
        "daily_sharpe": daily_sharpe,
        "max_dd_bps": max_dd,
    }
    if "stop_bps" in df.columns:
        rec["pct_stop"] = float((df["reason"] == "stop").mean() * 100)
        rec["pct_tgt"] = float((df["reason"] == "target").mean() * 100)
        rec["pct_to"] = float((df["reason"] == "timeout").mean() * 100)
        rec["pct_eod"] = float((df["reason"] == "eod_flat").mean() * 100)
        rec["pct_end"] = float((df["reason"] == "data_end").mean() * 100)
        rec["avg_stop_bps"] = float(df["stop_bps"].mean())
        rec["avg_tgt_bps"] = float(df["target_bps"].mean())
    return rec


def print_table(rows: list[dict]):
    print(f"  {'variant':<40} {'n':>4} {'win%':>6} {'tot_bps':>9} {'tot_$':>10} "
          f"{'avg_bps':>8} {'avg_qty':>8} {'hold':>7} {'sharpe':>7} {'maxdd':>8}")
    print("  " + "-" * 110)
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['name']:<40} no trades")
            continue
        print(f"  {r['name']:<40} {r['n']:>4d} {r['win']*100:>5.1f}% "
              f"{r['total_bps_net']:>+9.1f} {r['total_dollars']:>+10.1f} "
              f"{r['avg_bps']:>+8.2f} {r['avg_qty']:>8.0f} "
              f"{r['avg_hold']:>6.1f}m {r['daily_sharpe']:>7.2f} {r['max_dd_bps']:>+8.1f}")
    print()
    # Vol-scaled-only extras
    extras = [r for r in rows if r["n"] > 0 and "pct_stop" in r]
    if extras:
        print(f"  {'variant':<40} {'%stop':>6} {'%tgt':>6} {'%to':>6} {'%eod':>6} {'%end':>6} "
              f"{'avg_stop':>10} {'avg_tgt':>10}")
        print("  " + "-" * 96)
        for r in extras:
            print(f"  {r['name']:<40} {r['pct_stop']:>5.0f}% {r['pct_tgt']:>5.0f}% "
                  f"{r['pct_to']:>5.0f}% {r.get('pct_eod', 0):>5.0f}% {r.get('pct_end', 0):>5.0f}% "
                  f"{r['avg_stop_bps']:>9.1f}b {r['avg_tgt_bps']:>9.1f}b")
        print()


# -------- main --------

def run_window(label: str, start: str, end: str, Ks: list[float],
                include_no_horizon: bool = True, no_horizon_K: float = 1.5,
                include_unlimited: bool = False, unlimited_K: float = 1.5,
                artifact_name: str = "latest"):
    print("=" * 100)
    print(f" {label}: {start} → {end}    [artifact={artifact_name}]")
    print("=" * 100)
    print("Building frame + predictions…")
    signals, bars = get_predictions_and_bars(start, end, artifact_name=artifact_name)
    if not signals:
        print("  no signals fired (model conviction below threshold).")
        return
    longs = sum(1 for s in signals if s["side"] == "long")
    shorts = sum(1 for s in signals if s["side"] == "short")
    print(f"  raw signals (p≥0.57): {len(signals)}  ({longs} long, {shorts} short)")
    print(f"  Each variant blocks subsequent signals until its trade exits.")
    print()

    rows = []
    base = walk_with_blocking(signals, bars, baseline_trade)
    rows.append(summarize("Baseline (15m, 10% notional)", base))
    for K in Ks:
        v = walk_with_blocking(signals, bars,
                                lambda b, e, K=K: vol_trade(b, e, K))
        rows.append(summarize(f"Vol-scaled K={K} (15m cap)", v))
    if include_no_horizon:
        nh = walk_with_blocking(signals, bars,
                                 lambda b, e: vol_trade_no_horizon(b, e, no_horizon_K))
        rows.append(summarize(f"Vol-scaled K={no_horizon_K} (no horizon, eod_flat)", nh))
    if include_unlimited:
        un = walk_with_blocking(signals, bars,
                                 lambda b, e: vol_trade_unlimited(b, e, unlimited_K))
        rows.append(summarize(f"Vol-scaled K={unlimited_K} (UNLIMITED hold)", un))

    print_table(rows)

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    safe = f"{start}_{end}".replace("-", "")
    if base:
        pd.DataFrame(base).to_csv(out_dir / f"compare_{safe}_baseline.csv", index=False)
    for K in Ks:
        v = walk_with_blocking(signals, bars,
                                lambda b, e, K=K: vol_trade(b, e, K))
        if v:
            pd.DataFrame(v).to_csv(out_dir / f"compare_{safe}_volK{K}.csv", index=False)
    if include_no_horizon:
        nh = walk_with_blocking(signals, bars,
                                 lambda b, e: vol_trade_no_horizon(b, e, no_horizon_K))
        if nh:
            pd.DataFrame(nh).to_csv(out_dir / f"compare_{safe}_volK{no_horizon_K}_noh.csv", index=False)
    if include_unlimited:
        un = walk_with_blocking(signals, bars,
                                 lambda b, e: vol_trade_unlimited(b, e, unlimited_K))
        if un:
            pd.DataFrame(un).to_csv(out_dir / f"compare_{safe}_volK{unlimited_K}_unlimited.csv", index=False)
    print(f"  per-trade CSVs in {out_dir}/")
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="single window start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="single window end")
    ap.add_argument("--label", default="WINDOW")
    ap.add_argument("--Ks", default="0.5,0.75,1.0,1.5",
                    help="comma-separated K values to test")
    args = ap.parse_args()
    Ks = [float(x) for x in args.Ks.split(",")]

    if args.start and args.end:
        run_window(args.label, args.start, args.end, Ks)
    else:
        run_window("HOLDOUT MONTH", "2026-03-01", "2026-04-01", Ks)
        run_window("PRODUCTION LIVE MONTH", "2026-04-01", "2026-04-29", Ks)
    print("=" * 100)
    print(" Caveats")
    print("=" * 100)
    print("""
  Same model + same entries throughout. Differences are pure exit logic + sizing.
  • Baseline is the current live policy (10% notional, no stop/target, 15-min hold).
  • Vol-scaled stop = K × realized_vol(30m), target = 2 × stop (1:2 R/R),
    qty = floor(equity × 0.5% / stop_distance_dollars), hold capped at 15 min.
  • bps numbers are size-independent and the cleanest comparison.
  • $ numbers depend on qty; vol-scaled tends to size much larger when stops
    are tight, so a winning vol-scaled trade prints big $ but a losing one
    also bleeds big $. Watch buying power before deploying.
  • Costs: 1.0 bp/trade (matches v1.yaml: 0.5 half-spread + 0.5 slippage).
  • Bars are IEX (free feed); model trained on same. Live execution may differ.
""")


if __name__ == "__main__":
    main()
