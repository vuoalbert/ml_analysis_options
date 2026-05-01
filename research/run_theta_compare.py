"""Compare theta_protect variants for live behavior alignment.

Tests four configurations on the same OOS window:
  1. baseline_no_theta:    theta_protect_any_dte=False (= what backtest validated)
  2. live_replicate:       theta_protect_any_dte=True, mins=60, thresh=0.10 (= what live bot does)
  3. opt_a_disable:        theta_protect_mins=0, any_dte=True (= turn it off)
  4. opt_b_tight:          mins=15, any_dte=True (= shorter window)
  5. opt_c_breakeven:      mins=60, thresh=0.0, any_dte=True (= only kill losing trades)

All use Phase 6 winner config: ITM 2.5%, target 30%, stop 50%, DTE=1, multi=15, risk 2%.

Run:
    python -m research.run_theta_compare
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

# Cache-only fetcher — local Alpaca keys may be stale; we want to use only
# bars already cached from prior backtests.
import data_pull.bars as _bars
_orig_pull = _bars.pull
def _cache_only_pull(symbol, start, end, use_cache=True):
    from data_pull import cache as _cache
    name = f"bars_{symbol.replace('=', '').replace('^', '')}_{start}_{end}"
    cached = _cache.load(name)
    if cached is not None:
        return cached
    # If not cached, return empty rather than hitting Alpaca with stale keys
    import pandas as _pd
    return _pd.DataFrame()
_bars.pull = _cache_only_pull

from utils.config import load as load_cfg
from research.backtest_original import build_frame
from research.v3_combinations import V3Bundle
from research.feature_extensions import add_extensions
from features.build import build as build_features
from research.option_simulator import OptionsSimConfig, make_options_simulator
from research.multi_walker import walk_multi


# OOS window — recent 12 months (most relevant to current live behavior)
MONTHS = []
for y, m_lo, m_hi in [(2025, 5, 12), (2026, 1, 4)]:
    for m in range(m_lo, m_hi + 1):
        start = pd.Timestamp(year=y, month=m, day=1)
        end = (start + pd.offsets.MonthEnd(1))
        MONTHS.append((start.strftime("%Y-%m"),
                       start.strftime("%Y-%m-%d"),
                       end.strftime("%Y-%m-%d")))

MAX_CONCURRENT = 15
BASE_CONFIG = {
    "dte": 1, "conviction_min": 0.55,
    "risk_pct": 0.02, "max_qty": 100,
    "use_iv_dynamics": True, "iv_beta_call": 5.0, "iv_beta_put": 8.0,
    "moneyness": "itm", "itm_offset_pct": 0.025,
    "target_pct": 0.30, "stop_pct": 0.50,
}

# Theta variants
VARIANTS = {
    "1_baseline_no_theta": {"theta_protect_any_dte": False, "theta_protect_mins": 0,
                             "theta_protect_profit_thresh": 0.10},
    "2_live_replicate": {"theta_protect_any_dte": True, "theta_protect_mins": 60,
                          "theta_protect_profit_thresh": 0.10},
    "3_opt_a_disable": {"theta_protect_any_dte": True, "theta_protect_mins": 0,
                         "theta_protect_profit_thresh": 0.10},
    "4_opt_b_tight": {"theta_protect_any_dte": True, "theta_protect_mins": 15,
                       "theta_protect_profit_thresh": 0.10},
    "5_opt_c_breakeven": {"theta_protect_any_dte": True, "theta_protect_mins": 60,
                           "theta_protect_profit_thresh": 0.0},
}


def _vix_from_cache(end_str):
    from data_pull import cache as _cache
    end_ts = pd.Timestamp(end_str, tz="UTC")
    for vp in sorted(_cache.cache_dir().glob("yf_VIX_*.parquet"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        df = pd.read_parquet(vp)
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = df.index.astype("datetime64[ns, UTC]")
        if not df.empty and df.index.max() >= end_ts - pd.Timedelta(days=10):
            return (df["close"] if "close" in df.columns else df.iloc[:, 0]).rename("vix_close")
    return pd.Series(dtype=float)


def get_signals_and_bars(start, end, bundle):
    cfg = load_cfg("v1")
    cfg = dict(cfg)
    cfg["universe"] = dict(cfg["universe"])
    sym = "spy"

    out = build_frame(start, end, cfg)
    if out.empty:
        return [], pd.DataFrame()

    feats = build_features(out, cfg)
    feats = add_extensions(feats, out, sym=sym, add_volume=False, add_mtf=True)
    new_cols = [c for c in feats.columns if c.startswith(("vp_", "mtf_"))]
    if new_cols:
        feats[new_cols] = feats[new_cols].ffill().fillna(0.0)

    for col in bundle.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[bundle.feature_cols]
    essential = [col for col in bundle.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    feats = feats[feats[essential].notna().all(axis=1)]
    if feats.empty:
        return [], pd.DataFrame()

    proba = bundle.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    minutes_of_day = pred.index.hour * 60 + pred.index.minute
    rth_lo = 13 * 60 + 30 + cfg["risk"]["skip_first_minutes"]
    rth_hi = 20 * 60 - cfg["risk"]["skip_last_minutes"]
    pred = pred[(minutes_of_day >= rth_lo) & (minutes_of_day < rth_hi)]

    bars = out[[f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close"]].rename(columns={
        f"{sym}_open": "open", f"{sym}_high": "high",
        f"{sym}_low": "low", f"{sym}_close": "close",
    }).dropna()

    thr_up = float(bundle.thresholds["up"])
    thr_dn = float(bundle.thresholds["down"])
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
    return signals, bars


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
    reasons = df["reason"].value_counts().to_dict() if "reason" in df.columns else {}
    return {
        "n": len(df),
        "win_pct": float((df["pnl_dollars"] > 0).mean() * 100),
        "total_$": float(df["pnl_dollars"].sum()),
        "avg_$": float(df["pnl_dollars"].mean()),
        "sharpe": float(sharpe),
        "max_dd_$": float(dd.min() if len(dd) else 0.0),
        "days": int(df["et_date"].nunique()),
        "target_n": int(reasons.get("target", 0)),
        "stop_n": int(reasons.get("stop", 0)),
        "theta_n": int(reasons.get("theta_protect", 0)),
        "eod_n": int(reasons.get("eod_flat", 0)),
        "data_end_n": int(reasons.get("data_end", 0)),
    }


def run_variant(name, theta_kwargs):
    print(f"\n{'='*100}")
    print(f" VARIANT: {name}  ({theta_kwargs})")
    print(f"{'='*100}")
    bundle = V3Bundle("mtf")
    full_kwargs = {**BASE_CONFIG, **theta_kwargs}
    monthly = []
    for label, start, end in MONTHS:
        t0 = time.time()
        sig, bars = get_signals_and_bars(start, end, bundle)
        vix = _vix_from_cache(end)
        if not sig or bars.empty:
            print(f"[{label}] no data")
            continue
        cfg = OptionsSimConfig(vix_series=vix, **full_kwargs)
        sim = make_options_simulator(cfg)
        trades = walk_multi(sig, bars, sim, max_concurrent=MAX_CONCURRENT)
        m = summarise(trades) or {}
        m["month"] = label
        monthly.append(m)
        if m:
            print(f"[{label}] n={m['n']:>4} win={m['win_pct']:>5.1f}% "
                  f"$={m['total_$']:>+9,.0f}  "
                  f"target={m['target_n']:>3} stop={m['stop_n']:>3} "
                  f"theta={m['theta_n']:>3} eod={m['eod_n']:>3}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return monthly


def main():
    print(f"Comparing {len(VARIANTS)} theta_protect variants over {len(MONTHS)} months")
    all_results = {}
    for name, kw in VARIANTS.items():
        all_results[name] = run_variant(name, kw)

    print()
    print("=" * 110)
    print(" SUMMARY — by variant (totals across all months)")
    print("=" * 110)
    print(f" {'variant':<22}{'n':>6}{'win%':>7}{'total $':>12}{'avg $':>9}{'target':>8}{'stop':>6}"
          f"{'theta':>7}{'eod':>5}{'data_end':>10}{'sharpe':>8}")
    print(" " + "-" * 100)
    for name, monthly in all_results.items():
        if not monthly:
            continue
        total_n = sum(m.get("n", 0) for m in monthly)
        total_pnl = sum(m.get("total_$", 0) for m in monthly)
        total_target = sum(m.get("target_n", 0) for m in monthly)
        total_stop = sum(m.get("stop_n", 0) for m in monthly)
        total_theta = sum(m.get("theta_n", 0) for m in monthly)
        total_eod = sum(m.get("eod_n", 0) for m in monthly)
        total_data_end = sum(m.get("data_end_n", 0) for m in monthly)
        win_n = sum(m.get("n", 0) * m.get("win_pct", 0) / 100 for m in monthly)
        win_pct = (win_n / total_n * 100) if total_n else 0
        avg = (total_pnl / total_n) if total_n else 0
        sharpes = [m["sharpe"] for m in monthly if "sharpe" in m]
        sharpe = float(np.mean(sharpes)) if sharpes else 0
        print(f" {name:<22}{total_n:>6}{win_pct:>6.1f}%{total_pnl:>+12,.0f}{avg:>+9.1f}"
              f"{total_target:>8}{total_stop:>6}{total_theta:>7}{total_eod:>5}{total_data_end:>10}"
              f"{sharpe:>+8.2f}")

    out_path = ROOT / "research" / "outputs" / "theta_compare.csv"
    out_path.parent.mkdir(exist_ok=True)
    flat = []
    for name, monthly in all_results.items():
        for m in monthly:
            m["variant"] = name
            flat.append(m)
    pd.DataFrame(flat).to_csv(out_path, index=False)
    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
