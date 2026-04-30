"""Vol-scaled exit prototype (Approach 3) with risk-based sizing.

Replaces the live loop's two assumptions:
  1. Fixed 15-min horizon-only exit (no stop, no target)
  2. Fixed 10% notional per trade — same dollar exposure regardless of vol

…with:
  1. stop_bps = clip(K × realized_vol_bps(lookback=30), 5, 100)
     target_bps = 2 × stop_bps              (1:2 R/R, the canonical default)
     hold_min = 15                          (unchanged — horizon model trained on this)
  2. qty = floor((equity × risk_pct) / stop_distance_dollars)
     where stop_distance_dollars = entry_price × stop_bps / 10000

Why vol-scaled stops are defensible:
  • Tight stops on calm tape don't fire prematurely on noise; loose stops on
    volatile tape don't get gimme-stopped.
  • Risk-based sizing keeps dollar-risk-per-trade constant across regimes —
    the existing 10% notional rule risks 5x more on a high-vol day vs low-vol.

K is the only tunable parameter. Tune via grid search on training data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------- realized vol estimator ----------

def realized_vol_bps(closes: np.ndarray | pd.Series, lookback_min: int = 30) -> float:
    """Std of trailing 1-min log returns, expressed in bps over the lookback.

    `closes` should be the most recent `lookback_min + 1` closes; the function
    only uses the last (lookback_min + 1) values regardless. Returns 0.0 on
    insufficient data.
    """
    arr = np.asarray(closes, dtype=float)
    if arr.size < 2:
        return 0.0
    arr = arr[-(lookback_min + 1):]
    log_rets = np.diff(np.log(arr))
    if log_rets.size == 0:
        return 0.0
    sigma = float(np.std(log_rets, ddof=0))
    # Convert per-minute vol to bps over the lookback horizon.
    return sigma * np.sqrt(lookback_min) * 10_000.0


# ---------- stop / target / size ----------

@dataclass
class ExitPlan:
    stop_bps: float
    target_bps: float
    qty: int
    risk_dollars: float                 # what we're risking on this trade
    rv_bps: float                       # observed trailing vol (for logging)
    note: str


def compute_exit_plan(
    *,
    bars_close: np.ndarray | pd.Series,
    entry_price: float,
    equity: float,
    side: str,                          # "long" or "short" — only affects `note`
    K: float = 1.0,
    rr_ratio: float = 2.0,
    risk_pct: float = 0.005,            # 0.5% of equity at risk per trade
    lookback_min: int = 30,
    min_stop_bps: float = 5.0,
    max_stop_bps: float = 100.0,
    min_qty: int = 1,
    max_notional_frac: float = 1.5,     # cap qty so notional ≤ 1.5× equity
) -> ExitPlan:
    """Compute stop, target, and quantity for a fresh entry.

    Bounds:
      • stop_bps clipped to [min_stop_bps, max_stop_bps] — below min is
        half-spread noise, above max is bigger than any realistic horizon move
      • qty initially derived from risk budget (risk_pct × equity / stop_distance),
        then capped at max_notional_frac × equity / entry_price. Tight stops on
        low-vol tape would otherwise produce 4-5× leverage; the cap keeps the
        position size within Alpaca paper buying-power limits.
      • qty floored at `min_qty` so a fired signal always takes at least 1 share.
    """
    rv = realized_vol_bps(bars_close, lookback_min=lookback_min)
    raw_stop = K * rv
    stop_bps = float(np.clip(raw_stop, min_stop_bps, max_stop_bps))
    target_bps = stop_bps * rr_ratio

    risk_dollars = equity * risk_pct
    stop_distance_dollars = entry_price * (stop_bps / 10_000.0)
    if stop_distance_dollars <= 0:
        qty = 0
    else:
        qty_risk = int(risk_dollars // stop_distance_dollars)
        qty_cap = int((equity * max_notional_frac) // max(entry_price, 1e-9))
        qty = max(min_qty, min(qty_risk, qty_cap))

    capped = qty < (int(risk_dollars // stop_distance_dollars) if stop_distance_dollars > 0 else 0)
    cap_marker = " [CAPPED]" if capped else ""
    note = (f"{side} K={K} rv={rv:.1f}bps stop={stop_bps:.1f}bps "
            f"target={target_bps:.1f}bps qty={qty} "
            f"risk=${risk_dollars:.0f}{cap_marker}")
    return ExitPlan(
        stop_bps=stop_bps, target_bps=target_bps, qty=qty,
        risk_dollars=risk_dollars, rv_bps=rv, note=note,
    )


# ---------- in-loop exit check (per-bar) ----------

def should_exit(
    *,
    side: str,
    entry_price: float,
    last_high: float,
    last_low: float,
    stop_bps: float,
    target_bps: float,
) -> Optional[str]:
    """Per-bar barrier check — call this every minute while a position is open.

    Returns "stop", "target", or None. The live loop already exits on horizon
    timeout; this just adds the two price-driven barriers on top.
    Conservative tie-break: if both barriers fire on the same bar, return "stop".
    """
    sp = stop_bps / 10_000.0
    tp = target_bps / 10_000.0
    if side == "long":
        stop_px = entry_price * (1.0 - sp)
        target_px = entry_price * (1.0 + tp)
        hit_stop = last_low <= stop_px
        hit_target = last_high >= target_px
    else:
        stop_px = entry_price * (1.0 + sp)
        target_px = entry_price * (1.0 - tp)
        hit_stop = last_high >= stop_px
        hit_target = last_low <= target_px
    if hit_stop:
        return "stop"
    if hit_target:
        return "target"
    return None


# ---------- mini backtest harness ----------

@dataclass
class BacktestRow:
    entry_ts: pd.Timestamp
    side: str
    qty: int
    entry_price: float
    exit_price: float
    exit_ts: pd.Timestamp
    reason: str
    pnl_dollars: float
    pnl_bps: float
    stop_bps: float
    target_bps: float
    rv_bps: float


def simulate_one_trade(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    side: str,
    plan: ExitPlan,
    horizon_min: int = 15,
) -> Optional[BacktestRow]:
    """Walk forward from entry_ts checking stop/target each bar, timeout
    after horizon_min."""
    if entry_ts not in bars.index:
        idx = bars.index.searchsorted(entry_ts)
        if idx >= len(bars):
            return None
        entry_ts = bars.index[idx]
    entry_idx = bars.index.get_loc(entry_ts)
    entry_price = float(bars.iloc[entry_idx]["close"])
    end = min(entry_idx + horizon_min + 1, len(bars))
    window = bars.iloc[entry_idx + 1: end]
    if window.empty:
        return None

    sign = 1.0 if side == "long" else -1.0
    sp = plan.stop_bps / 10_000.0
    tp = plan.target_bps / 10_000.0

    for ts, bar in window.iterrows():
        reason = should_exit(
            side=side, entry_price=entry_price,
            last_high=float(bar["high"]), last_low=float(bar["low"]),
            stop_bps=plan.stop_bps, target_bps=plan.target_bps,
        )
        if reason == "stop":
            exit_px = entry_price * (1 - sp) if side == "long" else entry_price * (1 + sp)
            return _to_row(entry_ts, ts, side, plan, entry_price, exit_px, "stop")
        if reason == "target":
            exit_px = entry_price * (1 + tp) if side == "long" else entry_price * (1 - tp)
            return _to_row(entry_ts, ts, side, plan, entry_price, exit_px, "target")

    last_close = float(window.iloc[-1]["close"])
    return _to_row(entry_ts, window.index[-1], side, plan, entry_price, last_close, "timeout")


def simulate_one_trade_no_horizon(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    side: str,
    plan: ExitPlan,
    flat_minutes_before_close: int = 5,
) -> Optional[BacktestRow]:
    """No-horizon variant: walk forward until stop OR target hits, with a
    same-trading-day cap (SPY day-trading shouldn't hold overnight). If
    neither barrier fires before session close, exit on the last bar of
    that trading day at its close — labelled 'eod_flat'.

    `flat_minutes_before_close` controls how many bars before the actual
    session close we force-flat the position (matches the live loop's
    flat_by_minutes_before_close rule).
    """
    if entry_ts not in bars.index:
        idx = bars.index.searchsorted(entry_ts)
        if idx >= len(bars):
            return None
        entry_ts = bars.index[idx]
    entry_idx = bars.index.get_loc(entry_ts)
    entry_price = float(bars.iloc[entry_idx]["close"])

    # Find the index of the last bar of the entry's trading day (in ET).
    entry_et_date = entry_ts.tz_convert("America/New_York").date()
    et_dates = bars.index.tz_convert("America/New_York").date
    same_day_mask = et_dates == entry_et_date
    same_day_idx = np.where(same_day_mask)[0]
    if len(same_day_idx) == 0:
        return None
    last_same_day = int(same_day_idx[-1])
    # Force-flat `flat_minutes_before_close` bars before the actual close.
    end_idx = max(entry_idx + 1, last_same_day - flat_minutes_before_close + 1)

    window = bars.iloc[entry_idx + 1: end_idx + 1]
    if window.empty:
        return None

    sp = plan.stop_bps / 10_000.0
    tp = plan.target_bps / 10_000.0

    for ts, bar in window.iterrows():
        reason = should_exit(
            side=side, entry_price=entry_price,
            last_high=float(bar["high"]), last_low=float(bar["low"]),
            stop_bps=plan.stop_bps, target_bps=plan.target_bps,
        )
        if reason == "stop":
            exit_px = entry_price * (1 - sp) if side == "long" else entry_price * (1 + sp)
            return _to_row(entry_ts, ts, side, plan, entry_price, exit_px, "stop")
        if reason == "target":
            exit_px = entry_price * (1 + tp) if side == "long" else entry_price * (1 - tp)
            return _to_row(entry_ts, ts, side, plan, entry_price, exit_px, "target")

    last_close = float(window.iloc[-1]["close"])
    return _to_row(entry_ts, window.index[-1], side, plan, entry_price, last_close, "eod_flat")


def simulate_one_trade_unlimited(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    side: str,
    plan: ExitPlan,
) -> Optional[BacktestRow]:
    """No-cap variant: walk forward until stop OR target hits, with NO
    same-day cap and NO horizon. Holds across overnight, weekends, anything.

    The bars frame is RTH-only, so the simulator implicitly skips overnight
    minutes. If SPY gaps overnight, the first bar of the next session will
    show that gap in its high/low — the barrier check fires on it as if a
    single-bar overnight move happened. That's a reasonable approximation
    of how the live broker would have closed the position on a stop fill at
    next-day open.

    If the bars frame runs out without either barrier firing, exit at the
    last available close — labelled 'data_end'. In live, this corresponds
    to "still open at end of backtest window".
    """
    if entry_ts not in bars.index:
        idx = bars.index.searchsorted(entry_ts)
        if idx >= len(bars):
            return None
        entry_ts = bars.index[idx]
    entry_idx = bars.index.get_loc(entry_ts)
    entry_price = float(bars.iloc[entry_idx]["close"])
    window = bars.iloc[entry_idx + 1:]
    if window.empty:
        return None

    sp = plan.stop_bps / 10_000.0
    tp = plan.target_bps / 10_000.0

    for ts, bar in window.iterrows():
        reason = should_exit(
            side=side, entry_price=entry_price,
            last_high=float(bar["high"]), last_low=float(bar["low"]),
            stop_bps=plan.stop_bps, target_bps=plan.target_bps,
        )
        if reason == "stop":
            exit_px = entry_price * (1 - sp) if side == "long" else entry_price * (1 + sp)
            return _to_row(entry_ts, ts, side, plan, entry_price, exit_px, "stop")
        if reason == "target":
            exit_px = entry_price * (1 + tp) if side == "long" else entry_price * (1 - tp)
            return _to_row(entry_ts, ts, side, plan, entry_price, exit_px, "target")

    last_close = float(window.iloc[-1]["close"])
    return _to_row(entry_ts, window.index[-1], side, plan, entry_price, last_close, "data_end")


def _to_row(entry_ts, exit_ts, side, plan, entry_price, exit_price, reason) -> BacktestRow:
    sign = 1.0 if side == "long" else -1.0
    pnl_per_share = sign * (exit_price - entry_price)
    return BacktestRow(
        entry_ts=entry_ts, side=side, qty=plan.qty,
        entry_price=entry_price, exit_price=exit_price, exit_ts=exit_ts,
        reason=reason,
        pnl_dollars=pnl_per_share * plan.qty,
        pnl_bps=(pnl_per_share / max(entry_price, 1e-9)) * 1e4,
        stop_bps=plan.stop_bps, target_bps=plan.target_bps, rv_bps=plan.rv_bps,
    )


def integration_patch_for_loop_py() -> str:
    return """
    # ---- replace LiveTrader._size with a vol-scaled equivalent ----
    # In configs/v1.yaml, add under `risk:`:
    #   vol_scaled_K: 1.0
    #   vol_scaled_rr: 2.0
    #   risk_pct_per_trade: 0.005
    #   vol_lookback_min: 30
    #
    # Then in loop.py:
    from research.vol_scaled_exits import compute_exit_plan, should_exit
    # in iterate(), AFTER computing last_price and BEFORE _enter():
    closes = df[f"{self.symbol.lower()}_close"].values[-31:]
    plan = compute_exit_plan(
        bars_close=closes, entry_price=last_price,
        equity=equity_now, side="long" if p_up >= up_t else "short",
        K=self.cfg["risk"].get("vol_scaled_K", 1.0),
        rr_ratio=self.cfg["risk"].get("vol_scaled_rr", 2.0),
        risk_pct=self.cfg["risk"].get("risk_pct_per_trade", 0.005),
        lookback_min=self.cfg["risk"].get("vol_lookback_min", 30),
    )
    # then _enter(side, plan.qty) AND save plan.stop_bps/target_bps onto Position
    # so the per-bar barrier check can run during iterate()
    # ---------------------------------------------------------------
"""
