"""Bar-by-bar simulator for the options strategy.

Plugs into research.backtest_compare.walk_with_blocking — exposes the same
`simulator(bars, entry_signal) -> trade_dict | None` interface that the stock
backtest uses, so we get apples-to-apples comparison via the existing harness.

Per signal:
  1. Resolve entry bar (handle weekend/missing minutes by snapping forward).
  2. Pick strike = round(spot)  ← ATM, matches live live/options.py.
  3. Price entry premium via Black-Scholes (S, K, T_to_4pm_ET, σ_from_VIX).
  4. Size qty = floor(equity × risk_pct / (premium × 100)), capped at max_qty.
  5. Walk forward minute-by-minute. At each bar, reprice premium using
     updated spot + shrinking T. Check exit conditions (matches the live
     check_options_exit logic in live/options.py exactly):
        • stop:           premium drops ≥ stop_pct (default 50%)
        • target:         premium rises ≥ target_pct (default 100%)
        • theta_protect:  ≤ theta_protect_mins to EOD-flat AND not in profit
        • eod_flat:       ≤ flat_by_minutes_before_close to 4pm ET
  6. Compute P&L:
        gross_$ = (exit_premium − entry_premium) × qty × 100
        cost_$  = entry_cost_bps + exit_cost_bps applied to premium notional
        net_$   = gross_$ − cost_$

A note on costs: SPY 0DTE bid/ask is typically $0.01-0.05 wide on the
liquid strikes. We model 50 bps each way (100 bps round-trip) of the
premium notional — a reasonable conservative average for retail size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from research.option_pricer import (
    bs_price, iv_from_vix, iv_with_dynamics,
    years_to_expiry, years_to_expiry_0dte, expiry_for_dte,
)


# ---------- defaults (mirror configs/v1.yaml strategy.options block) ----------
EQUITY = 30_000.0           # match the stock backtest baseline so results compare
RISK_PCT = 0.01             # 1% of equity per trade in premium
MAX_QTY = 10
MIN_QTY = 1
STOP_PCT = 0.50
TARGET_PCT = 1.00
THETA_PROTECT_MINS = 60
FLAT_BEFORE_CLOSE_MINS = 5
ENTRY_COST_BPS = 50.0       # 0.5% slippage one-way (bid/ask spread + slip)
EXIT_COST_BPS = 50.0
RFR = 0.045                 # avg DFII10 over the backtest window


# ---------- helpers ----------

def _snap_to_bar(bars: pd.DataFrame, ts):
    """Find the first bar index ≥ ts. Returns (idx, snapped_ts) or None."""
    if ts in bars.index:
        return bars.index.get_loc(ts), ts
    idx = bars.index.searchsorted(ts)
    if idx >= len(bars):
        return None
    return idx, bars.index[idx]


def _mins_to_eod_flat(ts):
    """Minutes from `ts` to (4pm ET − FLAT_BEFORE_CLOSE_MINS).

    Returns 0 if already past that cutoff.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert("America/New_York")
    flat_et = ts_et.normalize() + pd.Timedelta(hours=16) - pd.Timedelta(minutes=FLAT_BEFORE_CLOSE_MINS)
    if ts_et >= flat_et:
        return 0
    return int((flat_et - ts_et).total_seconds() // 60)


def _vix_for_date(vix_series: pd.Series, ts) -> float:
    """Use the most recent VIX close on or before `ts`.

    `vix_series` may be daily or minute-aligned; we just take the most
    recent value at or before the entry timestamp.
    """
    ts = pd.Timestamp(ts)
    if isinstance(vix_series.index, pd.DatetimeIndex):
        # Match tz: if series is tz-aware, ensure ts is too (and vice versa).
        if vix_series.index.tz is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert(vix_series.index.tz)
        else:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        idx_loc = vix_series.index.searchsorted(ts, side="right") - 1
        if idx_loc < 0:
            return float(vix_series.iloc[0])
        return float(vix_series.iloc[idx_loc])
    # fallback: try direct lookup by date
    d = ts.date()
    if d in vix_series.index:
        return float(vix_series.loc[d])
    return float(vix_series.iloc[-1])


# ---------- the simulator ----------

@dataclass
class OptionsSimConfig:
    vix_series: pd.Series                # daily VIX close (or 1-min, doesn't matter)
    equity: float = EQUITY
    risk_pct: float = RISK_PCT
    max_qty: int = MAX_QTY
    min_qty: int = MIN_QTY
    stop_pct: float = STOP_PCT
    target_pct: float = TARGET_PCT
    theta_protect_mins: int = THETA_PROTECT_MINS
    theta_protect_profit_thresh: float = 0.10  # if mins_to_flat ≤ X and pct_change < this, exit
    theta_protect_any_dte: bool = False  # if True, fires for any DTE (matches live behaviour)
    entry_cost_bps: float = ENTRY_COST_BPS
    exit_cost_bps: float = EXIT_COST_BPS
    rfr: float = RFR
    dte: int = 0                          # days-to-expiry at entry (0 = same day)
    conviction_min: float = 0.0           # filter: skip signals where p < this (0 = no filter)
    hold_overnight: bool = False          # if True, multi-day holds; if False, force EOD flat
    # Conviction-weighted sizing — when conviction_hi > conviction_lo, scales risk linearly:
    #   s = clip((p - conviction_lo) / (conviction_hi - conviction_lo), 0, 1)
    #   risk_pct = risk_pct + (risk_pct_max - risk_pct) * s
    conviction_lo: float = 0.0
    conviction_hi: float = 0.0
    risk_pct_max: float = 0.0
    moneyness: str = "atm"                # "atm" (Δ ≈ 0.5) or "itm" (Δ ≈ 0.7) or "otm" (Δ ≈ 0.3)
    itm_offset_pct: float = 0.005         # 0.5% ITM for delta ~0.65-0.7 on 7DTE
    use_iv_dynamics: bool = False         # if True, recompute IV at each bar using spot move
    iv_beta_call: float = 5.0             # tunable for stress tests (realistic 5; pessimistic 10)
    iv_beta_put: float = 8.0              # tunable for stress tests (realistic 8; pessimistic 4)
    vix_min: float = 0.0                  # skip signals when VIX < this (calm tape kills options)
    vix_max: float = 100.0                # skip signals when VIX > this (panic — IV too rich)
    skip_hours_et: tuple = ()             # tuple of ET hours to skip, e.g. (11, 12, 13) for midday
    # Trailing stop: once premium has gained X%, lock in (Y%) of those gains as a floor.
    # When set, exits if premium drops below entry × (1 + (max_gain × (1 - trail_lock_frac))).
    # E.g., trailing_threshold_pct=0.30, trail_lock_frac=0.50 means: once up 30%, lock in 15%.
    trailing_threshold_pct: float = 0.0   # 0 = trailing stop disabled
    trail_lock_frac: float = 0.5          # fraction of gains to lock in
    # Per-symbol concentration cap — limit how many of the N concurrent positions
    # can be on the SAME OCC symbol (prevents stacking 15 positions on one strike).
    max_per_symbol: int = 0               # 0 = no cap (current behavior)


def make_options_simulator(cfg: OptionsSimConfig):
    """Returns a `simulator(bars, entry_signal)` callable for walk_with_blocking."""

    def simulator(bars: pd.DataFrame, entry: dict) -> Optional[dict]:
        # Conviction filter — skip low-confidence signals
        if cfg.conviction_min > 0:
            p = entry.get("p_up", 0) if entry["side"] == "long" else entry.get("p_dn", 0)
            if p < cfg.conviction_min:
                return None

        # Time-of-day filter — skip ET hours in skip_hours_et
        if cfg.skip_hours_et:
            ts_et = pd.Timestamp(entry["ts"]).tz_convert("America/New_York")
            if ts_et.hour in cfg.skip_hours_et:
                return None

        # VIX regime filter — skip signals where VIX is outside band
        if cfg.vix_min > 0 or cfg.vix_max < 100:
            vix_check = _vix_for_date(cfg.vix_series, entry["ts"])
            if vix_check < cfg.vix_min or vix_check > cfg.vix_max:
                return None

        snapped = _snap_to_bar(bars, entry["ts"])
        if snapped is None:
            return None
        entry_idx, entry_ts = snapped
        side_dir = entry["side"]                 # "long" / "short"
        side_opt = "call" if side_dir == "long" else "put"

        S0 = float(bars.iloc[entry_idx]["close"])
        # Strike selection — ATM, ITM, or OTM
        if cfg.moneyness == "itm":
            # ITM call → strike below spot; ITM put → strike above spot
            shift = S0 * cfg.itm_offset_pct
            K = round(S0 - shift) if side_dir == "long" else round(S0 + shift)
        elif cfg.moneyness == "otm":
            shift = S0 * cfg.itm_offset_pct
            K = round(S0 + shift) if side_dir == "long" else round(S0 - shift)
        else:
            K = round(S0)                        # ATM
        vix = _vix_for_date(cfg.vix_series, entry_ts)
        sigma = iv_from_vix(vix)

        # Pick expiration based on DTE config
        expiry_date = expiry_for_dte(entry_ts, cfg.dte)
        T0 = years_to_expiry(entry_ts, expiry_date)
        if T0 <= 0:
            # Entry too late in the day for 0DTE
            return None
        entry_premium = bs_price(S=S0, K=K, T=T0, r=cfg.rfr, sigma=sigma, side=side_opt)
        if entry_premium <= 0.05:
            # Premium too small — would round to qty=0 or have terrible spread cost in real life
            return None

        # ---------- sizing ----------
        # Conviction-weighted risk if configured (matches stocks pattern).
        if cfg.conviction_hi > cfg.conviction_lo and cfg.risk_pct_max > cfg.risk_pct:
            p = entry.get("p_up", 0) if side_dir == "long" else entry.get("p_dn", 0)
            s = max(0.0, min(1.0,
                (p - cfg.conviction_lo) / (cfg.conviction_hi - cfg.conviction_lo)))
            risk_pct_eff = cfg.risk_pct + (cfg.risk_pct_max - cfg.risk_pct) * s
        else:
            risk_pct_eff = cfg.risk_pct
        premium_per_contract = entry_premium * 100.0
        risk_dollars = cfg.equity * risk_pct_eff
        qty_raw = int(risk_dollars // premium_per_contract)
        qty = max(cfg.min_qty, min(qty_raw, cfg.max_qty))
        if qty < 1:
            return None

        # ---------- walk forward ----------
        exit_ts = None
        exit_premium = None
        reason = None
        max_gain = 0.0   # high-water mark for trailing stop
        for j in range(entry_idx + 1, len(bars)):
            ts_j = bars.index[j]
            S_j = float(bars.iloc[j]["close"])
            T_j = years_to_expiry(ts_j, expiry_date)

            # Optionally update IV with spot move (vol crush on rallies, vol expansion on selloffs)
            if cfg.use_iv_dynamics:
                spot_move_pct = (S_j - S0) / S0
                sigma_j = iv_with_dynamics(
                    sigma_entry=sigma, spot_pct_move=spot_move_pct, side_opt=side_opt,
                    iv_beta_call=cfg.iv_beta_call, iv_beta_put=cfg.iv_beta_put)
            else:
                sigma_j = sigma
            premium_j = bs_price(S=S_j, K=K, T=T_j, r=cfg.rfr, sigma=sigma_j, side=side_opt)
            pct_change = (premium_j - entry_premium) / entry_premium
            mins_to_flat = _mins_to_eod_flat(ts_j)

            # Trailing stop: once we've crossed trailing_threshold_pct, lock in some gains.
            # Floor moves up monotonically as max_gain grows.
            if cfg.trailing_threshold_pct > 0 and pct_change > cfg.trailing_threshold_pct:
                max_gain = max(max_gain, pct_change)
                # locked floor = max_gain × (1 - trail_lock_frac)
                # if trail_lock_frac=0.5 → floor at 50% of peak
                trail_floor = max_gain * (1.0 - cfg.trail_lock_frac)
                if pct_change <= trail_floor:
                    exit_ts = ts_j; exit_premium = premium_j; reason = "trailing"; break

            # Exit checks — order matches live check_options_exit
            if pct_change <= -cfg.stop_pct:
                exit_ts = ts_j; exit_premium = premium_j; reason = "stop"; break
            if pct_change >= cfg.target_pct:
                exit_ts = ts_j; exit_premium = premium_j; reason = "target"; break
            # Theta protection only meaningful for 0DTE — turn off if DTE > 0 unless overridden
            # theta_protect — fires for DTE=0 always; for DTE>0 if theta_protect_any_dte=True
            # (matches live `live/options.py::check_options_exit` which doesn't gate on DTE)
            if ((cfg.dte == 0 or cfg.theta_protect_any_dte)
                and cfg.theta_protect_mins > 0
                and mins_to_flat <= cfg.theta_protect_mins
                and pct_change < cfg.theta_protect_profit_thresh):
                exit_ts = ts_j; exit_premium = premium_j; reason = "theta_protect"; break
            if not cfg.hold_overnight and mins_to_flat <= 0:
                # forced flat at session end
                exit_ts = ts_j; exit_premium = premium_j; reason = "eod_flat"; break
            # If holding overnight, also check expiry
            if T_j <= 0:
                exit_ts = ts_j; exit_premium = premium_j; reason = "expiry"; break

        if exit_ts is None:
            # ran off end of bars — close at last available premium
            j = len(bars) - 1
            ts_j = bars.index[j]
            S_j = float(bars.iloc[j]["close"])
            T_j = years_to_expiry(ts_j, expiry_date)
            if cfg.use_iv_dynamics:
                sigma_j = iv_with_dynamics(
                    sigma_entry=sigma, spot_pct_move=(S_j - S0) / S0, side_opt=side_opt,
                    iv_beta_call=cfg.iv_beta_call, iv_beta_put=cfg.iv_beta_put)
            else:
                sigma_j = sigma
            exit_premium = bs_price(S=S_j, K=K, T=T_j, r=cfg.rfr, sigma=sigma_j, side=side_opt)
            exit_ts = ts_j
            reason = "data_end"

        # ---------- P&L ----------
        # Premium move × 100 shares × qty contracts, less round-trip costs.
        gross = (exit_premium - entry_premium) * 100.0 * qty
        # Costs are bps of premium notional, applied to entry and exit notionals
        entry_notional = entry_premium * 100.0 * qty
        exit_notional = exit_premium * 100.0 * qty
        cost = (entry_notional * cfg.entry_cost_bps / 1e4) + (exit_notional * cfg.exit_cost_bps / 1e4)
        net_dollars = gross - cost

        # bps relative to risk dollars (so it's comparable across qty)
        pnl_bps_gross = (gross / max(entry_notional, 1.0)) * 1e4
        pnl_bps_net = (net_dollars / max(entry_notional, 1.0)) * 1e4

        hold_min = int((pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 60)

        return {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side_dir,
            "side_opt": side_opt,
            "qty": qty,
            "strike": K,
            "spot_at_entry": S0,
            "spot_at_exit": float(bars.loc[exit_ts]["close"]) if exit_ts in bars.index else S0,
            "entry_premium": entry_premium,
            "exit_premium": exit_premium,
            "vix_at_entry": vix,
            "pnl_dollars": net_dollars,
            "pnl_dollars_gross": gross,
            "pnl_bps_gross": pnl_bps_gross,
            "pnl_bps_net": pnl_bps_net,
            "reason": reason,
            "hold_min": hold_min,
            "p_up": entry.get("p_up"),
            "p_dn": entry.get("p_dn"),
        }

    return simulator
