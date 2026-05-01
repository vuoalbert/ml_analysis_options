"""Vertical spread simulator — buy one leg, sell another at different strike.

Two flavors:

DEBIT SPREAD (directional bet, pay net premium):
  • Long ATM call + short OTM call (call debit spread / bull call spread)
  • Long ATM put + short OTM put (put debit spread / bear put spread)
  Pay debit upfront. Max gain = (strike width - debit) × 100. Max loss = debit × 100.
  Caps upside but cheaper entry. More trades fit per equity, less IV-crush exposure.

CREDIT SPREAD (theta-positive, collect net premium):
  • Sell ATM put + buy OTM put (bull put spread / put credit spread)
  • Sell ATM call + buy OTM call (bear call spread / call credit spread)
  Receive credit upfront. Max gain = credit × 100. Max loss = (width - credit) × 100.
  Theta works FOR you. Profits if direction holds OR sideways.

The simulator wraps existing BS pricer to value both legs at entry/exit and
tracks the net position P&L through bar-by-bar evolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from research.option_pricer import (
    bs_price, iv_from_vix, iv_with_dynamics,
    years_to_expiry, expiry_for_dte,
)


@dataclass
class SpreadConfig:
    """Config for vertical spread strategy."""
    vix_series: pd.Series
    spread_kind: Literal["debit", "credit"] = "debit"
    leg_offset_pct: float = 0.02     # OTM offset of the second leg (e.g. 2% wider strikes)
    dte: int = 7
    conviction_min: float = 0.55
    risk_pct: float = 0.02
    max_qty: int = 100
    min_qty: int = 1
    equity: float = 30_000.0
    # Net P&L thresholds — % of net premium / max gain
    stop_pct: float = 0.50          # debit: exit if net premium drops 50%; credit: if loss = 50% of max
    target_pct: float = 0.50        # debit: exit at +50% of debit; credit: when 50% of credit captured
    entry_cost_bps: float = 100.0   # higher than naked because two legs
    exit_cost_bps: float = 100.0
    rfr: float = 0.045
    use_iv_dynamics: bool = False
    iv_beta_call: float = 5.0
    iv_beta_put: float = 8.0


def _snap_to_bar(bars: pd.DataFrame, ts):
    if ts in bars.index:
        return bars.index.get_loc(ts), ts
    idx = bars.index.searchsorted(ts)
    if idx >= len(bars):
        return None
    return idx, bars.index[idx]


def _mins_to_eod_flat(ts, flat_before_close=5):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert("America/New_York")
    flat_et = ts_et.normalize() + pd.Timedelta(hours=16) - pd.Timedelta(minutes=flat_before_close)
    if ts_et >= flat_et:
        return 0
    return int((flat_et - ts_et).total_seconds() // 60)


def _vix_for_date(vix_series: pd.Series, ts) -> float:
    ts = pd.Timestamp(ts)
    if vix_series.index.tz is not None:
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert(vix_series.index.tz)
    else:
        ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
    idx_loc = vix_series.index.searchsorted(ts, side="right") - 1
    if idx_loc < 0:
        return float(vix_series.iloc[0])
    return float(vix_series.iloc[idx_loc])


def make_spread_simulator(cfg: SpreadConfig):
    """Returns simulator(bars, entry) -> trade dict for vertical spread strategy.

    Strategy mapping:
      • model "long" + debit  → BUY ATM call, SELL OTM call (bull call debit)
      • model "long" + credit → SELL ATM put,  BUY OTM put  (bull put credit)
      • model "short" + debit → BUY ATM put,   SELL OTM put (bear put debit)
      • model "short" + credit→ SELL ATM call, BUY OTM call (bear call credit)
    """

    def simulator(bars: pd.DataFrame, entry: dict) -> Optional[dict]:
        if cfg.conviction_min > 0:
            p = entry.get("p_up", 0) if entry["side"] == "long" else entry.get("p_dn", 0)
            if p < cfg.conviction_min:
                return None

        snapped = _snap_to_bar(bars, entry["ts"])
        if snapped is None:
            return None
        entry_idx, entry_ts = snapped
        side_dir = entry["side"]
        S0 = float(bars.iloc[entry_idx]["close"])

        # Determine leg structure
        # ATM leg = strike at spot
        # OTM leg = strike at spot ± leg_offset_pct (away from spot in direction of bet)
        atm = round(S0)
        offset = S0 * cfg.leg_offset_pct
        if side_dir == "long" and cfg.spread_kind == "debit":
            # Bull call debit: buy ATM call, sell OTM call (above ATM)
            long_strike = atm
            short_strike = round(S0 + offset)
            long_side = short_side = "call"
        elif side_dir == "long" and cfg.spread_kind == "credit":
            # Bull put credit: sell ATM put, buy OTM put (below ATM)
            long_strike = round(S0 - offset)
            short_strike = atm
            long_side = short_side = "put"
        elif side_dir == "short" and cfg.spread_kind == "debit":
            # Bear put debit: buy ATM put, sell OTM put (below ATM)
            long_strike = atm
            short_strike = round(S0 - offset)
            long_side = short_side = "put"
        else:  # short + credit
            # Bear call credit: sell ATM call, buy OTM call (above ATM)
            long_strike = round(S0 + offset)
            short_strike = atm
            long_side = short_side = "call"

        if long_strike == short_strike:
            return None

        vix = _vix_for_date(cfg.vix_series, entry_ts)
        sigma = iv_from_vix(vix)
        expiry_date = expiry_for_dte(entry_ts, cfg.dte)
        T0 = years_to_expiry(entry_ts, expiry_date)
        if T0 <= 0:
            return None

        # Premium of each leg at entry
        long_prem_0 = bs_price(S=S0, K=long_strike, T=T0, r=cfg.rfr, sigma=sigma, side=long_side)
        short_prem_0 = bs_price(S=S0, K=short_strike, T=T0, r=cfg.rfr, sigma=sigma, side=short_side)

        # Net entry value (per share)
        # Debit: pay (long - short). Always positive (long is closer to ATM).
        # Credit: receive (short - long). Always positive.
        if cfg.spread_kind == "debit":
            net_entry = long_prem_0 - short_prem_0
            if net_entry <= 0.05:
                return None
        else:  # credit
            net_entry = short_prem_0 - long_prem_0
            if net_entry <= 0.05:
                return None

        # Strike width = max gain (debit) or max loss before credit (credit spread)
        width = abs(short_strike - long_strike)
        max_gain = (width - net_entry) if cfg.spread_kind == "debit" else net_entry
        max_loss = net_entry if cfg.spread_kind == "debit" else (width - net_entry)
        if max_loss <= 0:
            return None

        # Sizing: risk dollars / max loss per contract
        risk_dollars = cfg.equity * cfg.risk_pct
        cost_per_contract = max_loss * 100  # max $ at risk per contract
        qty_raw = int(risk_dollars // cost_per_contract)
        qty = max(cfg.min_qty, min(qty_raw, cfg.max_qty))
        if qty < 1:
            return None

        # Walk forward
        exit_ts = exit_net = None
        reason = None
        for j in range(entry_idx + 1, len(bars)):
            ts_j = bars.index[j]
            S_j = float(bars.iloc[j]["close"])
            T_j = years_to_expiry(ts_j, expiry_date)

            if cfg.use_iv_dynamics:
                spot_pct = (S_j - S0) / S0
                sigma_j_long = iv_with_dynamics(
                    sigma_entry=sigma, spot_pct_move=spot_pct, side_opt=long_side,
                    iv_beta_call=cfg.iv_beta_call, iv_beta_put=cfg.iv_beta_put)
                sigma_j_short = iv_with_dynamics(
                    sigma_entry=sigma, spot_pct_move=spot_pct, side_opt=short_side,
                    iv_beta_call=cfg.iv_beta_call, iv_beta_put=cfg.iv_beta_put)
            else:
                sigma_j_long = sigma_j_short = sigma

            long_prem_j = bs_price(S=S_j, K=long_strike, T=T_j, r=cfg.rfr,
                                    sigma=sigma_j_long, side=long_side)
            short_prem_j = bs_price(S=S_j, K=short_strike, T=T_j, r=cfg.rfr,
                                     sigma=sigma_j_short, side=short_side)

            if cfg.spread_kind == "debit":
                net_j = long_prem_j - short_prem_j
                # P&L per share = net_j - net_entry (positive when spread widens)
                pnl_per_share = net_j - net_entry
            else:  # credit
                net_j = short_prem_j - long_prem_j
                # P&L per share = net_entry - net_j (positive when spread narrows)
                pnl_per_share = net_entry - net_j

            mins_to_flat = _mins_to_eod_flat(ts_j)

            # Stop/target on % of max_loss / max_gain
            if pnl_per_share <= -cfg.stop_pct * max_loss:
                exit_ts = ts_j; exit_net = net_j; reason = "stop"; break
            if pnl_per_share >= cfg.target_pct * max_gain:
                exit_ts = ts_j; exit_net = net_j; reason = "target"; break
            if mins_to_flat <= 0:
                exit_ts = ts_j; exit_net = net_j; reason = "eod_flat"; break
            if T_j <= 0:
                exit_ts = ts_j; exit_net = net_j; reason = "expiry"; break

        if exit_ts is None:
            j = len(bars) - 1
            ts_j = bars.index[j]
            S_j = float(bars.iloc[j]["close"])
            T_j = years_to_expiry(ts_j, expiry_date)
            long_prem_j = bs_price(S=S_j, K=long_strike, T=T_j, r=cfg.rfr, sigma=sigma, side=long_side)
            short_prem_j = bs_price(S=S_j, K=short_strike, T=T_j, r=cfg.rfr, sigma=sigma, side=short_side)
            if cfg.spread_kind == "debit":
                net_j = long_prem_j - short_prem_j
            else:
                net_j = short_prem_j - long_prem_j
            exit_ts = ts_j
            exit_net = net_j
            reason = "data_end"

        # Final P&L
        if cfg.spread_kind == "debit":
            pnl_per_share = exit_net - net_entry
        else:
            pnl_per_share = net_entry - exit_net
        gross = pnl_per_share * 100 * qty

        # Costs (2 legs each way = 4× spread cost)
        notional_per_leg = (long_prem_0 + short_prem_0) * 100 * qty
        cost = notional_per_leg * (cfg.entry_cost_bps + cfg.exit_cost_bps) / 1e4

        net_dollars = gross - cost
        hold_min = int((pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 60)

        return {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side_dir,
            "qty": qty,
            "spread_kind": cfg.spread_kind,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "net_entry": net_entry,
            "exit_net": exit_net,
            "max_loss_$": max_loss * 100 * qty,
            "max_gain_$": max_gain * 100 * qty,
            "pnl_dollars": net_dollars,
            "pnl_dollars_gross": gross,
            "reason": reason,
            "hold_min": hold_min,
            "p_up": entry.get("p_up"),
            "p_dn": entry.get("p_dn"),
            "spot_at_entry": S0,
        }

    return simulator
