"""Options execution layer for the SPY ML strategy.

The entry signal (p_up / p_down from the LightGBM model) is unchanged —
it predicts SPY direction. This module translates that directional view
into an options trade:

  • Pick a contract (strike + expiration) given current SPY price + side
  • Compute a position size in contracts (risk-budget aware)
  • Format the OCC option symbol for the broker
  • Decide when to exit (premium-based stop/target + theta-decay protection)

Designed to mirror the structure of research.vol_scaled_exits.compute_exit_plan
so the live loop can dispatch on a config flag without much restructuring.

Default config (configs/v1.yaml strategy.options) is the research-validated
winner: 7-DTE ATM, p≥0.55, risk 2%, multi=3 concurrent positions, no theta
protection. See research/sweep_v4_stress_oos.py for validation history.

LIVE INTEGRATION TODO (loop.py changes needed for the new config):
  1. Pass conviction_min from config → reject entries where max(p_up, p_dn) < it.
  2. Pass max_concurrent_positions → loop should track open option positions
     in a list, allow up to N before blocking new entries.
  3. Use resolve_expiration() below when building the contract — currently
     loop.py calls pick_contract(expiration=date.today()) which forces 0DTE.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest, MarketOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest


# ---------- contract spec & symbol formatting ----------

@dataclass
class ContractSpec:
    """A specific option contract to trade."""
    underlying: str          # "SPY"
    expiration: date         # e.g., date(2026, 4, 30)
    strike: float            # e.g., 711.0
    side: str                # "call" or "put"
    occ_symbol: str          # e.g., "SPY260430C00711000"


def format_occ_symbol(underlying: str, expiration: date, side: str, strike: float) -> str:
    """Format an OCC-standard option symbol.

    Format: ROOT(6) + YYMMDD(6) + CALL/PUT(1) + STRIKE×1000(8)
    Example: SPY260430C00711000  (SPY, 2026-04-30, call, $711.00)
    """
    root = underlying.ljust(6)[:6]
    date_str = expiration.strftime("%y%m%d")
    cp = "C" if side == "call" else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{root.strip()}{date_str}{cp}{strike_str}"


# ---------- expiration resolution ----------

def resolve_expiration(spec: str, today: Optional[date] = None) -> date:
    """Convert config string → expiration date.

    Supported values for `spec`:
      • "same_day" / "0dte"           — today
      • "next_friday"                  — next Friday on or after today
      • "next_monthly"                 — third Friday of next month (standard)
      • "7_business_days" / "7dte"    — 7 business days from today
      • "Ndte" or "N_business_days"   — N business days from today (parameterised)

    The 7DTE default matches the research-validated winner config.
    """
    if today is None:
        today = date.today()

    s = (spec or "").strip().lower()
    if s in {"same_day", "0dte", ""}:
        return today
    if s == "next_friday":
        # 4 = Friday in Python's weekday()
        days_ahead = (4 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)
    if s == "next_monthly":
        # Third Friday of next month
        if today.month == 12:
            year, month = today.year + 1, 1
        else:
            year, month = today.year, today.month + 1
        first = date(year, month, 1)
        first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
        return first_friday + timedelta(days=14)

    # "Ndte" or "N_business_days"
    n = None
    if s.endswith("dte"):
        try:
            n = int(s[:-3])
        except ValueError:
            n = None
    elif s.endswith("_business_days"):
        try:
            n = int(s.split("_")[0])
        except ValueError:
            n = None

    if n is None:
        # Unknown spec — fall back to same-day
        return today

    # Walk forward N business days (skip Sat/Sun)
    d = today
    added = 0
    while added < n:
        d = d + timedelta(days=1)
        if d.weekday() < 5:   # Mon-Fri
            added += 1
    return d


# ---------- contract selection ----------

def pick_contract(
    *,
    trading: TradingClient,
    underlying: str,
    side: str,                    # "call" if direction long, "put" if short
    last_price: float,
    expiration: Optional[date] = None,   # None = next 0DTE-eligible day
    strike_offset: float = 0.0,   # +1 = $1 OTM, -1 = $1 ITM (legacy fixed-$ offset)
    moneyness: str = "atm",       # "atm" / "otm" / "itm"
    itm_offset_pct: float = 0.005,  # 0.5% ITM by default — research-validated 2.5% beats baseline +41%
) -> Optional[ContractSpec]:
    """Query Alpaca for the best matching option contract.

    Returns None if no suitable contract found (e.g., expiration not listed).

    Strike selection by moneyness:
      • atm: round(spot)
      • itm: spot × (1 ± itm_offset_pct), so strike is in-the-money by that pct
      • otm: spot × (1 ± itm_offset_pct), strike out-of-the-money by that pct
    """
    if expiration is None:
        expiration = date.today()
    target_strike = round(last_price + strike_offset)

    # Percent-based moneyness offset — research showed deeper ITM (2.5%) beats baseline +41%
    shift = last_price * itm_offset_pct
    if moneyness == "otm":
        # OTM call → strike above spot; OTM put → strike below
        target_strike = round(last_price + shift) if side == "call" else round(last_price - shift)
    elif moneyness == "itm":
        # ITM call → strike below spot; ITM put → strike above
        target_strike = round(last_price - shift) if side == "call" else round(last_price + shift)

    try:
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date=expiration,
            strike_price_gte=str(target_strike - 2),
            strike_price_lte=str(target_strike + 2),
            type=side,                     # "call" or "put"
            status=AssetStatus.ACTIVE,
            limit=20,
        )
        resp = trading.get_option_contracts(req)
    except Exception:
        return None

    contracts = resp.option_contracts if hasattr(resp, "option_contracts") else resp
    if not contracts:
        return None

    # Pick the strike closest to target
    best = min(contracts, key=lambda c: abs(float(c.strike_price) - target_strike))
    return ContractSpec(
        underlying=underlying,
        expiration=best.expiration_date,
        strike=float(best.strike_price),
        side=side,
        occ_symbol=best.symbol,
    )


# ---------- premium quote ----------

def get_premium_quote(option_data: OptionHistoricalDataClient,
                      symbol: str) -> tuple[float, float]:
    """Return (bid, ask) for an option contract."""
    try:
        req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
        resp = option_data.get_option_latest_quote(req)
        q = resp.get(symbol)
        if q is None:
            return 0.0, 0.0
        return float(q.bid_price or 0.0), float(q.ask_price or 0.0)
    except Exception:
        return 0.0, 0.0


# ---------- position sizing ----------

@dataclass
class OptionPlan:
    contract: ContractSpec
    qty_contracts: int
    entry_premium: float       # estimated $ premium per contract (× 100 = total)
    risk_dollars: float        # max loss = qty × premium × 100
    note: str


def plan_options_entry(
    *,
    contract: ContractSpec,
    bid: float,
    ask: float,
    equity: float,
    risk_pct: float = 0.005,
    max_qty: int = 10,
    min_qty: int = 1,
) -> Optional[OptionPlan]:
    """Compute how many contracts to buy.

    Risk per trade = equity × risk_pct.
    Premium per contract = ask × 100 (× 100 because each contract is 100 shares).
    qty = floor(risk_$ / premium_per_contract), capped at max_qty.
    """
    if ask <= 0:
        return None
    premium_per_contract = ask * 100  # $ paid per contract
    risk_dollars = equity * risk_pct
    qty_raw = int(risk_dollars // premium_per_contract)
    qty = max(min_qty, min(qty_raw, max_qty))
    if qty < 1:
        return None
    return OptionPlan(
        contract=contract,
        qty_contracts=qty,
        entry_premium=ask,
        risk_dollars=qty * premium_per_contract,
        note=(f"{contract.side.upper()} {contract.occ_symbol} "
              f"ask=${ask:.2f} premium/contract=${premium_per_contract:.0f} "
              f"qty={qty} max_loss=${qty * premium_per_contract:.0f}"),
    )


# ---------- exit decision ----------

def check_options_exit(
    *,
    side: str,
    entry_premium: float,
    current_premium: float,
    mins_held: int,
    mins_to_eod_flat: int,
    stop_pct: float = 0.50,           # exit if premium drops 50%
    target_pct: float = 1.00,         # exit if premium doubles
    theta_protect_mins: int = 60,     # if 60 min before EOD-flat and not in profit, exit
) -> Optional[str]:
    """Premium-based + time-decay-aware exit decision.

    Returns one of: "stop", "target", "theta_protect", "eod_flat", or None.
    """
    if entry_premium <= 0:
        return None
    pct_change = (current_premium - entry_premium) / entry_premium
    if pct_change <= -stop_pct:
        return "stop"
    if pct_change >= target_pct:
        return "target"
    # Theta protection — 0DTE options lose value rapidly in last hour.
    # If we're not in profit (>10% gain) and EOD-flat is approaching, get out.
    if mins_to_eod_flat <= theta_protect_mins and pct_change < 0.10:
        return "theta_protect"
    if mins_to_eod_flat <= 5:
        return "eod_flat"
    return None
