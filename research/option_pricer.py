"""Black-Scholes pricer used by the options backtest.

This is the standard textbook model. We use it because:
  • At-the-money near-dated SPY options are extremely well-modelled by BS
    (constant-vol assumption is least wrong when K ≈ S and T is small).
  • Alpaca free tier doesn't include historical option chains, so we have
    no observed premium series to walk against. BS is the cheapest honest
    proxy that captures the two effects we care about for 0DTE strategies:
        1. Delta — price moves with SPY
        2. Theta — premium decays toward zero as T → 0

What we DON'T model (and the resulting bias):
  • Skew — real OTM puts are pricier than BS. We trade ATM, so this is small.
  • Vol-of-vol / IV reactions — real IV expands when the market drops. A long
    put position would benefit from this in real life and it's invisible here.
  • Bid/ask spread — added separately as a fixed bps cost in the simulator.
  • Early exercise — irrelevant for European-style cash-settled SPY.

IV input: we use VIX as a proxy. VIX is 30-day SPX implied vol, which
overstates ATM 0DTE SPY IV by ~5-15% on quiet days and understates it on
event days. Good enough for a first-pass strategy comparison; not good enough
for go/no-go on real money.
"""
from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone

import numpy as np


# ---------- Black-Scholes ----------

def _phi(x: float) -> float:
    """Standard normal CDF via erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    *,
    S: float,             # spot
    K: float,             # strike
    T: float,             # time to expiry in years (1 day = 1/252 ≈ 0.00397)
    r: float,             # risk-free rate (annualized, decimal)
    sigma: float,         # implied vol (annualized, decimal)
    side: str,            # "call" or "put"
) -> float:
    """Return per-share Black-Scholes price.

    Multiply by 100 for total premium per contract.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Expired — intrinsic only
        if side == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if side == "call":
        return S * _phi(d1) - K * math.exp(-r * T) * _phi(d2)
    else:
        return K * math.exp(-r * T) * _phi(-d2) - S * _phi(-d1)


# ---------- session-time helpers ----------

# US equity options: SPY expires at 4pm ET on the expiration date.
# We measure time-to-expiry in years using a 252-day calendar (industry std).
# Within a session, we use 6.5 trading hours per day to get smooth intraday
# theta. Across days, we count calendar days × 6.5h (still 252 trading days
# per year, just billed as if every day had a session — fine for a backtest
# at the resolution of "is theta noticeable today vs. a week from now").
SECONDS_PER_TRADING_YEAR = 252 * 6.5 * 3600   # 252 days × 6.5 RTH hours
SECONDS_PER_TRADING_DAY = 6.5 * 3600


def years_to_expiry(ts_utc, expiry_date) -> float:
    """Time from `ts_utc` to 4pm ET on `expiry_date`, in years (252-day basis).

    `expiry_date` is a python date or pd.Timestamp — interpreted in ET.
    Returns 0 if ts is at or after 4pm ET on the expiry date.

    Within the expiry day, decay tracks intraday seconds. Earlier days
    contribute a flat SECONDS_PER_TRADING_DAY each (industry convention —
    overnight theta is real but we don't separate it from intraday here).
    """
    import pandas as pd
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert("America/New_York")

    # Normalize expiry to a date in ET
    exp = pd.Timestamp(expiry_date)
    if exp.tzinfo is None:
        exp_et = exp.tz_localize("America/New_York")
    else:
        exp_et = exp.tz_convert("America/New_York")
    expiry_close_et = exp_et.normalize() + pd.Timedelta(hours=16)

    if ts_et >= expiry_close_et:
        return 0.0

    # Same-session: just measure intraday seconds remaining
    if ts_et.normalize() == exp_et.normalize():
        seconds_left = (expiry_close_et - ts_et).total_seconds()
        return seconds_left / SECONDS_PER_TRADING_YEAR

    # Multi-day: time-left-today + N full sessions + intraday on expiry day
    today_close_et = ts_et.normalize() + pd.Timedelta(hours=16)
    seconds_today = max(0.0, (today_close_et - ts_et).total_seconds())

    # Count business days strictly between today and expiry day
    next_day = (ts_et + pd.Timedelta(days=1)).normalize()
    prev_day = exp_et.normalize() - pd.Timedelta(days=1)
    if next_day <= prev_day:
        full_days = pd.bdate_range(next_day.date(), prev_day.date()).size
    else:
        full_days = 0

    seconds_full = full_days * SECONDS_PER_TRADING_DAY
    seconds_expiry_day = SECONDS_PER_TRADING_DAY  # full session on expiry day until 4pm
    total_seconds = seconds_today + seconds_full + seconds_expiry_day
    return total_seconds / SECONDS_PER_TRADING_YEAR


def years_to_expiry_0dte(ts_utc) -> float:
    """Backwards-compatible 0DTE shortcut — expiry = same date as ts."""
    import pandas as pd
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return years_to_expiry(ts, ts.tz_convert("America/New_York").date())


def expiry_for_dte(ts_utc, dte: int):
    """Return the expiration date for an N-DTE option entered at `ts_utc`.

    DTE counts calendar days for SPY weekly expirations (Mon/Wed/Fri),
    but for our purposes we approximate with business days — dte=0 is
    today, dte=7 is 7 business days forward, etc.
    """
    import pandas as pd
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert("America/New_York")
    if dte == 0:
        return ts_et.date()
    # Use business days (skip weekends) to land on a real expiry
    target = pd.bdate_range(start=ts_et.normalize(), periods=dte + 1)[-1]
    return target.date()


# ---------- IV from VIX ----------

def iv_from_vix(vix_close: float) -> float:
    """Convert VIX index level to a usable σ for Black-Scholes.

    VIX is reported as a percentage (e.g. 18.4 means 18.4% annualized vol).
    The naive conversion is just vix/100. For 0DTE ATM SPY this is a slight
    overestimate (VIX is 30-day SPX, ATM 0DTE SPY tends to run 5-15% lower
    on calm days). We don't correct for this — overstating IV biases the
    backtest pessimistically, which is the safer direction for a strategy
    sanity check.
    """
    return max(0.05, float(vix_close) / 100.0)


DEFAULT_IV_BETA_CALL = 5.0  # mid-realistic
DEFAULT_IV_BETA_PUT = 8.0


def iv_with_dynamics(
    *,
    sigma_entry: float,
    spot_pct_move: float,    # signed % move of underlying since entry (e.g. +0.005 = +0.5%)
    side_opt: str,            # "call" or "put"
    iv_beta_call: float = DEFAULT_IV_BETA_CALL,
    iv_beta_put: float = DEFAULT_IV_BETA_PUT,
    iv_floor: float = 0.05,
    iv_ceiling: float = 1.50,
) -> float:
    """Approximate intraday IV evolution.

    Empirical SPY ATM IV vs. spot regression (~1y of 5-min bars):
      ΔIV/IV ≈ −β × ΔS/S    for long calls (rally → IV compresses)
      ΔIV/IV ≈ −β × ΔS/S    for long puts (selloff → IV expands)

    The β's above are conservative. β_call < β_put captures the negative
    skew (puts respond more to selloffs than calls do to rallies).

    For long calls, the strategy gets hurt: rally compresses IV → less
    premium gain than BS-with-constant-IV would suggest. For long puts,
    the strategy benefits: selloff expands IV → more premium gain.
    """
    if side_opt == "call":
        # Long calls in a rally: IV compresses (negative spot move ≠ scenario for calls)
        new_sigma = sigma_entry * (1.0 - iv_beta_call * spot_pct_move)
    else:  # put
        # Long puts in a selloff: IV expands as spot drops (negative ΔS).
        # spot_pct_move negative → new_sigma > sigma_entry. Same formula as calls
        # but with the put-specific beta.
        new_sigma = sigma_entry * (1.0 - iv_beta_put * spot_pct_move)
    return max(iv_floor, min(iv_ceiling, new_sigma))


# ---------- entry/exit pricing helpers ----------

def price_at(
    *,
    S: float,
    K: float,
    ts_utc,
    side: str,
    vix: float,
    expiry_date=None,    # if None, uses 0DTE (same-day expiry)
    r: float = 0.045,    # rough avg DFII10 over the backtest window — not material
) -> float:
    """All-in helper: spot, strike, timestamp, side, VIX → per-share BS price."""
    if expiry_date is None:
        T = years_to_expiry_0dte(ts_utc)
    else:
        T = years_to_expiry(ts_utc, expiry_date)
    sigma = iv_from_vix(vix)
    return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, side=side)
