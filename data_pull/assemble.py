"""Assemble SPY + cross-asset + macro + events into a single minute-grid parquet.

Design:
  - Minute index comes from NYSE RTH schedule (UTC).
  - SPY bars: join on minute.
  - Cross-asset ETFs (Alpaca): join on minute.
  - VIX/ES=F (yfinance daily): forward-fill into minute grid with 1-day lag to kill lookahead.
  - FRED daily macro: same 1-day-lag forward-fill.
  - Event flags (FOMC/0DTE): binary per bar.

The output has one row per SPY minute. Non-SPY price columns are prefixed with symbol.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from utils.config import load as load_cfg
from utils.calendar import rth_index, is_fomc_day, is_zero_dte, minutes_into_session
from utils.logging import get
from . import bars, yf_daily, fred, cache

log = get("data.assemble")

ALPACA_CROSS = {"TLT", "UUP", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB", "XLC"}
YF_DAILY_CROSS = {"^VIX": "vix", "ES=F": "es", "DX=F": "dxy"}


def _align_minute(df: pd.DataFrame, minute_idx: pd.DatetimeIndex, cols: list[str], prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=minute_idx, columns=[f"{prefix}_{c}" for c in cols])
    sub = df[cols].copy()
    sub.columns = [f"{prefix}_{c}" for c in cols]
    sub = sub.reindex(minute_idx)
    return sub


def _daily_ffill(daily: pd.Series | pd.DataFrame, minute_idx: pd.DatetimeIndex, lag_days: int = 1) -> pd.DataFrame:
    """Forward-fill a daily series onto a minute index with a `lag_days` shift to prevent lookahead."""
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()
    if daily.empty:
        return pd.DataFrame(index=minute_idx, columns=daily.columns)
    # Shift by lag_days trading days (approximate with 1 calendar day — conservative).
    d = daily.copy()
    d.index = d.index + pd.Timedelta(days=lag_days)
    # reindex to minute_idx with forward-fill
    d = d.reindex(minute_idx.union(d.index)).sort_index().ffill().reindex(minute_idx)
    return d


def assemble(cfg: dict | None = None, use_cache: bool = True) -> pd.DataFrame:
    cfg = cfg or load_cfg()
    start = cfg["window"]["start"]
    end = cfg["window"]["end"]
    sym = cfg["universe"]["symbol"]

    name = f"assembled_{sym}_{start}_{end}"
    if use_cache and cache.exists(name):
        log.info("assembled cache hit")
        return cache.load(name)

    # 1) Build the minute index from NYSE RTH schedule (UTC).
    minute_idx = rth_index(start, end, tz="UTC")
    log.info("minute index rows=%d", len(minute_idx))

    # 2) SPY minute bars.
    spy = bars.pull(sym, start, end)
    spy_cols = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
    out = _align_minute(spy, minute_idx, spy_cols, sym.lower())

    # 3) Cross-asset ETFs via Alpaca.
    for s in cfg["universe"]["cross_asset"]:
        if s in ALPACA_CROSS:
            try:
                df = bars.pull(s, start, end)
                out = out.join(_align_minute(df, minute_idx, ["close", "volume"], s.lower()))
            except Exception as e:
                log.warning("alpaca cross-asset failed for %s: %s", s, e)

    # 4) VIX / ES / DXY via yfinance (daily, forward-fill).
    for s, prefix in YF_DAILY_CROSS.items():
        if s in cfg["universe"]["cross_asset"]:
            try:
                df = yf_daily.pull(s, start, end)
                if not df.empty and "close" in df.columns:
                    ff = _daily_ffill(df[["close"]].rename(columns={"close": f"{prefix}_close"}), minute_idx, lag_days=0)
                    out = out.join(ff)
            except Exception as e:
                log.warning("yfinance failed for %s: %s", s, e)

    # 5) FRED macro (daily, 1-day lag).
    fred_df = fred.pull_many(cfg.get("fred_series", []), start, end)
    if not fred_df.empty:
        ff = _daily_ffill(fred_df, minute_idx, lag_days=1)
        ff.columns = [f"fred_{c}" for c in ff.columns]
        out = out.join(ff)

    # 6) Event flags.
    out["evt_fomc_day"] = is_fomc_day(minute_idx).astype(np.int8).values
    out["evt_zero_dte"] = is_zero_dte(minute_idx).astype(np.int8).values
    out["session_min"] = minutes_into_session(minute_idx).values

    # 7) Drop rows where SPY bars are missing (exchange was closed for a partial holiday etc).
    out = out.dropna(subset=[f"{sym.lower()}_close"])

    log.info("assembled shape=%s cols=%d", out.shape, out.shape[1])
    cache.save(out, name)
    return out


if __name__ == "__main__":
    df = assemble()
    print(df.tail())
    print("shape:", df.shape)
    print("columns:", list(df.columns))
