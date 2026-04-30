"""Pull daily series via yfinance for instruments Alpaca doesn't cover (VIX, ES=F, DX=F)."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from utils.logging import get
from . import cache

log = get("data.yf")


def pull(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    clean = symbol.replace("=", "").replace("^", "")
    name = f"yf_{clean}_{start}_{end}"
    if use_cache:
        cached = cache.load(name)
        if cached is not None:
            log.info("yf cache hit %s rows=%d", symbol, len(cached))
            return cached
    log.info("fetching yfinance daily %s %s..%s", symbol, start, end)
    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        log.warning("yfinance returned empty for %s", symbol)
        return pd.DataFrame()
    # flatten columns if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.rename(columns=str.lower)
    cache.save(df, name)
    return df
