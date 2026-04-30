"""Pull minute bars from Alpaca IEX feed for a list of symbols."""
from __future__ import annotations

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from utils.env import alpaca_keys
from utils.logging import get
from . import cache

log = get("data.bars")


def _client() -> StockHistoricalDataClient:
    k, s = alpaca_keys()
    return StockHistoricalDataClient(k, s)


def pull(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Return DataFrame indexed by UTC minute timestamps with open/high/low/close/volume/vwap/trade_count."""
    name = f"bars_{symbol.replace('=', '').replace('^', '')}_{start}_{end}"
    if use_cache:
        cached = cache.load(name)
        if cached is not None:
            log.info("bars cache hit %s rows=%d", symbol, len(cached))
            return cached

    log.info("fetching Alpaca bars %s %s..%s", symbol, start, end)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        feed=DataFeed.IEX,
        adjustment="all",
    )
    bars = _client().get_stock_bars(req)
    df = bars.df
    if df.empty:
        log.warning("no bars returned for %s", symbol)
        return df
    # MultiIndex (symbol, timestamp) -> just timestamp
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=[c for c in ("symbol",) if c in df.columns])
    cache.save(df, name)
    log.info("saved %s rows=%d", symbol, len(df))
    return df


def pull_many(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    return {s: pull(s, start, end) for s in symbols}
