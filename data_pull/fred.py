"""Pull daily macro series from FRED using the REST API directly.

We bypass fredapi because it uses urllib with the system SSL trust store,
which fails on Python.org builds of macOS. `requests` ships with certifi.

Resilience: if FRED is down (5xx), times out, or returns garbage, fall back
to the most recent cached parquet for that series_id. Macro features are
1-day-lagged in the assemble pipeline, so yesterday's value is a perfect
substitute for today's during a transient outage — the model trained
on exactly this lag regime.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from utils.env import fred_key
from utils.logging import get
from . import cache

log = get("data.fred")

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def _stale_cache_fallback(series_id: str) -> pd.Series:
    """Find the most recent cache/fred_{series_id}_*.parquet and return its series.

    Returns an empty Series if no cached parquet exists at all. Logs which file
    it picked so live diagnostics can flag stale-cache fallback usage.
    """
    cache_root = cache.cache_dir()
    candidates = sorted(cache_root.glob(f"fred_{series_id}_*.parquet"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        log.warning("FRED %s: no stale cache available, returning empty series", series_id)
        return pd.Series(dtype=float, name=series_id)
    newest = candidates[0]
    log.warning("FRED %s: falling back to stale cache %s", series_id, newest.name)
    df = pd.read_parquet(newest)
    return df.iloc[:, 0]


def pull(series_id: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    name = f"fred_{series_id}_{start}_{end}"
    if use_cache:
        cached = cache.load(name)
        if cached is not None:
            return cached.iloc[:, 0]
    key = fred_key()
    if not key:
        log.warning("FRED_API_KEY missing, skipping %s", series_id)
        return _stale_cache_fallback(series_id)
    log.info("fetching FRED %s %s..%s", series_id, start, end)
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    try:
        r = requests.get(FRED_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("observations", [])
    except (requests.exceptions.RequestException, ValueError) as e:
        # 5xx, 4xx, timeout, JSON decode error, etc. — all handled the same way.
        log.warning("FRED %s fetch failed (%s); using stale cache", series_id, e)
        return _stale_cache_fallback(series_id)

    if not data:
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # "." means missing in FRED.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df["value"].rename(series_id)
    s.index = s.index.tz_localize("UTC")
    cache.save(s.to_frame(), name)
    return s


def pull_many(ids: list[str], start: str, end: str) -> pd.DataFrame:
    """Resilient bulk pull — one bad series doesn't kill the others.

    Each series goes through `pull()` which has its own stale-cache fallback.
    Any unexpected exception above that layer is caught here so the caller
    always gets a DataFrame (possibly with fewer columns than requested).
    """
    series = []
    for i in ids:
        try:
            s = pull(i, start, end)
            if not s.empty:
                series.append(s)
        except Exception as e:
            log.warning("FRED %s: unexpected error %s — skipping", i, e)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)
