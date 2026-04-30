"""Pull daily macro series from FRED using the REST API directly.

We bypass fredapi because it uses urllib with the system SSL trust store,
which fails on Python.org builds of macOS. `requests` ships with certifi.
"""
from __future__ import annotations

import pandas as pd
import requests

from utils.env import fred_key
from utils.logging import get
from . import cache

log = get("data.fred")

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def pull(series_id: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    name = f"fred_{series_id}_{start}_{end}"
    if use_cache:
        cached = cache.load(name)
        if cached is not None:
            return cached.iloc[:, 0]
    key = fred_key()
    if not key:
        log.warning("FRED_API_KEY missing, skipping %s", series_id)
        return pd.Series(dtype=float, name=series_id)
    log.info("fetching FRED %s %s..%s", series_id, start, end)
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    r = requests.get(FRED_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
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
    series = [pull(i, start, end) for i in ids]
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)
