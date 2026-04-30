"""Feature builder.

Pure function: assembled minute DataFrame -> feature DataFrame.
Same code is used in training and in the live loop to prevent train/serve skew.

No lookahead: every feature at index t uses only data observed at or before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _target_prefix(cfg: dict) -> str:
    """Column-name prefix for the target instrument's bar columns."""
    sym = cfg.get("universe", {}).get("symbol", "SPY")
    return sym.lower()


def _log_ret(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _bollinger_pctb(close: pd.Series, window: int, k: float = 2.0) -> pd.Series:
    m = close.rolling(window).mean()
    s = close.rolling(window).std()
    upper = m + k * s
    lower = m - k * s
    denom = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / denom


def _session_vwap(df: pd.DataFrame, target: str) -> pd.Series:
    """Session-resetting cumulative VWAP using target's typical price."""
    tp = (df[f"{target}_high"] + df[f"{target}_low"] + df[f"{target}_close"]) / 3.0
    vol = df[f"{target}_volume"].fillna(0.0)
    date = df.index.tz_convert("America/New_York").date
    grp = pd.Series(date, index=df.index)
    pv = (tp * vol).groupby(grp).cumsum()
    v = vol.groupby(grp).cumsum().replace(0.0, np.nan)
    return pv / v


def build(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute features on an assembled minute DataFrame.

    Input: the output of data_pull.assemble.assemble().
    Output: DataFrame aligned to df.index with feature columns. NaNs allowed for warmup rows.
    """
    fcfg = cfg["features"]
    target = _target_prefix(cfg)
    close = df[f"{target}_close"]
    open_ = df[f"{target}_open"]
    high = df[f"{target}_high"]
    low = df[f"{target}_low"]
    volume = df[f"{target}_volume"].fillna(0.0)

    out = pd.DataFrame(index=df.index)

    # Lagged log returns.
    for n in fcfg["return_lags"]:
        out[f"ret_{n}"] = _log_ret(close, n)

    # Realized vol (sum of squared 1-min log returns, rolled).
    r1 = out["ret_1"]
    for w in fcfg["vol_windows"]:
        out[f"rvol_{w}"] = np.sqrt((r1 * r1).rolling(w).sum())

    # RSI.
    out[f"rsi_{fcfg['rsi_window']}"] = _rsi(close, fcfg["rsi_window"])

    # MACD.
    f, s, sig = fcfg["macd"]
    macd, macd_sig, macd_hist = _macd(close, f, s, sig)
    out["macd"] = macd
    out["macd_sig"] = macd_sig
    out["macd_hist"] = macd_hist

    # Bollinger %B.
    out[f"bb_pctb_{fcfg['bollinger_window']}"] = _bollinger_pctb(close, fcfg["bollinger_window"])

    # VWAP deviation.
    vwap = _session_vwap(df, target)
    out["vwap_dev"] = (close - vwap) / vwap
    out[f"vwap_dev_roll_{fcfg['vwap_window']}"] = out["vwap_dev"].rolling(fcfg["vwap_window"]).mean()

    # Return autocorrelation over last 60 min.
    out["ret_ac_60"] = r1.rolling(60).corr(r1.shift(1))

    # --- Regime features ---
    # Use local-ET date as session key. A "session" is one trading day.
    et_idx = df.index.tz_convert("America/New_York")
    date_key = pd.Series(et_idx.date, index=df.index)
    session_min_series = df["session_min"].astype(float)
    # Minute-of-day (ET) — used by percentile/vol-z features below.
    minute_of_day = et_idx.hour * 60 + et_idx.minute
    mod = pd.Series(minute_of_day, index=df.index)

    # Per-day session open (first bar's open) and prior-day close (last bar's close).
    day_first_open = df[f"{target}_open"].groupby(date_key).transform("first")
    day_last_close = df[f"{target}_close"].groupby(date_key).transform("last")
    # Prior trading day's close: take the unique per-day close, shift by 1, broadcast back.
    daily_close = df[f"{target}_close"].groupby(date_key).last()
    prev_daily_close = daily_close.shift(1)
    prev_close_broadcast = date_key.map(prev_daily_close)
    out["overnight_gap"] = (day_first_open - prev_close_broadcast) / prev_close_broadcast

    # Opening range (first 15 min of session): high, low, and where current price sits within it.
    # Use session_min thresholds; NYSE minute bars are end-labelled so the first bar has session_min==1.
    opening_mask = (session_min_series >= 0) & (session_min_series <= 16)
    open_range_high = df[f"{target}_high"].where(opening_mask).groupby(date_key).cummax()
    open_range_low = df[f"{target}_low"].where(opening_mask).groupby(date_key).cummin()
    # Forward-fill within session so later bars still know the opening range.
    open_range_high = open_range_high.groupby(date_key).ffill()
    open_range_low = open_range_low.groupby(date_key).ffill()
    or_range = (open_range_high - open_range_low).replace(0.0, np.nan)
    out["or_pos"] = (close - open_range_low) / or_range       # 0=bottom of range, 1=top, <0 or >1 = breakout
    out["or_width_pct"] = or_range / close                    # opening-range width as % of price

    # Realized-vol percentile: rank rvol_60 vs past 20 sessions (same-minute buckets avoid stale comparisons).
    rv60 = out["rvol_60"]
    out["rvol_60_pct20d"] = rv60.groupby(mod).transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).rank(pct=True)
    )

    # Trend strength proxy: |30-min return| / 30-min realized vol (momentum-to-noise ratio).
    rvol_30 = np.sqrt((r1 * r1).rolling(30).sum())
    out["trend_strength_30"] = out["ret_30"].abs() / rvol_30.replace(0.0, np.nan)

    # Higher-timeframe trend: 60-min vs 240-min MA diff.
    ma60 = close.rolling(60).mean()
    ma240 = close.rolling(240).mean()
    out["ma_diff_60_240"] = (ma60 - ma240) / ma240

    # Signed-volume proxy (bar-level, since we don't have tick-level trades in v1).
    out["signed_vol_proxy"] = np.sign(close - open_) * volume
    out["signed_vol_proxy_z"] = (
        out["signed_vol_proxy"]
        - out["signed_vol_proxy"].rolling(60).mean()
    ) / out["signed_vol_proxy"].rolling(60).std()

    # Volume z-score vs same minute-of-day over prior N sessions.
    nsess = fcfg["vol_zscore_window"]
    # For each minute-of-day bucket, compute a rolling mean/std of past same-minute volumes.
    def _z(grp):
        mean = grp.shift(1).rolling(nsess, min_periods=5).mean()
        std = grp.shift(1).rolling(nsess, min_periods=5).std()
        return (grp - mean) / std
    out["vol_z_mod"] = volume.groupby(mod).transform(_z)

    # Cross-asset minute-level 5-min log returns for Alpaca ETFs.
    for col in df.columns:
        if col.endswith("_close") and not col.startswith(target) and not col.startswith(("vix", "es", "dxy")):
            prefix = col[:-len("_close")]
            out[f"{prefix}_ret_5"] = np.log(df[col] / df[col].shift(5))

    # Sector breadth: count of XL* with positive 5-min return.
    sector_cols = [c for c in out.columns if c.startswith("xl") and c.endswith("_ret_5")]
    if sector_cols:
        out["sector_breadth_5"] = (out[sector_cols] > 0).sum(axis=1)

    # Cross-asset daily deltas (already lagged 1 day in assemble for FRED/VIX).
    for prefix in ("vix", "es", "dxy"):
        col = f"{prefix}_close"
        if col in df.columns:
            out[f"{prefix}_chg_1d"] = df[col].pct_change(390)  # ~1 trading day of minutes
            out[f"{prefix}_level"] = df[col]

    # FRED macro levels + 20-day changes.
    for col in df.columns:
        if col.startswith("fred_"):
            out[col] = df[col]
            out[f"{col}_chg20d"] = df[col] - df[col].shift(20 * 390)

    # Event flags + session timing.
    out["evt_fomc_day"] = df["evt_fomc_day"].astype(float)
    out["evt_zero_dte"] = df["evt_zero_dte"].astype(float)
    session_min = df["session_min"].astype(float)
    out["session_min"] = session_min
    # Cyclical encoding (daily session length = 390 min).
    theta = 2.0 * np.pi * session_min.clip(lower=0) / 390.0
    out["session_sin"] = np.sin(theta)
    out["session_cos"] = np.cos(theta)
    out["lunch_lull"] = ((session_min >= 150) & (session_min <= 240)).astype(float)

    # Day of week one-hot.
    dow = df.index.tz_convert("America/New_York").dayofweek
    for d in range(5):
        out[f"dow_{d}"] = (dow == d).astype(float)

    return out


def feature_columns(df_features: pd.DataFrame) -> list[str]:
    """Canonical feature ordering for train/serve consistency."""
    return sorted(df_features.columns.tolist())
