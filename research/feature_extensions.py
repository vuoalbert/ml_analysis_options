"""Volume-profile and multi-timeframe features for the v3 entry-model
combination experiment.

Both feature families are computed from the same OHLCV bars the existing
feature builder uses — no new data sources, no Level 2.

  • add_volume_features()  → 5 columns:
      vp_value_pct_60        — current close's percentile within 60-min range
      vp_volume_imbalance_15 — sum(vol where close>open) / sum(vol) last 15m
      vp_signed_vol_ratio_30 — sum(signed_vol) / sum(abs_vol) last 30m
      vp_high_vol_dist       — distance from current close to highest-volume
                                bar's close in last 60m, in bps
      vp_spread_proxy_5      — avg (high-low)/close over last 5m

  • add_mtf_features()  → 5 columns:
      mtf_ret_60min          — log return over last 60 min
      mtf_ret_4h             — log return over last 240 min
      mtf_intraday_pct       — current close vs today's session range
      mtf_overnight_change   — current close vs prior session's last close
      mtf_5d_return          — log return vs 5 trading days ago (~1950 min)

Both functions take an OHLCV dataframe with `<sym>_close, <sym>_high,
<sym>_low, <sym>_volume` columns and return a dataframe of NEW features
indexed the same way. The caller joins them onto the standard feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_features(df: pd.DataFrame, sym: str = "spy") -> pd.DataFrame:
    """Compute volume-profile features from minute bars."""
    close = df[f"{sym}_close"].astype(float)
    high  = df[f"{sym}_high"].astype(float)
    low   = df[f"{sym}_low"].astype(float)
    open_ = df[f"{sym}_open"].astype(float)
    vol   = df[f"{sym}_volume"].astype(float).fillna(0.0)

    out = pd.DataFrame(index=df.index)

    # 1. Percentile of current close within last 60 min range
    rolling_min = low.rolling(60, min_periods=10).min()
    rolling_max = high.rolling(60, min_periods=10).max()
    rng = (rolling_max - rolling_min).replace(0, np.nan)
    out["vp_value_pct_60"] = (close - rolling_min) / rng

    # 2. Volume imbalance: bullish-bar volume / total volume over last 15m
    bullish_vol = vol.where(close > open_, 0.0)
    out["vp_volume_imbalance_15"] = (
        bullish_vol.rolling(15, min_periods=5).sum()
        / vol.rolling(15, min_periods=5).sum().replace(0, np.nan)
    )

    # 3. Signed volume ratio over 30 min (-1 to 1, like an order-flow proxy)
    signed = np.sign(close - open_) * vol
    out["vp_signed_vol_ratio_30"] = (
        signed.rolling(30, min_periods=10).sum()
        / vol.rolling(30, min_periods=10).sum().replace(0, np.nan)
    )

    # 4. Distance from current close to the close of the highest-volume bar
    #    in the last 60 min (bps). Uses argmax of rolling volume.
    def _high_vol_close_dist(window_close, window_vol):
        # vol/close arrays of same length (60); return bps from current_close
        # to close at argmax(vol)
        if len(window_vol) < 2:
            return np.nan
        idx = int(np.argmax(window_vol))
        return (window_close.iloc[-1] / window_close.iloc[idx] - 1) * 1e4

    # Vectorize via numpy stride trick — keep it simple with apply
    # This is O(N × 60); fine for one-time feature build
    high_vol_dist = []
    for i in range(len(close)):
        lo, hi = max(0, i - 59), i + 1
        cw = close.iloc[lo:hi]
        vw = vol.iloc[lo:hi]
        high_vol_dist.append(_high_vol_close_dist(cw, vw))
    out["vp_high_vol_dist"] = pd.Series(high_vol_dist, index=df.index)

    # 5. Spread proxy: avg (high-low)/close over last 5 min, in bps
    spread = (high - low) / close * 1e4
    out["vp_spread_proxy_5"] = spread.rolling(5, min_periods=2).mean()

    return out


def add_mtf_features(df: pd.DataFrame, sym: str = "spy") -> pd.DataFrame:
    """Compute multi-timeframe features from minute bars."""
    close = df[f"{sym}_close"].astype(float)
    log_close = np.log(close)

    out = pd.DataFrame(index=df.index)

    # 1. Last-60-min return
    out["mtf_ret_60min"] = log_close.diff(60)

    # 2. Last-4h return
    out["mtf_ret_4h"] = log_close.diff(240)

    # 3. Intraday percentile — where in TODAY's range we are
    et_date = pd.Series(df.index, index=df.index).dt.tz_convert("America/New_York").dt.date
    daily_low = close.groupby(et_date).cummin()
    daily_high = close.groupby(et_date).cummax()
    rng = (daily_high - daily_low).replace(0, np.nan)
    out["mtf_intraday_pct"] = (close - daily_low) / rng

    # 4. Overnight change — close vs prior session's last close
    # Compute last close of each ET date, then forward-fill at session opens
    last_close = close.groupby(et_date).transform("last")
    last_close_lagged = last_close.shift(1).groupby(et_date).transform("first")
    # When session_min reset happens (start of new day), close vs prev day's last
    # Simpler proxy: log(close / first_close_of_day) won't work — what we want is
    # the diff between current bar's close and the previous DAY's last close
    prev_day_close = pd.Series(np.nan, index=df.index)
    for d, group in close.groupby(et_date):
        # find the prior trading day's last close
        prior_dates = [pd for pd in close.groupby(et_date).groups.keys() if pd < d]
        if prior_dates:
            prior_last = close.groupby(et_date).get_group(max(prior_dates)).iloc[-1]
            prev_day_close.loc[group.index] = prior_last
    out["mtf_overnight_change"] = np.log(close / prev_day_close)

    # 5. 5-day return — close vs 5 trading days ago. ~390 mins/day × 5 = 1950 mins.
    out["mtf_5d_return"] = log_close.diff(1950)

    return out


def add_extensions(feats: pd.DataFrame, raw_bars: pd.DataFrame, sym: str,
                    *, add_volume: bool = False, add_mtf: bool = False) -> pd.DataFrame:
    """Join optional volume + MTF features onto an existing feature dataframe.

    raw_bars must have <sym>_close/high/low/open/volume columns and the same
    minute-resolution index that feats was computed on.
    """
    out = feats
    if add_volume:
        vf = add_volume_features(raw_bars, sym=sym)
        out = out.join(vf, how="left")
    if add_mtf:
        mf = add_mtf_features(raw_bars, sym=sym)
        out = out.join(mf, how="left")
    return out
