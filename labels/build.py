"""Label builder.

At time t: y = sign(log(close[t+H]/close[t]) - dead_zone)
Three classes: 0 = down, 1 = flat, 2 = up. Flat = within dead-zone.

The dead-zone (in bp) prevents the model from learning edges smaller than trading cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _target_close(df: pd.DataFrame, cfg: dict) -> pd.Series:
    sym = cfg.get("universe", {}).get("symbol", "SPY").lower()
    col = f"{sym}_close"
    if col not in df.columns:
        # Cross-sectional path renames target columns to spy_close.
        col = "spy_close"
    return df[col]


def build(df: pd.DataFrame, cfg: dict) -> pd.Series:
    horizon = cfg["label"]["horizon_min"]
    dead_zone_bp = cfg["label"]["dead_zone_bp"]
    dz = dead_zone_bp / 1e4  # bp -> fraction

    close = _target_close(df, cfg)
    fwd = np.log(close.shift(-horizon) / close)

    y = pd.Series(1, index=df.index, dtype=np.int8)  # default flat
    y[fwd > dz] = 2
    y[fwd < -dz] = 0
    # Future is unknown for the last `horizon` rows.
    y.iloc[-horizon:] = -1
    return y


def forward_return(df: pd.DataFrame, horizon: int, cfg: dict | None = None) -> pd.Series:
    if cfg is not None:
        close = _target_close(df, cfg)
    else:
        # Backward-compat: callers that pre-date the cfg arg pass spy_close.
        close = df.get("spy_close", df.iloc[:, 0])
    return np.log(close.shift(-horizon) / close)
