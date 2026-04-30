from __future__ import annotations

import numpy as np
import pandas as pd


def round_trip_cost_bp(cfg: dict) -> float:
    c = cfg["costs"]
    return 2.0 * (c["half_spread_bp"] + c["slippage_bp"])


def net_pnl_per_trade(signal: pd.Series, fwd_ret: pd.Series, cfg: dict) -> pd.Series:
    """Per-bar net return assuming a single `horizon_min` round-trip when |signal| == 1."""
    cost = round_trip_cost_bp(cfg) / 1e4
    return signal * fwd_ret - np.abs(signal) * cost


def disjoint_sample(series: pd.Series, step: int) -> pd.Series:
    """Sample one bar every `step` bars to avoid overlapping-holding contamination in Sharpe."""
    return series.iloc[::step]


def annualized_sharpe(r: pd.Series, periods_per_year: float) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def compute_metrics(signal: pd.Series, fwd_ret: pd.Series, cfg: dict) -> dict:
    horizon = cfg["label"]["horizon_min"]
    pnl = net_pnl_per_trade(signal, fwd_ret, cfg)
    # Disjoint non-overlapping trades: one every horizon minutes.
    disjoint = disjoint_sample(pnl, horizon).dropna()
    # ~390 RTH min / day / horizon = trades per day, 252 days per year.
    periods_per_year = (390 / horizon) * 252
    trades = int((signal != 0).sum())
    disjoint_trades = int((disjoint != 0).sum())
    return {
        "net_pnl_sum_bp": float(pnl.sum() * 1e4),
        "disjoint_sharpe": annualized_sharpe(disjoint, periods_per_year),
        "avg_trade_bp": float(pnl[signal != 0].mean() * 1e4) if trades else float("nan"),
        "hit_rate": float((pnl[signal != 0] > 0).mean()) if trades else float("nan"),
        "trades": trades,
        "disjoint_trades": disjoint_trades,
        "turnover_per_day": trades / max(1, len(signal) / 390),
        "max_dd_bp": float(_max_drawdown(pnl) * 1e4),
    }


def _max_drawdown(r: pd.Series) -> float:
    eq = r.fillna(0).cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min()) if len(dd) else 0.0
