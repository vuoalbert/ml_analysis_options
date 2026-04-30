"""Dashboard data layer.

Fetchers, prediction replayer, and order-history helpers. All functions are
side-effect-free and idempotent; Streamlit caches them.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, ClosePositionRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# Timeframe options exposed in the dashboard's selector.
TIMEFRAMES: dict[str, TimeFrame] = {
    "1m":  TimeFrame(1, TimeFrameUnit.Minute),
    "5m":  TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1, TimeFrameUnit.Hour),
    "1d":  TimeFrame(1, TimeFrameUnit.Day),
}

from utils.env import alpaca_keys
from utils.config import load as load_cfg
from utils.calendar import NYSE, is_fomc_day, is_zero_dte, minutes_into_session
from features.build import build as build_features
from model.artifact import load as load_artifact
from data_pull import yf_daily, fred
from data_pull.assemble import (YF_DAILY_CROSS, ALPACA_CROSS, _daily_ffill, _align_minute)


# ---- clients ----

@st.cache_resource
def trading_client() -> TradingClient:
    k, s = alpaca_keys()
    return TradingClient(k, s, paper=True)


@st.cache_resource
def data_client() -> StockHistoricalDataClient:
    k, s = alpaca_keys()
    return StockHistoricalDataClient(k, s)


@st.cache_resource
def artifact():
    return load_artifact("latest")


@st.cache_resource
def cfg():
    return load_cfg("v1")


# ---- account ----

def account_info() -> dict:
    try:
        a = trading_client().get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "daytrade_count": int(getattr(a, "daytrade_count", 0) or 0),
        }
    except Exception as e:
        return {"error": str(e)}


def current_position(symbol: str) -> dict | None:
    try:
        for p in trading_client().get_all_positions():
            if p.symbol == symbol:
                return {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
    except Exception:
        return None
    return None


def all_positions() -> list:
    """Return raw position objects (Alpaca SDK) for all open positions.

    Used by the dashboard to count concurrent option contracts in options-mode.
    """
    try:
        return list(trading_client().get_all_positions())
    except Exception:
        return []


# ---- orders / fills ----

def recent_fills(lookback_hours: int = 8, symbol: str | None = None) -> pd.DataFrame:
    """Return filled orders in the last N hours as a tidy DataFrame."""
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=start.to_pydatetime(),
            limit=500,
            symbols=[symbol] if symbol else None,
        )
        orders = trading_client().get_orders(filter=req)
    except Exception as e:
        return pd.DataFrame()

    rows = []
    for o in orders:
        if str(o.status).lower().endswith("filled"):
            rows.append({
                "submitted_at": pd.to_datetime(o.submitted_at, utc=True),
                "filled_at": pd.to_datetime(o.filled_at, utc=True) if o.filled_at else pd.NaT,
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "qty": float(o.qty),
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else float("nan"),
                "order_id": str(o.id),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("filled_at").reset_index(drop=True)
    return df


def pair_entries_exits(fills: pd.DataFrame) -> pd.DataFrame:
    """Pair each entry fill with its next opposite-side fill as the exit.

    Assumes one-position-at-a-time (matches our live policy). Any unmatched
    entry at the end is kept as 'open'.
    """
    if fills.empty:
        return pd.DataFrame()
    pairs = []
    stack = []
    for _, f in fills.iterrows():
        if not stack:
            stack.append(f)
            continue
        last = stack[-1]
        if f["side"] != last["side"]:
            # This is an exit for the open position.
            pnl_per_share = (f["filled_avg_price"] - last["filled_avg_price"])
            if last["side"] == "sell":
                pnl_per_share = -pnl_per_share
            qty = min(last["filled_qty"], f["filled_qty"])
            pnl_dollars = pnl_per_share * qty
            pnl_bp = (pnl_per_share / last["filled_avg_price"]) * 1e4
            pairs.append({
                "entry_at": last["filled_at"],
                "exit_at": f["filled_at"],
                "symbol": last["symbol"],
                "side": "long" if last["side"] == "buy" else "short",
                "entry_px": last["filled_avg_price"],
                "exit_px": f["filled_avg_price"],
                "qty": qty,
                "pnl_dollars": pnl_dollars,
                "pnl_bp": pnl_bp,
            })
            stack.pop()
        else:
            stack.append(f)
    df = pd.DataFrame(pairs)
    return df


# ---- bars ----

@st.cache_data(ttl=60, show_spinner=False)
def fetch_bars(symbol: str, hours: int = 4, timeframe: str = "1m") -> pd.DataFrame:
    """Fetch OHLCV bars for `symbol` going `hours` back at the requested timeframe.

    `timeframe` is one of TIMEFRAMES.keys() — '1m', '5m', '15m', '1h', '1d'.
    Defaults to '1m' for backwards-compat with callers that don't specify.
    """
    tf = TIMEFRAMES.get(timeframe, TimeFrame.Minute)
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=hours)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment="all",
    )
    df = data_client().get_stock_bars(req).df
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=[c for c in ("symbol",) if c in df.columns])
    return df


# ---- prediction replay ----

@st.cache_data(ttl=60, show_spinner="Recomputing model predictions…")
def replay_predictions(hours: int = 4) -> pd.DataFrame:
    """Build features + predict for the last N hours so the signal panel matches live behaviour.

    Returns DataFrame indexed by UTC minute with columns: p_down, p_flat, p_up, close.
    """
    c = cfg()
    art = artifact()

    now = pd.Timestamp.now(tz="UTC")
    # Need enough warmup for feature lookbacks. 4 trading days = 1560 min ≈ safe.
    start = now - pd.Timedelta(days=8)

    sched = NYSE.schedule(start_date=start.date(), end_date=now.date())
    if sched.empty:
        return pd.DataFrame()
    import pandas_market_calendars as mcal
    minute_idx = mcal.date_range(sched, frequency="1min").tz_convert("UTC")
    minute_idx = minute_idx[minute_idx <= now]

    sym = c["universe"]["symbol"]
    spy = fetch_bars(sym, hours=int((now - start).total_seconds() // 3600) + 1)
    cols = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
    out = _align_minute(spy, minute_idx, cols, sym.lower())

    for s in c["universe"]["cross_asset"]:
        if s in ALPACA_CROSS:
            try:
                df = fetch_bars(s, hours=int((now - start).total_seconds() // 3600) + 1)
                out = out.join(_align_minute(df, minute_idx, ["close", "volume"], s.lower()))
            except Exception:
                pass

    for s, prefix in YF_DAILY_CROSS.items():
        if s in c["universe"]["cross_asset"]:
            try:
                df = yf_daily.pull(s, str((now - pd.Timedelta(days=120)).date()),
                                   str(now.date()), use_cache=False)
                if not df.empty and "close" in df.columns:
                    ff = _daily_ffill(df[["close"]].rename(columns={"close": f"{prefix}_close"}),
                                      minute_idx, lag_days=0)
                    out = out.join(ff)
            except Exception:
                pass

    fred_df = fred.pull_many(c.get("fred_series", []),
                             str((now - pd.Timedelta(days=120)).date()), str(now.date()))
    if not fred_df.empty:
        ff = _daily_ffill(fred_df, minute_idx, lag_days=1)
        ff.columns = [f"fred_{col}" for col in ff.columns]
        out = out.join(ff)

    out["evt_fomc_day"] = is_fomc_day(minute_idx).astype(int).values
    out["evt_zero_dte"] = is_zero_dte(minute_idx).astype(int).values
    out["session_min"] = minutes_into_session(minute_idx).values
    out = out.dropna(subset=[f"{sym.lower()}_close"])

    feats = build_features(out, c)
    # Align to training schema.
    for col in art.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[art.feature_cols]

    # Trim to only rows with essential features.
    essential = [col for col in art.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    mask = feats[essential].notna().all(axis=1)
    feats = feats[mask]
    if feats.empty:
        return pd.DataFrame()

    proba = art.booster.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    pred["close"] = out.loc[pred.index, f"{sym.lower()}_close"]

    # Keep only the recent window the user requested.
    cutoff = now - pd.Timedelta(hours=hours)
    pred = pred[pred.index >= cutoff]
    return pred


# ---- SHAP for current prediction ----

@st.cache_data(ttl=60, show_spinner=False)
def shap_explain_latest() -> pd.DataFrame | None:
    """SHAP values for the most recent row. Returns top-k contributors for p_up."""
    import shap
    art = artifact()
    pred = replay_predictions(hours=1)
    if pred.empty:
        return None
    # Rebuild features for the very last row only.
    c = cfg()
    now = pd.Timestamp.now(tz="UTC")
    sym = c["universe"]["symbol"]
    sched = NYSE.schedule(start_date=(now - pd.Timedelta(days=8)).date(), end_date=now.date())
    import pandas_market_calendars as mcal
    minute_idx = mcal.date_range(sched, frequency="1min").tz_convert("UTC")
    minute_idx = minute_idx[minute_idx <= now]
    spy = fetch_bars(sym, hours=200)
    cols = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
    out = _align_minute(spy, minute_idx, cols, sym.lower())
    for s in c["universe"]["cross_asset"]:
        if s in ALPACA_CROSS:
            try:
                df = fetch_bars(s, hours=200)
                out = out.join(_align_minute(df, minute_idx, ["close", "volume"], s.lower()))
            except Exception:
                pass
    out["evt_fomc_day"] = is_fomc_day(minute_idx).astype(int).values
    out["evt_zero_dte"] = is_zero_dte(minute_idx).astype(int).values
    out["session_min"] = minutes_into_session(minute_idx).values
    out = out.dropna(subset=[f"{sym.lower()}_close"])
    feats = build_features(out, c)
    for col in art.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[art.feature_cols]
    essential = [col for col in art.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    mask = feats[essential].notna().all(axis=1)
    feats = feats[mask]
    if feats.empty:
        return None
    last = feats.iloc[[-1]]
    explainer = shap.TreeExplainer(art.booster)
    sv = explainer.shap_values(last)
    # LightGBM multiclass: list of arrays per class. We want class 2 (p_up).
    if isinstance(sv, list):
        sv_up = sv[2][0]  # class=2 (up), first (only) row
    else:
        # Newer SHAP versions return (n, n_features, n_classes).
        sv_up = sv[0, :, 2]
    df = pd.DataFrame({
        "feature": art.feature_cols,
        "value": last.iloc[0].values,
        "shap_up": sv_up,
    })
    df["abs"] = df["shap_up"].abs()
    return df.sort_values("abs", ascending=False).head(12).reset_index(drop=True)


# ---- calibration ----

@st.cache_data(ttl=60, show_spinner=False)
def calibration_from_holdout() -> pd.DataFrame | None:
    """Read the untouched-holdout calibration from artifact meta if present."""
    art = artifact()
    return pd.DataFrame(art.metrics.get("walk_forward", []))


def perf_stats(trades: pd.DataFrame) -> dict:
    """Institutional performance statistics from a trades dataframe.

    Returns: dict with sharpe, profit_factor, hit_rate, avg_trade, best_trade,
    worst_trade, max_dd, win_loss_ratio, avg_winner, avg_loser, total_pnl,
    n_trades, days_active.
    """
    if trades is None or trades.empty:
        return {"n_trades": 0}

    t = trades.copy()
    t["exit_at"] = pd.to_datetime(t["exit_at"])
    t["et_date"] = t["exit_at"].dt.tz_convert("America/New_York").dt.date

    n = len(t)
    pnl = t["pnl_dollars"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    daily_pnl = t.groupby("et_date")["pnl_dollars"].sum()
    days_active = len(daily_pnl)

    daily_sharpe = (
        daily_pnl.mean() / daily_pnl.std()
        if days_active > 1 and daily_pnl.std() > 0 else float("nan")
    )
    sharpe_annualized = daily_sharpe * (252 ** 0.5) if days_active > 1 else float("nan")

    cum = daily_pnl.cumsum()
    max_dd = float((cum - cum.cummax()).min()) if len(cum) > 0 else 0

    profit_factor = (wins.sum() / abs(losses.sum())
                      if len(losses) > 0 and losses.sum() < 0 else float("inf"))

    win_loss_ratio = (
        wins.mean() / abs(losses.mean())
        if len(wins) > 0 and len(losses) > 0 else float("nan")
    )

    return {
        "n_trades": n,
        "hit_rate": len(wins) / n,
        "total_pnl": float(pnl.sum()),
        "avg_trade": float(pnl.mean()),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "avg_winner": float(wins.mean()) if len(wins) else 0,
        "avg_loser": float(losses.mean()) if len(losses) else 0,
        "win_loss_ratio": float(win_loss_ratio),
        "profit_factor": float(profit_factor),
        "max_dd": max_dd,
        "daily_sharpe": float(daily_sharpe) if daily_sharpe == daily_sharpe else float("nan"),
        "sharpe_annualized": float(sharpe_annualized) if sharpe_annualized == sharpe_annualized else float("nan"),
        "days_active": days_active,
        "avg_per_day": float(daily_pnl.mean()) if days_active > 0 else 0,
    }


def close_position_now(symbol: str) -> str:
    try:
        r = trading_client().close_position(symbol)
        return f"closed: {getattr(r, 'id', 'ok')}"
    except Exception as e:
        return f"error: {e}"
