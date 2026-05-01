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
            "last_equity": float(a.last_equity) if a.last_equity else 0.0,  # yesterday's close
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

def _is_occ(sym: str) -> bool:
    """OCC option symbols are >= 15 chars and end in 8-digit strike."""
    return isinstance(sym, str) and len(sym) >= 15 and sym[-8:].isdigit()


def _parse_occ(sym: str) -> dict:
    """Parse SPY260511C00712000 → {root, expiry, side, strike}."""
    try:
        root = sym[:-15].strip() or sym[:6].strip()
        date_part = sym[-15:-9]
        cp = sym[-9]
        strike_int = int(sym[-8:])
        return {
            "root": root,
            "expiry": f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:6]}",
            "opt_side": "call" if cp == "C" else "put",
            "strike": strike_int / 1000.0,
        }
    except Exception:
        return {"root": sym, "expiry": "—", "opt_side": "?", "strike": 0.0}


def recent_fills(lookback_hours: int = 8, symbol: str | None = None) -> pd.DataFrame:
    """Return filled orders in the last N hours as a tidy DataFrame.

    When `symbol` is given (e.g. "SPY"), this returns BOTH equity fills AND
    option fills whose underlying matches (i.e. OCC symbols starting with
    "SPY"). Pull is unfiltered then post-filtered by prefix, because Alpaca's
    symbols filter does an exact match and would exclude OCC symbols.
    """
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=start.to_pydatetime(),
            limit=500,
        )
        orders = trading_client().get_orders(filter=req)
    except Exception as e:
        return pd.DataFrame()

    rows = []
    for o in orders:
        if not str(o.status).lower().endswith("filled"):
            continue
        sym = o.symbol
        if symbol is not None:
            # Match equity (exact) OR option (prefix match on root)
            if sym != symbol and not (
                _is_occ(sym) and sym.startswith(symbol[:6].strip())
            ):
                continue
        is_option = _is_occ(sym)
        meta = _parse_occ(sym) if is_option else {
            "root": sym, "expiry": None, "opt_side": None, "strike": None,
        }
        rows.append({
            "submitted_at": pd.to_datetime(o.submitted_at, utc=True),
            "filled_at": pd.to_datetime(o.filled_at, utc=True) if o.filled_at else pd.NaT,
            "symbol": sym,
            "side": o.side.value if hasattr(o.side, "value") else str(o.side),
            "qty": float(o.qty),
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else float("nan"),
            "order_id": str(o.id),
            "is_option": is_option,
            "underlying": meta["root"],
            "expiry": meta["expiry"],
            "opt_side": meta["opt_side"],
            "strike": meta["strike"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("filled_at").reset_index(drop=True)
    return df


def pair_entries_exits(fills: pd.DataFrame) -> pd.DataFrame:
    """Pair entry fills with exit fills using proper FIFO with quantity tracking.

    Critical fix: when a batch sell (e.g., qty=8) closes multiple buys, this
    properly walks through the buy queue consuming each one's quantity until
    the sell is fully accounted for. The previous implementation only paired
    1 buy per sell, dropping the rest and severely under-counting P&L.

    For OPTIONS: pairs per OCC symbol independently (multi=N strategy).
    For EQUITY: assumes one-position-at-a-time stack.

    Each row represents one round-trip lot. A single batch close can produce
    multiple rows (one per buy lot it consumes).
    """
    if fills.empty:
        return pd.DataFrame()

    pairs = []

    # Split by symbol so each OCC contract has its own queue
    for sym, group in fills.groupby("symbol"):
        group = group.sort_values("filled_at").reset_index(drop=True)
        is_option = bool(group["is_option"].iloc[0]) if "is_option" in group.columns else False

        # Queue of (fill_dict, remaining_qty) for unmatched buys
        entries: list[tuple[dict, float]] = []

        for _, f in group.iterrows():
            f_dict = f.to_dict()
            f_qty = float(f["filled_qty"])

            if f["side"] == "buy":
                entries.append((f_dict, f_qty))
                continue

            if f["side"] != "sell":
                continue

            # Walk the entries queue, consuming buy quantities until sell is filled
            sell_remaining = f_qty
            while sell_remaining > 0 and entries:
                buy_fill, buy_remaining = entries[0]
                matched_qty = min(buy_remaining, sell_remaining)

                pnl_per_share = (f["filled_avg_price"] - buy_fill["filled_avg_price"])
                pnl_dollars = pnl_per_share * matched_qty * (100 if is_option else 1)
                pnl_bp = (pnl_per_share / buy_fill["filled_avg_price"]) * 1e4 if buy_fill["filled_avg_price"] else 0

                pairs.append({
                    "entry_at": buy_fill["filled_at"],
                    "exit_at": f["filled_at"],
                    "symbol": sym,
                    "side": "long",
                    "entry_px": buy_fill["filled_avg_price"],
                    "exit_px": f["filled_avg_price"],
                    "qty": matched_qty,
                    "pnl_dollars": pnl_dollars,
                    "pnl_bp": pnl_bp,
                    "is_option": is_option,
                    "underlying": buy_fill.get("underlying", sym),
                    "expiry": buy_fill.get("expiry"),
                    "opt_side": buy_fill.get("opt_side"),
                    "strike": buy_fill.get("strike"),
                })

                buy_remaining -= matched_qty
                sell_remaining -= matched_qty

                if buy_remaining <= 0:
                    entries.pop(0)
                else:
                    # Update the partial buy at front of queue
                    entries[0] = (buy_fill, buy_remaining)

    df = pd.DataFrame(pairs)
    if not df.empty:
        df = df.sort_values("entry_at").reset_index(drop=True)
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
