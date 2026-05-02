"""Live insights panels for the options dashboard.

5 real-time panels driven entirely by live Alpaca fills + bar data:
  1. Reason-code chart (today's exits by inferred reason)
  2. Today's tail-risk (worst trade, intraday DD, current losing streak)
  3. Live turnover gauge (trades/hr, conversion %)
  4. Rolling alpha (correlation of bot daily P&L vs SPY)
  5. Trail-stop heatmap (open positions vs trail-arm threshold)

All functions are pure — they take a trades DataFrame (from
data.pair_entries_exits) plus optional context and return either a small dict
of metrics or a plotly figure.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Bot config (mirror configs/v1.yaml strategy.options block)
TARGET_PCT = 1.00       # 100% premium gain target
STOP_PCT = 0.50         # 50% premium loss stop
TRAIL_THRESHOLD_PCT = 0.30   # arm trail at +30% gain
EOD_BUFFER_MIN = 5      # exits within 5 min of close are "eod_flat"


# =============================================================================
# Helpers
# =============================================================================

def _et_today() -> pd.Timestamp:
    """Today's calendar date in America/New_York."""
    return pd.Timestamp.now(tz="UTC").tz_convert("America/New_York").normalize()


def _et_market_close(date) -> pd.Timestamp:
    """4pm ET for the given date, returned in UTC."""
    if isinstance(date, pd.Timestamp):
        date = date.tz_convert("America/New_York")
    close = pd.Timestamp(date).tz_localize("America/New_York") if date.tz is None else date
    close = close.normalize() + pd.Timedelta(hours=16)
    return close.tz_convert("UTC")


def _filter_today_exits(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    df = trades.copy()
    df["exit_at"] = pd.to_datetime(df["exit_at"], utc=True)
    today_et = _et_today()
    today_utc_start = today_et.tz_convert("UTC")
    today_utc_end = today_utc_start + pd.Timedelta(days=1)
    return df[(df["exit_at"] >= today_utc_start) & (df["exit_at"] < today_utc_end)]


# =============================================================================
# Panel 1: reason-code chart
# =============================================================================

def infer_exit_reason(row, target_pct=TARGET_PCT, stop_pct=STOP_PCT,
                      trail_pct=TRAIL_THRESHOLD_PCT) -> str:
    """Derive an exit-reason label from one trade's price ratio + timing.

    Hierarchy of checks (first match wins):
      eod_flat → exit within 5 min of 4pm ET
      target   → premium gain ≥ target_pct (with 5% slippage tolerance)
      stop     → premium loss ≥ stop_pct (with 5% tolerance)
      trail    → exited in profit AND peak premium implied trail engaged
                 (we don't have the peak — proxy: profit > trail_threshold)
      other    → everything else
    """
    if not row.get("entry_px") or row["entry_px"] <= 0:
        return "other"
    pnl_pct = (row["exit_px"] - row["entry_px"]) / row["entry_px"]

    exit_at = pd.Timestamp(row["exit_at"], tz="UTC")
    et = exit_at.tz_convert("America/New_York")
    mins_to_close = (16 * 60) - (et.hour * 60 + et.minute)
    if 0 <= mins_to_close <= EOD_BUFFER_MIN:
        return "eod_flat"

    if pnl_pct >= target_pct - 0.05:
        return "target"
    if pnl_pct <= -(stop_pct - 0.05):
        return "stop"
    if pnl_pct >= trail_pct:
        return "trail"
    if pnl_pct > 0:
        return "trail"      # any profitable exit not at target → likely trail
    return "other"


def reason_code_chart(trades: pd.DataFrame) -> go.Figure:
    """Bar chart of today's exits grouped by inferred reason, with $ overlay."""
    today = _filter_today_exits(trades).copy()
    if today.empty:
        return _empty_fig("No exits today yet")

    today["reason"] = today.apply(infer_exit_reason, axis=1)
    grouped = today.groupby("reason").agg(
        n=("pnl_dollars", "size"),
        total=("pnl_dollars", "sum"),
        avg=("pnl_dollars", "mean"),
    ).reset_index()

    order = ["target", "trail", "eod_flat", "stop", "other"]
    grouped["sort"] = grouped["reason"].map({k: i for i, k in enumerate(order)}).fillna(99)
    grouped = grouped.sort_values("sort")

    colors = {"target": "#10b981", "trail": "#06b6d4", "eod_flat": "#a78bfa",
              "stop": "#ef4444", "other": "#6b7280"}
    bar_colors = [colors.get(r, "#6b7280") for r in grouped["reason"]]

    fig = go.Figure()
    fig.add_bar(
        x=grouped["reason"], y=grouped["n"],
        marker_color=bar_colors,
        text=[f"{int(n)} trades<br>${t:+,.0f}" for n, t in zip(grouped["n"], grouped["total"])],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>n=%{y}<br>%{text}<extra></extra>",
    )
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=11),
        yaxis=dict(title="trades", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


# =============================================================================
# Panel 2: today's tail-risk
# =============================================================================

def today_tail_risk(trades: pd.DataFrame) -> dict:
    """Today's worst trade, intraday DD, current losing streak."""
    today = _filter_today_exits(trades)
    if today.empty:
        return {"worst": 0.0, "intraday_dd": 0.0, "loss_streak": 0,
                "n_today": 0, "win_today": 0.0}

    today = today.sort_values("exit_at")
    pnl = today["pnl_dollars"].values
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    intraday_dd = float(dd.min()) if len(dd) else 0.0

    # Current losing streak: count from the END
    loss_streak = 0
    for v in reversed(pnl):
        if v < 0:
            loss_streak += 1
        else:
            break

    return {
        "worst": float(pnl.min()) if len(pnl) else 0.0,
        "intraday_dd": intraday_dd,
        "loss_streak": int(loss_streak),
        "n_today": int(len(today)),
        "win_today": float((pnl > 0).mean() * 100) if len(pnl) else 0.0,
    }


# =============================================================================
# Panel 3: live turnover gauge
# =============================================================================

def live_turnover(trades: pd.DataFrame, hours_lookback: int = 8) -> dict:
    """Trades per hour over the last N hours + intraday cumulative."""
    if trades.empty:
        return {"trades_per_hr": 0.0, "n_lookback": 0,
                "long_pct": 0.0, "short_pct": 0.0, "avg_hold_min": 0.0}

    df = trades.copy()
    df["exit_at"] = pd.to_datetime(df["exit_at"], utc=True)
    df["entry_at"] = pd.to_datetime(df["entry_at"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_lookback)
    sub = df[df["exit_at"] >= cutoff]
    n = len(sub)
    if n == 0:
        return {"trades_per_hr": 0.0, "n_lookback": 0,
                "long_pct": 0.0, "short_pct": 0.0, "avg_hold_min": 0.0}

    sub["hold_min"] = (sub["exit_at"] - sub["entry_at"]).dt.total_seconds() / 60
    long_n = int((sub.get("opt_side", "") == "call").sum()) if "opt_side" in sub.columns else int((sub.get("side", "") == "long").sum())
    short_n = n - long_n

    return {
        "trades_per_hr": n / hours_lookback,
        "n_lookback": n,
        "long_pct": long_n / n * 100,
        "short_pct": short_n / n * 100,
        "avg_hold_min": float(sub["hold_min"].mean()),
    }


# =============================================================================
# Panel 4: rolling alpha (P&L vs SPY correlation)
# =============================================================================

def rolling_alpha(trades: pd.DataFrame, spy_bars: pd.DataFrame,
                   window_days: int = 20) -> dict:
    """Daily P&L correlation with SPY daily returns over last N days."""
    if trades.empty or spy_bars.empty:
        return {"corr": 0.0, "n_days": 0, "alpha_dollars_per_day": 0.0}

    df = trades.copy()
    df["exit_at"] = pd.to_datetime(df["exit_at"], utc=True)
    df["et_date"] = df["exit_at"].dt.tz_convert("America/New_York").dt.date
    daily_pnl = df.groupby("et_date")["pnl_dollars"].sum()

    # SPY daily returns from minute bars
    bars = spy_bars.copy()
    if "close" not in bars.columns or bars.empty:
        return {"corr": 0.0, "n_days": 0, "alpha_dollars_per_day": 0.0}
    bars.index = pd.to_datetime(bars.index, utc=True)
    daily_spy = bars["close"].resample("1D").last().dropna()
    daily_spy_ret = daily_spy.pct_change().dropna()
    daily_spy_ret.index = daily_spy_ret.index.date

    # Align
    aligned = pd.concat([daily_pnl.rename("pnl"), daily_spy_ret.rename("spy")],
                        axis=1).dropna()
    aligned = aligned.tail(window_days)
    if len(aligned) < 3:
        return {"corr": 0.0, "n_days": len(aligned),
                "alpha_dollars_per_day": float(daily_pnl.mean()) if len(daily_pnl) else 0.0}

    corr = float(aligned["pnl"].corr(aligned["spy"]))
    # Quick alpha estimate: residual after fitting a linear regression
    x = aligned["spy"].values; y = aligned["pnl"].values
    slope, intercept = np.polyfit(x, y, 1) if len(x) >= 2 else (0, float(y.mean()))
    return {
        "corr": corr if not np.isnan(corr) else 0.0,
        "n_days": int(len(aligned)),
        "alpha_dollars_per_day": float(intercept),  # P&L when SPY return = 0
        "beta_dollars_per_pct": float(slope),       # $ per 1% SPY move
    }


# =============================================================================
# Panel 5: trail-stop heatmap
# =============================================================================

def trail_stop_states(positions: list, current_prices: dict | None = None,
                      trail_pct: float = TRAIL_THRESHOLD_PCT) -> dict:
    """For each open position, classify by % gain vs trail-arm threshold.

    Each position dict needs: symbol, qty, avg_entry_price, current_price.
    Returns counts + list of position-level details.
    """
    if not positions:
        return {"n_open": 0, "n_above_arm": 0, "n_in_profit": 0,
                "n_underwater": 0, "details": []}

    details = []
    n_above = 0; n_in_prof = 0; n_under = 0
    for p in positions:
        try:
            qty = float(getattr(p, "qty", 0) or 0)
            entry = float(getattr(p, "avg_entry_price", 0) or 0)
            curr = float(getattr(p, "current_price", 0) or 0)
            if not curr and current_prices:
                sym = getattr(p, "symbol", None)
                curr = float(current_prices.get(sym, 0) or 0)
        except Exception:
            continue
        if entry <= 0 or curr <= 0:
            continue
        pnl_pct = (curr - entry) / entry
        if pnl_pct >= trail_pct:
            n_above += 1
            state = "trail_armed"
        elif pnl_pct >= 0:
            n_in_prof += 1
            state = "in_profit"
        else:
            n_under += 1
            state = "underwater"
        details.append({
            "symbol": getattr(p, "symbol", "?"),
            "qty": qty,
            "entry": entry,
            "current": curr,
            "pnl_pct": pnl_pct,
            "state": state,
        })

    return {
        "n_open": len(details),
        "n_above_arm": n_above,
        "n_in_profit": n_in_prof,
        "n_underwater": n_under,
        "details": sorted(details, key=lambda d: -d["pnl_pct"]),
    }


def trail_stop_chart(states: dict, trail_pct: float = TRAIL_THRESHOLD_PCT) -> go.Figure:
    """Horizontal bar chart of open positions by current % gain."""
    details = states.get("details", [])
    if not details:
        return _empty_fig("No open positions")

    syms = [d["symbol"][:18] for d in details]
    pcts = [d["pnl_pct"] * 100 for d in details]
    colors = []
    for d in details:
        if d["state"] == "trail_armed":
            colors.append("#10b981")
        elif d["state"] == "in_profit":
            colors.append("#06b6d4")
        else:
            colors.append("#ef4444")

    fig = go.Figure()
    fig.add_bar(
        y=syms, x=pcts, orientation="h",
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in pcts], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    )
    # Vertical line at trail_pct threshold
    fig.add_vline(x=trail_pct * 100, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                  annotation_text=f"trail @ +{trail_pct*100:.0f}%", annotation_position="top")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)")

    fig.update_layout(
        height=max(180, 28 * len(syms) + 60),
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=11),
        xaxis=dict(title="% gain since entry", gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        showlegend=False,
    )
    return fig


# =============================================================================
# Empty-state placeholder
# =============================================================================

def _empty_fig(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(color="#94a3b8", size=13))
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                       yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    return fig
