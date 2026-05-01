"""Streamlit dashboard for the live paper-trading loop.

Tabbed layout:
  Live        — chart, open positions, recent trades, model state
  Performance — equity curve, daily P&L, hourly heatmap, stats panel
  Risk        — exposure, leverage, BP, capital deployment
  Health      — system + API + webhook diagnostics

Run:
    .venv/bin/streamlit run ui/dashboard.py
"""
from __future__ import annotations

import os
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import data as d
from ui import charts as c
from ui import state


st.set_page_config(
    page_title="ML Options Trader",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": None, "Get help": None, "Report a bug": None},
)

# Window options
WINDOW_OPTIONS = {
    "1m":  ([4, 8, 24, 48], 24),
    "5m":  ([8, 24, 48, 168], 48),
    "15m": ([24, 72, 168, 336], 168),
    "1h":  ([72, 168, 336, 720], 336),
    "1d":  ([720, 2160, 8760, 17520], 2160),
}
WINDOW_LABELS = {4: "4h", 8: "8h", 24: "1d", 48: "2d", 72: "3d", 168: "1w",
                 336: "2w", 720: "30d", 2160: "90d", 8760: "1y", 17520: "2y"}


# ============================================================================
# Typography + layout
# ============================================================================
st.markdown(
    """
    <style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1700px;}

    /* Tighten vertical density */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {gap: 0.4rem;}

    /* Tab bar — make it look professional */
    div[data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 1px solid #1a1f2c;
        margin-bottom: 1rem;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #6b7280 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        padding: 0.6rem 1.2rem !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #d1d4dc !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f97316 !important;
        border-bottom-color: #f97316 !important;
    }

    /* Pill-style timeframe radio */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        flex-direction: row;
        gap: 4px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: #0f1117;
        border: 1px solid #1a1f2c;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #9ca3af;
        cursor: pointer;
        margin: 0 !important;
        min-width: 36px;
        text-align: center;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: #3b82f6;
        color: #d1d4dc;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: #f97316;
        border-color: #f97316;
        color: white;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    div[data-testid="stRadio"] > label {
        font-size: 0.62rem !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* Font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.7rem;
        color: #6b7280;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    /* Code-style numbers */
    .num {font-family: "SF Mono", "Menlo", monospace; font-variant-numeric: tabular-nums;}

    h4 {color:#e5e7eb; font-size:0.8rem; font-weight:600; text-transform:uppercase;
        letter-spacing:0.10em; margin-top:1.2rem; margin-bottom:0.5rem;}

    /* Background — true black for OLED look */
    .stApp {background: #08090d !important;}
    section[data-testid="stSidebar"] {background: #0a0d14 !important;}
    [data-testid="stDataFrame"] {background: #0f1117 !important;}

    hr {border-color: #1a1f2c; margin: 1.4rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Sidebar — controls
# ============================================================================
with st.sidebar:
    st.markdown("#### Controls")
    halted = state.halted()
    if halted:
        st.markdown(
            '<div style="background:#2a1a1a;border-left:3px solid #ef5350;'
            'padding:8px 12px;border-radius:3px;margin-bottom:8px;">'
            '<div style="font-size:0.7rem;color:#ef5350;letter-spacing:0.1em;">HALTED</div>'
            '<div style="font-size:0.8rem;color:#9ca3af;">No new entries</div>'
            "</div>", unsafe_allow_html=True,
        )
        if st.button("Resume trading"):
            state.set_halt(False)
            st.rerun()
    else:
        st.markdown(
            '<div style="background:#0f2118;border-left:3px solid #26a69a;'
            'padding:8px 12px;border-radius:3px;margin-bottom:8px;">'
            '<div style="font-size:0.7rem;color:#26a69a;letter-spacing:0.1em;">TRADING</div>'
            '<div style="font-size:0.8rem;color:#9ca3af;">New entries allowed</div>'
            "</div>", unsafe_allow_html=True,
        )
        if st.button("Halt new entries", type="primary"):
            state.set_halt(True)
            st.rerun()

    if st.button("Flatten position now"):
        msg = d.close_position_now("SPY")
        st.toast(msg)
    auto = st.toggle("Auto-refresh (30s)", value=False)
    if auto:
        st_autorefresh(interval=30_000, key="auto_refresh")

    st.markdown("---")
    st.caption("Phase 6 winner deployed")
    st.caption("Validated: 19/19 months Pareto-dominant")


# ============================================================================
# Pull live state once for the whole page
# ============================================================================
acct = d.account_info()
hb = state.read_heartbeat()
hb_age = state.heartbeat_age_seconds()
art = d.artifact()
pos = d.current_position("SPY")
all_pos = d.all_positions()

# Read strategy config
try:
    from utils.config import load as _load_cfg
    _cfg = _load_cfg("v1")
    _mode = _cfg.get("strategy", {}).get("mode", "stocks")
    _opt_cfg = _cfg.get("strategy", {}).get("options", {})
    _exp = _opt_cfg.get("expiration", "—")
    _conv = float(_opt_cfg.get("conviction_min", 0.55))
    _max_conc = int(_opt_cfg.get("max_concurrent_positions", 1))
    _moneyness = _opt_cfg.get("moneyness", "atm").upper()
    _itm_offset = float(_opt_cfg.get("itm_offset_pct", 0))
    _target_pct = float(_opt_cfg.get("target_pct", 0))
    _stop_pct = float(_opt_cfg.get("stop_pct", 0))
    _risk_pct = float(_opt_cfg.get("risk_pct_per_trade", 0))
except Exception:
    _mode = "stocks"; _exp="—"; _conv=0; _max_conc=0; _moneyness="—"
    _itm_offset = _target_pct = _stop_pct = _risk_pct = 0

# Today's fills + trades
today_fills = d.recent_fills(lookback_hours=24, symbol="SPY")
today_trades = d.pair_entries_exits(today_fills) if not today_fills.empty else pd.DataFrame()
day_pnl = float(today_trades["pnl_dollars"].sum()) if not today_trades.empty else 0.0
eq = float(acct.get("equity", 0))
last_price = (hb or {}).get("last_price") or 0
day_pnl_pct = (day_pnl / eq * 100) if eq else 0
ot = (hb or {}).get("open_trade") or {}

# Session state
import pytz
now_et_dt = datetime.now(pytz.timezone("America/New_York"))
mins_into = now_et_dt.hour * 60 + now_et_dt.minute - (9 * 60 + 30)
is_weekday = now_et_dt.weekday() < 5
if not is_weekday:
    session_state, session_color = "WEEKEND", "#6b7280"
elif mins_into < 0:
    mins_to_open = -mins_into
    session_state = f"PRE-MKT  T-{mins_to_open//60:02d}:{mins_to_open%60:02d}"
    session_color = "#fbbf24"
elif mins_into < 390:
    mins_left = 390 - mins_into
    session_state = f"OPEN  {mins_left//60:02d}:{mins_left%60:02d} left"
    session_color = "#26a69a"
else:
    session_state, session_color = "CLOSED", "#6b7280"

# Heartbeat color
if hb_age is None:
    hb_text, hb_color = "—", "#6b7280"
elif hb_age < 120:
    hb_text, hb_color = f"{int(hb_age)}s", "#26a69a"
elif hb_age < 600:
    hb_text, hb_color = f"{int(hb_age)}s", "#fbbf24"
else:
    hb_text, hb_color = f"{int(hb_age/60)}m", "#ef5350"

# Status
if halted:
    status_text, status_color = "HALTED", "#ef5350"
elif hb and hb.get("killed_for_day"):
    status_text, status_color = "DD KILL", "#fbbf24"
elif hb and hb.get("in_window"):
    status_text, status_color = "LIVE", "#26a69a"
else:
    status_text, status_color = "IDLE", "#6b7280"

# Count option positions
def _is_occ(sym): return isinstance(sym, str) and len(sym) >= 15
option_positions = [p for p in all_pos if _is_occ(getattr(p, "symbol", None))]
n_open = len(option_positions)


# ============================================================================
# TOP STRIP — 4 big KPI tiles + strategy badge
# ============================================================================
session_clock = now_et_dt.strftime("%H:%M:%S ET")

# Color logic
pnl_color = "#26a69a" if day_pnl >= 0 else "#ef5350"
exposure_dollars = sum(float(p.cost_basis) for p in option_positions) if option_positions else 0.0
unrealized_dollars = sum(float(p.unrealized_pl) for p in option_positions) if option_positions else 0.0
unreal_color = "#26a69a" if unrealized_dollars >= 0 else "#ef5350"
exposure_pct = (exposure_dollars / eq * 100) if eq else 0

# Strategy badge
_strat_pill = (
    f'<span style="background:#f97316;color:white;font-weight:700;font-size:0.65rem;'
    f'letter-spacing:0.10em;padding:0.18rem 0.55rem;border-radius:3px;">📈 OPTIONS</span>'
    if _mode == "options" else
    f'<span style="background:#3b82f6;color:white;font-weight:700;font-size:0.65rem;'
    f'letter-spacing:0.10em;padding:0.18rem 0.55rem;border-radius:3px;">📊 STOCKS</span>'
)
strat_subtitle = (
    f"{_exp.replace('_', ' ')} · {_moneyness} {_itm_offset*100:.1f}% · "
    f"target +{_target_pct*100:.0f}% · stop −{_stop_pct*100:.0f}% · "
    f"risk {_risk_pct*100:.1f}% · max {_max_conc} concurrent"
) if _mode == "options" else "stocks-mode (legacy)"

st.markdown(
    f'<div style="display:flex;align-items:center;justify-content:space-between;'
    f'background:#1a0f02;border:1px solid #f97316;border-radius:6px;'
    f'padding:0.5rem 1rem;margin-bottom:0.6rem;">'
    f'  <div style="display:flex;align-items:center;gap:0.7rem;">'
    f'    {_strat_pill}'
    f'    <span style="color:#e5e7eb;font-weight:500;font-size:0.85rem;font-family:\"SF Mono\",monospace;">{strat_subtitle}</span>'
    f'  </div>'
    f'  <div style="color:#9ca3af;font-size:0.72rem;font-family:\"SF Mono\",monospace;">'
    f'    SPY · v3_mtf · paper · {session_clock}'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# 4 KPI tiles row
_MONO = '"SF Mono","Menlo",monospace'

def _tile(label: str, value: str, sub: str = "", value_color: str = "#e5e7eb",
          sub_color: str = "#9ca3af", flex: int = 1) -> str:
    sub_html = ""
    if sub:
        sub_html = (
            f'<div style="font-size:0.72rem;color:{sub_color};margin-top:0.2rem;'
            f'font-family:{_MONO};">{sub}</div>'
        )
    return (
        f'<div style="display:flex;flex-direction:column;justify-content:center;'
        f'padding:0.7rem 1rem;background:#0a0d14;border:1px solid #1a1f2c;'
        f'border-radius:6px;flex:{flex};min-width:0;">'
        f'  <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.10em;'
        f'color:#6b7280;font-weight:600;margin-bottom:0.25rem;">{label}</div>'
        f'  <div style="font-size:1.3rem;color:{value_color};font-weight:600;'
        f'font-family:{_MONO};font-variant-numeric:tabular-nums;line-height:1.1;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{value}</div>'
        f'  {sub_html}'
        f'</div>'
    )

_bp_val = float(acct.get("buying_power", 0))
_eq_tile = _tile("Equity", f"${eq:,.0f}", f"buying power ${_bp_val:,.0f}")
_pnl_sub = f"{day_pnl_pct:+.2f}% · {len(today_trades)} trades"
_pnl_tile = _tile("Today P&L", f"${day_pnl:+,.0f}", _pnl_sub, value_color=pnl_color)
_pos_sub = f"deployed ${exposure_dollars:,.0f} ({exposure_pct:.0f}% of equity)" if n_open else "no positions"
_pos_tile = _tile("Open Positions", f"{n_open} / {_max_conc}", _pos_sub)
_unreal_pct = (unrealized_dollars / exposure_dollars * 100) if exposure_dollars else 0
_unreal_tile = _tile("Unrealized", f"${unrealized_dollars:+,.0f}",
                     f"{_unreal_pct:+.1f}% on cost", value_color=unreal_color)
_status_tile = _tile("Status", status_text, f"heartbeat {hb_text}",
                     value_color=status_color, sub_color=hb_color)

st.markdown(
    '<div style="display:flex;gap:0.5rem;margin-bottom:0.6rem;">'
    + _eq_tile + _pnl_tile + _pos_tile + _unreal_tile + _status_tile +
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================================
# TABS — Live / Performance / Risk / Health
# ============================================================================
tab_live, tab_perf, tab_risk, tab_health = st.tabs([
    "Live", "Performance", "Risk", "Health"
])


# ---------------- LIVE TAB ----------------
with tab_live:
    # Default timeframe / window
    symbol = "SPY"
    timeframe = st.session_state.get("timeframe", "1m")
    _opts, _default = WINDOW_OPTIONS[timeframe]
    hours = st.session_state.get("hours", _default)

    # Chart controls
    ctrl_l, ctrl_r = st.columns([2, 1])
    with ctrl_l:
        timeframe = st.radio(
            "Timeframe", options=["1m", "5m", "15m", "1h", "1d"],
            horizontal=True,
            index=["1m", "5m", "15m", "1h", "1d"].index(timeframe),
            key="timeframe", label_visibility="visible",
        )
    with ctrl_r:
        _opts, _default = WINDOW_OPTIONS[timeframe]
        if hours not in _opts:
            hours = _default
        hours = st.select_slider(
            "Window", options=_opts, value=hours,
            format_func=lambda h: WINDOW_LABELS.get(h, f"{h}h"), key="hours",
        )

    # Chart
    bars = d.fetch_bars(symbol, hours=hours, timeframe=timeframe)
    preds = d.replay_predictions(hours=hours) if timeframe == "1m" else pd.DataFrame()
    fills = d.recent_fills(lookback_hours=max(hours, 48), symbol=symbol)
    all_trades_paired = d.pair_entries_exits(fills) if not fills.empty else pd.DataFrame()

    # Split into equity vs option round-trips
    if not all_trades_paired.empty and "is_option" in all_trades_paired.columns:
        equity_trades = all_trades_paired[~all_trades_paired["is_option"]].reset_index(drop=True)
        option_trades = all_trades_paired[all_trades_paired["is_option"]].reset_index(drop=True)
    else:
        equity_trades = all_trades_paired
        option_trades = pd.DataFrame()
    trades = equity_trades  # backwards-compat name for chart

    event_marks = pd.DataFrame()
    if not bars.empty:
        from utils.calendar import is_fomc_day
        f = is_fomc_day(bars.index)
        fomc_bars = bars.index[f.values]
        if len(fomc_bars) > 0:
            fomc_days = pd.Series(fomc_bars.tz_convert("America/New_York").date).drop_duplicates()
            event_marks = pd.DataFrame({
                "ts": [bars.index[bars.index.tz_convert("America/New_York").date == d_][0]
                       for d_ in fomc_days],
                "label": ["FOMC"] * len(fomc_days),
            })

    # Trade selector — pick one option trade to highlight on chart
    selected_trade_dict = None
    if not option_trades.empty:
        # Build human-readable labels for the selector
        opt_labels = ["— all trades, no highlight —"]
        for i, row in option_trades.iterrows():
            ts_et = pd.to_datetime(row["entry_at"]).tz_convert("America/New_York")
            opt_labels.append(
                f"#{i+1}  {ts_et.strftime('%m/%d %H:%M')}  "
                f"{(row['opt_side'] or '?').upper()} ${row.get('strike', 0):.0f}  "
                f"premium ${row['entry_px']:.2f}  "
                f"P&L ${row['pnl_dollars']:+.0f}"
            )
        selected_label = st.selectbox(
            "Highlight a trade (TP/SL projected on chart)",
            options=opt_labels, index=0, key="selected_trade_label",
        )
        if selected_label != opt_labels[0]:
            sel_idx = opt_labels.index(selected_label) - 1  # account for "all" entry
            selected_trade_dict = option_trades.iloc[sel_idx].to_dict()

    thr = (float(art.thresholds["up"]), float(art.thresholds["down"]))
    if bars.empty:
        st.info(f"No `{timeframe}` bars in last {hours}h — likely outside RTH. Try a longer window.")
    else:
        fig = c.candlestick_with_trades(
            bars, trades, preds,
            thresholds=thr,
            event_marks=event_marks,
            title="",
            timeframe=timeframe,
            open_trade=ot or None,
            option_trades=option_trades if not option_trades.empty else None,
            selected_option_trade=selected_trade_dict,
            target_pct=_target_pct,
            stop_pct=_stop_pct,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False}, key="candlestick")
    st.caption(
        "Drag to zoom · y-axis for price · rangeslider for time · click legend to hide series · "
        "Pick a trade above to overlay its specific TP/SL on the chart"
    )

    # Open Options Positions panel
    if option_positions:
        st.markdown("#### Open Option Positions")
        pos_rows = []
        for p in sorted(option_positions, key=lambda x: float(x.unrealized_pl), reverse=True):
            sym = p.symbol
            # Parse OCC: SPY260511C00712000 → SPY 260511 C 712.000
            try:
                root = sym[:6].strip() or "SPY"
                date_part = sym[-15:-9]
                cp = sym[-9]
                strike_int = int(sym[-8:])
                strike = strike_int / 1000.0
                exp_pretty = f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:6]}"
                opt_side = "CALL" if cp == "C" else "PUT"
            except Exception:
                root, exp_pretty, opt_side, strike = sym, "—", "?", 0
            pos_rows.append({
                "Symbol": sym,
                "Side": opt_side,
                "Strike": strike,
                "Expires": exp_pretty,
                "Qty": int(float(p.qty)),
                "Cost": float(p.cost_basis),
                "Mkt Value": float(p.market_value),
                "P&L $": float(p.unrealized_pl),
                "P&L %": float(p.unrealized_plpc) * 100,
            })
        pos_df = pd.DataFrame(pos_rows)
        st.dataframe(
            pos_df.style.format({
                "Strike": "${:,.2f}", "Cost": "${:,.0f}", "Mkt Value": "${:,.0f}",
                "P&L $": "${:+,.0f}", "P&L %": "{:+.1f}%",
            }).background_gradient(subset=["P&L $"], cmap="RdYlGn", vmin=-1000, vmax=1000),
            hide_index=True, use_container_width=True,
            height=min(400, 80 + len(pos_rows) * 32),
        )

    # Recent trades + SHAP
    st.markdown("---")
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown("#### Trades — last 48h")
        # Combined view: option trades take priority since that's what the bot does
        display_trades = option_trades if not option_trades.empty else equity_trades
        if display_trades.empty:
            st.markdown(
                '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
                "No round-trips yet. Markers populate once the loop trades."
                "</div>", unsafe_allow_html=True,
            )
        else:
            show = display_trades.copy()
            show["Entry"] = show["entry_at"].dt.tz_convert("America/New_York").dt.strftime("%m/%d %H:%M")
            show["Exit"] = show["exit_at"].dt.tz_convert("America/New_York").dt.strftime("%m/%d %H:%M")
            if "is_option" in show.columns and show["is_option"].any():
                # Options table layout
                show["Type"] = show["opt_side"].str.upper().fillna("EQUITY")
                show["Strike"] = show.get("strike", pd.Series([None]*len(show)))
                show["Entry $"] = show["entry_px"]
                show["Exit $"] = show["exit_px"]
                show = show[["Entry", "Exit", "Type", "Strike", "Entry $", "Exit $", "qty", "pnl_dollars"]]
                show.columns = ["Entry", "Exit", "Type", "Strike", "Entry $", "Exit $", "Qty", "P&L $"]
                fmt = {
                    "Strike": "${:,.0f}", "Entry $": "${:.2f}", "Exit $": "${:.2f}",
                    "Qty": "{:.0f}", "P&L $": "${:+,.0f}",
                }
            else:
                show = show[["Entry", "Exit", "side", "entry_px", "exit_px", "qty",
                             "pnl_dollars", "pnl_bp"]]
                show.columns = ["Entry", "Exit", "Side", "Entry $", "Exit $", "Qty", "P&L $", "P&L bp"]
                fmt = {
                    "Entry $": "${:,.2f}", "Exit $": "${:,.2f}",
                    "Qty": "{:.0f}", "P&L $": "${:+,.2f}", "P&L bp": "{:+.1f}",
                }
            st.dataframe(
                show.iloc[::-1].style.format(fmt),
                hide_index=True, use_container_width=True, height=320,
            )
            total_pnl = display_trades["pnl_dollars"].sum()
            wins = (display_trades["pnl_dollars"] > 0).sum()
            hit = wins / len(display_trades) if len(display_trades) else 0
            st.markdown(
                f'<div style="color:#9ca3af;font-size:0.85rem;margin-top:0.4rem;">'
                f'<span style="color:#e5e7eb;font-weight:600;">{len(display_trades)}</span> trades · '
                f'Hit rate <span style="color:#e5e7eb;font-weight:600;">{hit*100:.1f}%</span> · '
                f'Cumulative <span style="color:{"#26a69a" if total_pnl>=0 else "#ef5350"};'
                f'font-weight:600;">${total_pnl:+.2f}</span>'
                "</div>", unsafe_allow_html=True,
            )
    with right:
        st.markdown("#### Feature attribution — latest prediction")
        shap_df = d.shap_explain_latest()
        if shap_df is None:
            st.markdown(
                '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
                "Waiting for fresh data to compute attribution."
                "</div>", unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(c.shap_bar(shap_df), use_container_width=True,
                            config={"displayModeBar": False}, key="shap_bar")


# ---------------- PERFORMANCE TAB ----------------
with tab_perf:
    all_fills = d.recent_fills(lookback_hours=24*30, symbol="SPY")
    all_trades = d.pair_entries_exits(all_fills) if not all_fills.empty else pd.DataFrame()
    stats = d.perf_stats(all_trades) if not all_trades.empty else {"n_trades": 0}

    def _stat(label, value, color="#e5e7eb"):
        return (
            f'<div style="display:flex;flex-direction:column;flex:1;padding:0 0.7rem;'
            f'border-left:1px solid #1a1f2c;">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.10em;'
            f'color:#6b7280;font-weight:500;">{label}</div>'
            f'<div style="font-size:1.05rem;color:{color};font-weight:600;'
            f'font-variant-numeric:tabular-nums;line-height:1.4;">{value}</div>'
            f'</div>'
        )

    st.markdown("#### Performance — Last 30 days")
    if stats.get("n_trades", 0) == 0:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
            "No closed trades yet. Stats populate after the loop completes round-trips."
            "</div>", unsafe_allow_html=True,
        )
    else:
        sharpe_color = "#26a69a" if stats["daily_sharpe"] >= 1 else "#fbbf24" if stats["daily_sharpe"] >= 0 else "#ef5350"
        pnl_color_30 = "#26a69a" if stats["total_pnl"] >= 0 else "#ef5350"
        st.markdown(
            '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
            'border-radius:4px;padding:0.75rem 0.4rem;">' +
            _stat("TOTAL P&L", f"${stats['total_pnl']:+,.2f}", pnl_color_30) +
            _stat("HIT RATE", f"{stats['hit_rate']*100:.1f}%") +
            _stat("SHARPE (D)", f"{stats['daily_sharpe']:.2f}", sharpe_color) +
            _stat("SHARPE (A)", f"{stats['sharpe_annualized']:.2f}", sharpe_color) +
            _stat("PROF FACTOR", f"{stats['profit_factor']:.2f}") +
            _stat("TRADES", f"{stats['n_trades']}") +
            '</div>', unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
            'border-radius:4px;padding:0.75rem 0.4rem;margin-top:0.4rem;">' +
            _stat("AVG TRADE", f"${stats['avg_trade']:+,.2f}") +
            _stat("BEST", f"${stats['best_trade']:+,.2f}", "#26a69a") +
            _stat("WORST", f"${stats['worst_trade']:+,.2f}", "#ef5350") +
            _stat("WIN/LOSS", f"{stats['win_loss_ratio']:.2f}") +
            _stat("MAX DD", f"${stats['max_dd']:+,.2f}", "#ef5350") +
            _stat("AVG/DAY", f"${stats['avg_per_day']:+,.2f}") +
            '</div>', unsafe_allow_html=True,
        )

    st.markdown("#### Live equity curve")
    st.plotly_chart(c.equity_curve(all_trades if not all_trades.empty else trades),
                    use_container_width=True,
                    config={"displayModeBar": False}, key="equity")

    st.markdown("---")
    pnl_left, pnl_right = st.columns([1.4, 1.0], gap="large")
    with pnl_left:
        st.markdown("#### Daily P&L — last 30 days")
        if all_trades.empty:
            st.markdown('<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">No daily P&L yet.</div>',
                        unsafe_allow_html=True)
        else:
            st.plotly_chart(c.daily_pnl_bars(all_trades, days=30),
                            use_container_width=True, config={"displayModeBar": False},
                            key="daily_pnl")
    with pnl_right:
        st.markdown("#### P&L by hour of day")
        if all_trades.empty:
            st.markdown('<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">Awaiting closed trades.</div>',
                        unsafe_allow_html=True)
        else:
            st.plotly_chart(c.hourly_pnl_heatmap(all_trades),
                            use_container_width=True, config={"displayModeBar": False},
                            key="hourly_heat")


# ---------------- RISK TAB ----------------
with tab_risk:
    eq_now = float(acct.get("equity", 0))
    bp_now = float(acct.get("buying_power", 0))
    notional = exposure_dollars
    leverage = notional / eq_now if eq_now else 0
    leverage_color = "#26a69a" if leverage < 0.5 else "#fbbf24" if leverage < 0.8 else "#ef5350"
    bp_color = "#26a69a" if (bp_now - notional) / max(bp_now, 1) > 0.4 else "#fbbf24" if (bp_now - notional) / max(bp_now, 1) > 0.2 else "#ef5350"

    st.markdown("#### Capital Exposure")

    def _stat2(label, value, color="#e5e7eb"):
        return (
            f'<div style="display:flex;flex-direction:column;flex:1;padding:0 0.7rem;'
            f'border-left:1px solid #1a1f2c;">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.10em;'
            f'color:#6b7280;font-weight:500;">{label}</div>'
            f'<div style="font-size:1.05rem;color:{color};font-weight:600;'
            f'font-variant-numeric:tabular-nums;line-height:1.4;">{value}</div>'
            f'</div>'
        )

    st.markdown(
        '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
        'border-radius:4px;padding:0.75rem 0.4rem;">' +
        _stat2("EQUITY", f"${eq_now:,.0f}") +
        _stat2("DEPLOYED", f"${notional:,.0f}") +
        _stat2("DEPLOYED %", f"{notional/eq_now*100 if eq_now else 0:.1f}%", leverage_color) +
        _stat2("BP AVAIL", f"${bp_now - notional:,.0f}", bp_color) +
        _stat2("MAX CONCURRENT", f"{n_open} / {_max_conc}") +
        '</div>', unsafe_allow_html=True,
    )

    # Per-position risk if options
    if option_positions:
        st.markdown("#### Per-position risk")
        # Compute -50% stop loss for each position
        risk_rows = []
        total_max_loss = 0.0
        for p in option_positions:
            cost = float(p.cost_basis)
            stop_loss_dollars = cost * _stop_pct  # -50% premium = -$X
            target_gain_dollars = cost * _target_pct  # +30% premium = +$Y
            total_max_loss += stop_loss_dollars
            risk_rows.append({
                "Symbol": p.symbol,
                "Cost": cost,
                "If hits stop (−50%)": -stop_loss_dollars,
                "If hits target (+30%)": target_gain_dollars,
                "Current P&L": float(p.unrealized_pl),
            })
        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(
            risk_df.style.format({
                "Cost": "${:,.0f}", "If hits stop (−50%)": "${:+,.0f}",
                "If hits target (+30%)": "${:+,.0f}", "Current P&L": "${:+,.0f}",
            }),
            hide_index=True, use_container_width=True, height=min(350, 80 + len(risk_rows)*32),
        )
        st.markdown(
            f'<div style="color:#9ca3af;font-size:0.85rem;margin-top:0.5rem;">'
            f'Worst-case if ALL stops hit: '
            f'<span style="color:#ef5350;font-weight:600;">−${total_max_loss:,.0f}</span> '
            f'({total_max_loss/eq_now*100 if eq_now else 0:.1f}% of equity)'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Account status")
    st.markdown(
        '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
        'border-radius:4px;padding:0.75rem 0.4rem;">' +
        _stat2("PAPER ACCOUNT", acct.get("account_number", "—") if "account_number" in acct else "active") +
        _stat2("OPTIONS LEVEL", str(acct.get("options_trading_level", "—"))) +
        _stat2("DAYTRADE COUNT", str(acct.get("daytrade_count", 0))) +
        _stat2("DAILY DD KILL", "armed" if not (hb or {}).get("killed_for_day") else "TRIPPED",
               "#26a69a" if not (hb or {}).get("killed_for_day") else "#ef5350") +
        '</div>', unsafe_allow_html=True,
    )


# ---------------- HEALTH TAB ----------------
with tab_health:
    st.markdown("#### System Health Diagnostics")
    st.caption("Live checks of every external integration. Click 'Run all checks' to refresh.")

    if "health_results" not in st.session_state:
        st.session_state.health_results = None

    cols = st.columns([1, 4])
    with cols[0]:
        if st.button("▶ Run all checks", type="primary"):
            results = {}
            # 1) Alpaca
            t0 = time.time()
            try:
                tc = d.trading_client()
                a = tc.get_account()
                results["alpaca"] = {
                    "ok": True,
                    "ms": int((time.time() - t0) * 1000),
                    "detail": f"account {a.account_number} · {a.status} · ${float(a.equity):,.0f} equity · "
                              f"options L{a.options_trading_level if hasattr(a,'options_trading_level') else '?'}",
                }
            except Exception as e:
                results["alpaca"] = {"ok": False, "ms": int((time.time()-t0)*1000), "detail": str(e)[:200]}

            # 2) Discord webhook
            t0 = time.time()
            try:
                wh = os.environ.get("DISCORD_WEBHOOK_URL")
                if not wh:
                    results["discord"] = {"ok": False, "ms": 0, "detail": "DISCORD_WEBHOOK_URL not set"}
                else:
                    r = requests.post(wh, json={"content": "🔵 dashboard health check ping (ignore)"}, timeout=10)
                    results["discord"] = {
                        "ok": r.status_code in (200, 204),
                        "ms": int((time.time()-t0)*1000),
                        "detail": f"status {r.status_code}",
                    }
            except Exception as e:
                results["discord"] = {"ok": False, "ms": int((time.time()-t0)*1000), "detail": str(e)[:200]}

            # 3) FRED
            t0 = time.time()
            try:
                fk = os.environ.get("FRED_API_KEY")
                if not fk:
                    results["fred"] = {"ok": False, "ms": 0, "detail": "FRED_API_KEY not set"}
                else:
                    r = requests.get(
                        "https://api.stlouisfed.org/fred/series/observations",
                        params={"series_id": "T10Y2Y", "api_key": fk, "file_type": "json",
                                "observation_start": "2026-04-25", "observation_end": "2026-04-30"},
                        timeout=10,
                    )
                    obs = r.json().get("observations", [])
                    results["fred"] = {
                        "ok": r.status_code == 200 and len(obs) > 0,
                        "ms": int((time.time()-t0)*1000),
                        "detail": f"status {r.status_code} · {len(obs)} observations · latest T10Y2Y={obs[-1].get('value', 'n/a') if obs else 'n/a'}",
                    }
            except Exception as e:
                results["fred"] = {"ok": False, "ms": int((time.time()-t0)*1000), "detail": str(e)[:200]}

            # 4) Yfinance (VIX)
            t0 = time.time()
            try:
                from data_pull import yf_daily
                df = yf_daily.pull("^VIX",
                                    str((pd.Timestamp.now() - pd.Timedelta(days=10)).date()),
                                    str(pd.Timestamp.now().date()),
                                    use_cache=False)
                results["yfinance"] = {
                    "ok": not df.empty,
                    "ms": int((time.time()-t0)*1000),
                    "detail": f"{len(df)} bars · latest close ${float(df['close'].iloc[-1]):.2f}" if not df.empty else "empty",
                }
            except Exception as e:
                results["yfinance"] = {"ok": False, "ms": int((time.time()-t0)*1000), "detail": str(e)[:200]}

            # 5) Heartbeat
            results["heartbeat"] = {
                "ok": hb_age is not None and hb_age < 600,
                "ms": int(hb_age) if hb_age else 9999,
                "detail": (f"last beat {int(hb_age)}s ago · session {(hb or {}).get('current_day', '—')}"
                           if hb_age is not None else "no heartbeat file found"),
            }

            # 6) Model artifact
            try:
                a = d.artifact()
                results["model"] = {
                    "ok": True,
                    "ms": 0,
                    "detail": f"{len(a.feature_cols)} features · thr_up={float(a.thresholds['up']):.2f} thr_dn={float(a.thresholds['down']):.2f}",
                }
            except Exception as e:
                results["model"] = {"ok": False, "ms": 0, "detail": str(e)[:200]}

            # 7) Config
            try:
                _cf = _load_cfg("v1")
                _opts = _cf.get("strategy", {}).get("options", {})
                results["config"] = {
                    "ok": True, "ms": 0,
                    "detail": (f"mode={_cf['strategy']['mode']} · expiration={_opts.get('expiration')} · "
                               f"itm_offset={_opts.get('itm_offset_pct')} · target={_opts.get('target_pct')}"),
                }
            except Exception as e:
                results["config"] = {"ok": False, "ms": 0, "detail": str(e)[:200]}

            # 8) Cache directory
            try:
                from data_pull import cache as _ch
                cache_root = _ch.cache_dir()
                files = list(cache_root.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files) / 1024 / 1024
                results["cache"] = {
                    "ok": len(files) > 0,
                    "ms": 0,
                    "detail": f"{len(files)} parquet files · {total_size:.0f} MB total · path={cache_root}",
                }
            except Exception as e:
                results["cache"] = {"ok": False, "ms": 0, "detail": str(e)[:200]}

            st.session_state.health_results = results

    with cols[1]:
        if st.session_state.health_results:
            st.caption(f"Last run: {datetime.now().strftime('%H:%M:%S')}")

    # Render results
    if st.session_state.health_results:
        results = st.session_state.health_results
        check_labels = {
            "alpaca":     ("Alpaca trading API",  "Account, positions, options orders"),
            "discord":    ("Discord webhook",      "Trade alerts (test ping sent)"),
            "fred":       ("FRED API",              "Macro features (T10Y2Y, BAMLH, DFII10)"),
            "yfinance":   ("Yahoo Finance",         "VIX + ES futures daily data"),
            "heartbeat":  ("Loop heartbeat",        "Live trader process pulse"),
            "model":      ("Model artifact",        "Trained v3_mtf LightGBM"),
            "config":     ("Strategy config",       "configs/v1.yaml"),
            "cache":      ("Bar cache",             "Local parquet files"),
        }
        for key, (label, sub) in check_labels.items():
            r = results.get(key, {"ok": False, "ms": 0, "detail": "not checked"})
            ok = r["ok"]
            color = "#26a69a" if ok else "#ef5350"
            icon = "✓" if ok else "✗"
            ms_str = f"{r['ms']}ms" if r["ms"] else "—"
            st.markdown(
                f'<div style="display:flex;align-items:center;background:#0a0d14;border:1px solid #1a1f2c;'
                f'border-left:3px solid {color};border-radius:4px;padding:0.7rem 1rem;margin-bottom:0.4rem;">'
                f'  <div style="color:{color};font-size:1.3rem;font-weight:700;width:28px;">{icon}</div>'
                f'  <div style="flex:1;">'
                f'    <div style="color:#e5e7eb;font-weight:600;font-size:0.9rem;">{label}'
                f'      <span style="color:#6b7280;font-weight:400;font-size:0.75rem;margin-left:0.5rem;">· {sub}</span>'
                f'    </div>'
                f'    <div style="color:#9ca3af;font-size:0.78rem;font-family:\"SF Mono\",monospace;margin-top:0.1rem;">{r["detail"]}</div>'
                f'  </div>'
                f'  <div style="color:#6b7280;font-size:0.75rem;font-family:\"SF Mono\",monospace;">{ms_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Summary
        n_ok = sum(1 for r in results.values() if r["ok"])
        n_total = len(results)
        sum_color = "#26a69a" if n_ok == n_total else "#fbbf24" if n_ok >= n_total - 1 else "#ef5350"
        st.markdown(
            f'<div style="margin-top:1rem;padding:0.8rem 1rem;background:#0a0d14;border:1px solid {sum_color};'
            f'border-radius:6px;text-align:center;">'
            f'  <span style="color:{sum_color};font-weight:700;font-size:1.1rem;">{n_ok}/{n_total}</span>'
            f'  <span style="color:#9ca3af;font-size:0.9rem;margin-left:0.5rem;">checks passing</span>'
            f'</div>', unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:2rem;text-align:center;'
            'background:#0a0d14;border:1px dashed #1a1f2c;border-radius:6px;">'
            "Click <b>Run all checks</b> to test every external integration."
            "</div>", unsafe_allow_html=True,
        )

    # Container info
    st.markdown("#### Process info")
    proc_info = [
        ("Loop heartbeat age", hb_text, hb_color),
        ("Halt flag", "set" if halted else "clear", "#ef5350" if halted else "#26a69a"),
        ("Killed for day", "yes" if (hb or {}).get("killed_for_day") else "no",
         "#ef5350" if (hb or {}).get("killed_for_day") else "#26a69a"),
        ("In trading window", "yes" if (hb or {}).get("in_window") else "no",
         "#26a69a" if (hb or {}).get("in_window") else "#6b7280"),
        ("Open positions (loop view)", str(n_open), "#e5e7eb"),
        ("Open positions (broker view)", str(n_open), "#e5e7eb"),  # they should match
    ]
    info_html = '<div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:0.5rem;">'
    for label, value, color in proc_info:
        info_html += (
            f'<div style="background:#0a0d14;border:1px solid #1a1f2c;border-radius:4px;'
            f'padding:0.6rem 0.8rem;">'
            f'  <div style="font-size:0.6rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">{label}</div>'
            f'  <div style="font-size:1rem;color:{color};font-weight:600;font-family:\"SF Mono\",monospace;">{value}</div>'
            f'</div>'
        )
    info_html += '</div>'
    st.markdown(info_html, unsafe_allow_html=True)
