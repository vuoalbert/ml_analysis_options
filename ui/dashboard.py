"""Streamlit dashboard for the live paper-trading loop.

Run:
    .venv/bin/streamlit run ui/dashboard.py
"""
from __future__ import annotations

import sys
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
    page_title="SPY — ML Paper Trader",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": None, "Get help": None, "Report a bug": None},
)

# Window options per timeframe (used both in sidebar and in the inline picker)
WINDOW_OPTIONS = {
    "1m":  ([4, 8, 24, 48], 24),
    "5m":  ([8, 24, 48, 168], 48),
    "15m": ([24, 72, 168, 336], 168),
    "1h":  ([72, 168, 336, 720], 336),
    "1d":  ([720, 2160, 8760, 17520], 2160),
}
WINDOW_LABELS = {4: "4h", 8: "8h", 24: "1d", 48: "2d", 72: "3d", 168: "1w",
                 336: "2w", 720: "30d", 2160: "90d", 8760: "1y", 17520: "2y"}


# ---------- Typography + layout polish ----------
st.markdown(
    """
    <style>
    /* Hide Streamlit chrome we don't need */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1600px;}

    /* Tighten the row gap for a more professional density */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {gap: 0.5rem;}

    /* Pill-style buttons for the timeframe selector */
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
        transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: #3b82f6;
        color: #d1d4dc;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: #3b82f6;
        border-color: #3b82f6;
        color: white;
    }
    /* Hide the radio bullet itself */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    /* Hide radio group label */
    div[data-testid="stRadio"] > label {
        font-size: 0.65rem !important;
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

    /* Metric labels — small, uppercase, muted */
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.70rem !important;
        color: #6b7280 !important;
        font-weight: 500;
    }
    /* Metric values — tabular numerics for alignment, denser sizing */
    [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        font-size: 1.15rem !important;
        font-weight: 600;
        color: #e5e7eb !important;
        font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricDelta"] {
        font-variant-numeric: tabular-nums;
        font-size: 0.78rem !important;
        font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
    }

    /* Section headings */
    h4 {
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-size: 0.72rem !important;
        font-weight: 600;
        color: #9ca3af !important;
        margin-bottom: 0.4rem !important;
        margin-top: 0.4rem !important;
    }

    /* Status dots */
    .dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .dot-green  {background: #26a69a; box-shadow: 0 0 6px rgba(38,166,154,.5);}
    .dot-red    {background: #ef5350;}
    .dot-amber  {background: #ffa726;}
    .dot-gray   {background: #4b5563;}

    /* Table density */
    [data-testid="stDataFrame"] {
        font-variant-numeric: tabular-nums;
    }

    /* Sidebar button styling */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Subtle dividers — barely visible on the deep-black background */
    hr {border-color: #1a1f2c !important; margin: 0.8rem 0 !important;}

    /* Tabular numerics on data tables for institutional alignment */
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace !important;
        font-size: 0.8rem !important;
    }

    /* Tighter section header spacing */
    h4 + div, h4 + p { margin-top: 0.2rem !important; }
    .block-container hr + h4 { margin-top: 0 !important; }

    /* Pull all section backgrounds toward true black for an OLED look */
    .stApp {background: #08090d !important;}
    section[data-testid="stSidebar"] {background: #0a0d14 !important;}
    [data-testid="stDataFrame"] {background: #0f1117 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _dot(color: str) -> str:
    return f'<span class="dot dot-{color}"></span>'


def _status_html(label: str, color: str) -> str:
    return f'<div style="font-size:1.2rem;font-weight:600;color:#e5e7eb;">{_dot(color)}{label}</div>'


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("#### Trading controls")
    halted = state.halted()
    if halted:
        st.markdown(
            '<div style="background:#2a1a1a;border-left:3px solid #ef5350;'
            'padding:8px 12px;border-radius:3px;margin-bottom:8px;">'
            '<div style="font-size:0.7rem;color:#ef5350;letter-spacing:0.1em;">HALTED</div>'
            '<div style="font-size:0.8rem;color:#9ca3af;">No new entries</div>'
            "</div>",
            unsafe_allow_html=True,
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
            "</div>",
            unsafe_allow_html=True,
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


# Defaults for the inline timeframe/window picker — set here so the KPI
# header can use them too. Real values come from the picker below.
symbol = "SPY"
timeframe = st.session_state.get("timeframe", "1m")
_opts, _default = WINDOW_OPTIONS[timeframe]
hours = st.session_state.get("hours", _default)

# ---------- Header KPIs ----------
acct = d.account_info()
hb = state.read_heartbeat()
hb_age = state.heartbeat_age_seconds()
art = d.artifact()
pos = d.current_position(symbol)

# Compute today's realized PnL from fills.
today_fills = d.recent_fills(lookback_hours=24, symbol=symbol)
today_trades = d.pair_entries_exits(today_fills) if not today_fills.empty else pd.DataFrame()
day_pnl = float(today_trades["pnl_dollars"].sum()) if not today_trades.empty else 0.0

ot = (hb or {}).get("open_trade") or {}
eq = float(acct.get("equity", 0))
last_price = (hb or {}).get("last_price") or 0
day_pnl_pct = (day_pnl / eq * 100) if eq else 0

# Market session state — open / pre-mkt / closed (NYSE: 9:30-16:00 ET)
from datetime import datetime as _dt
import pytz as _pytz
now_et_dt = _dt.now(_pytz.timezone("America/New_York"))
mins_into = now_et_dt.hour * 60 + now_et_dt.minute - (9 * 60 + 30)
is_weekday = now_et_dt.weekday() < 5
if not is_weekday:
    session_state, session_color = "WEEKEND", "#6b7280"
elif mins_into < 0:
    mins_to_open = -mins_into
    session_state, session_color = f"PRE-MKT  T-{mins_to_open//60:02d}:{mins_to_open%60:02d}", "#fbbf24"
elif mins_into < 390:  # 6.5h × 60
    mins_left = 390 - mins_into
    session_state, session_color = f"OPEN  {mins_left//60:02d}:{mins_left%60:02d} left", "#26a69a"
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

# Loop status
if halted:
    status_text, status_color, status_dot = "HALTED", "#ef5350", "red"
elif hb and hb.get("killed_for_day"):
    status_text, status_color, status_dot = "DD KILL", "#fbbf24", "amber"
elif hb and hb.get("in_window"):
    status_text, status_color, status_dot = "LIVE", "#26a69a", "green"
else:
    status_text, status_color, status_dot = "IDLE", "#6b7280", "gray"

# Position state
if pos:
    p_qty = int(pos["qty"])
    p_side = "LONG" if p_qty > 0 else "SHORT"
    p_color = "#26a69a" if p_qty > 0 else "#ef5350"
    p_pnl_pct = pos["unrealized_plpc"] * 100
    pnl_dollars_pos = float(pos.get("unrealized_pl", 0))
    pos_main = f"{p_side} {abs(p_qty)}"
    pos_sub = f"@${float(pos['avg_entry_price']):,.2f}  ·  P&L ${pnl_dollars_pos:+,.0f} ({p_pnl_pct:+.2f}%)"
    pos_color = "#26a69a" if pnl_dollars_pos >= 0 else "#ef5350"
elif ot.get("qty"):
    p_qty = int(ot["qty"])
    p_side = "LONG" if p_qty > 0 else "SHORT"
    pos_main = f"{p_side} {abs(p_qty)} (loop)"
    pos_sub = f"@${float(ot.get('entry_price', 0)):,.2f}"
    pos_color = "#fbbf24"
else:
    pos_main = "FLAT"
    pos_sub = "—"
    pos_color = "#6b7280"

# Day P&L color
pnl_color = "#26a69a" if day_pnl >= 0 else "#ef5350"

# ---- Render the institutional header strip ----
def _cell(label, value, sub="", value_color="#e5e7eb", monospace=True):
    family = '"SF Mono","Menlo","Monaco","Consolas",monospace' if monospace else 'inherit'
    sub_html = (f'<div style="font-size:0.65rem;color:#6b7280;'
                 f'font-family:{family};line-height:1.4;margin-top:0.15rem;'
                 f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{sub}</div>'
                 if sub else "")
    return (
        f'<div style="display:flex;flex-direction:column;justify-content:center;'
        f'padding:0 0.85rem;border-right:1px solid #1a1f2c;flex:1;min-width:0;">'
        f'<div style="font-size:0.58rem;color:#6b7280;text-transform:uppercase;'
        f'letter-spacing:0.12em;font-weight:600;margin-bottom:0.2rem;">{label}</div>'
        f'<div style="font-size:1.0rem;color:{value_color};font-weight:600;'
        f'font-family:{family};line-height:1.2;letter-spacing:-0.01em;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )

session_clock = now_et_dt.strftime("%H:%M:%S ET")
header = (
    '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
    'border-radius:6px;padding:0.6rem 0;margin-bottom:0.4rem;height:64px;">'
    + _cell(
        f"Symbol · {session_clock}",
        f"{symbol}  ${last_price:,.2f}" if last_price else symbol,
        sub=f'<span style="color:{session_color};font-weight:600;">{session_state}</span>',
    )
    + _cell(
        "Account",
        f"${eq:,.0f}",
        sub=f'BP ${float(acct.get("buying_power", 0)):,.0f}',
    )
    + _cell(
        "Position",
        pos_main,
        sub=pos_sub,
        value_color=pos_color,
    )
    + _cell(
        "Today P&L",
        f"${day_pnl:+,.2f}",
        sub=f"{day_pnl_pct:+.2f}% · {len(today_trades)} trades",
        value_color=pnl_color,
    )
    + _cell(
        "Heartbeat",
        f'<span style="color:{hb_color};">●</span> <span style="color:{hb_color};">{hb_text}</span>',
        sub="Alpaca: connected" if "error" not in acct else "Alpaca: down",
    )
    + _cell(
        "Status",
        f'<span style="color:{status_color};">{status_text}</span>',
        sub=f"thr {float(art.thresholds['up']):.2f} / {float(art.thresholds['down']):.2f}",
    )
    + '</div>'
)
st.markdown(header, unsafe_allow_html=True)

# Stop/Target/Entry strip — only when a position is open (loop or alpaca)
stop_px = ot.get("stop_price")
target_px = ot.get("target_price")
if stop_px and target_px and ot.get("entry_price"):
    entry_px_v = float(ot["entry_price"])
    cur_px_v = float(pos["current_price"]) if pos else last_price or entry_px_v
    side_v = ot.get("side", "long")
    # Progress to target / stop as %
    if side_v == "long":
        rng_total = target_px - stop_px
        progress = (cur_px_v - stop_px) / rng_total if rng_total else 0
    else:
        rng_total = stop_px - target_px
        progress = (stop_px - cur_px_v) / rng_total if rng_total else 0
    progress = max(0, min(1, progress)) * 100
    bar_color = "#26a69a" if progress > 50 else "#fbbf24" if progress > 30 else "#ef5350"
    st.markdown(
        '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
        'border-radius:6px;padding:0.55rem 0.85rem;margin-bottom:0.5rem;'
        'align-items:center;font-family:\"SF Mono\",\"Menlo\",monospace;'
        'font-size:0.78rem;">'
        '<div style="color:#6b7280;font-size:0.58rem;text-transform:uppercase;'
        'letter-spacing:0.12em;font-weight:600;margin-right:1rem;">Open Trade</div>'
        f'<span style="color:#ef5350;font-weight:600;">SL ${stop_px:,.2f}</span>'
        f'<div style="flex:1;margin:0 0.7rem;background:#1a1f2c;height:4px;'
        f'border-radius:2px;position:relative;">'
        f'<div style="background:{bar_color};width:{progress:.0f}%;height:100%;'
        f'border-radius:2px;"></div>'
        f'<div style="position:absolute;top:-3px;left:50%;width:1px;height:10px;'
        f'background:#9ca3af;"></div>'
        f'</div>'
        f'<span style="color:#26a69a;font-weight:600;">TP ${target_px:,.2f}</span>'
        f'<span style="color:#9ca3af;margin-left:1rem;">'
        f'now ${cur_px_v:,.2f} · entry ${entry_px_v:,.2f} · '
        f'<span style="color:{bar_color};">{progress:.0f}% to target</span>'
        f'</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------- Chart controls (timeframe pill bar + window picker) ----------
ctrl_l, ctrl_r = st.columns([2, 1])
with ctrl_l:
    timeframe = st.radio(
        "Timeframe",
        options=["1m", "5m", "15m", "1h", "1d"],
        horizontal=True,
        index=["1m", "5m", "15m", "1h", "1d"].index(timeframe),
        key="timeframe",
        label_visibility="visible",
    )
with ctrl_r:
    _opts, _default = WINDOW_OPTIONS[timeframe]
    if hours not in _opts:
        hours = _default
    hours = st.select_slider(
        "Window",
        options=_opts,
        value=hours,
        format_func=lambda h: WINDOW_LABELS.get(h, f"{h}h"),
        key="hours",
    )

# ---------- Chart ----------
bars = d.fetch_bars(symbol, hours=hours, timeframe=timeframe)
# Predictions are computed against minute bars — only show them on the 1m view
preds = d.replay_predictions(hours=hours) if timeframe == "1m" else pd.DataFrame()
fills = d.recent_fills(lookback_hours=max(hours, 48), symbol=symbol)
trades = d.pair_entries_exits(fills) if not fills.empty else pd.DataFrame()

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

thr = (float(art.thresholds["up"]), float(art.thresholds["down"]))
if bars.empty:
    st.info(
        f"No `{timeframe}` bars in the last {hours}h window — likely outside market hours. "
        f"Try a longer window in the sidebar, or switch to `1h` / `1d` timeframe.",
    )
else:
    fig = c.candlestick_with_trades(bars, trades, preds, thresholds=thr,
                                    event_marks=event_marks, title="",
                                    timeframe=timeframe,
                                    open_trade=(hb or {}).get("open_trade"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="candlestick")
st.caption(
    "Drag a box to zoom both axes. Drag the y-axis for price only, the bottom "
    "rangeslider for time. Click any legend item to hide a series. Double-click to reset."
)


# ---------- Trades + Attribution ----------
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.markdown("#### Trades — Last 48h")
    if trades.empty:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
            "No round-trips yet. Markers will populate once the live loop trades."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        show = trades.copy()
        show["Entry"] = show["entry_at"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        show["Exit"] = show["exit_at"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        show = show[["Entry", "Exit", "side", "entry_px", "exit_px", "qty",
                     "pnl_dollars", "pnl_bp"]]
        show.columns = ["Entry", "Exit", "Side", "Entry px", "Exit px", "Qty", "P&L $", "P&L bp"]
        st.dataframe(
            show.iloc[::-1].style.format({
                "Entry px": "${:,.2f}", "Exit px": "${:,.2f}",
                "Qty": "{:.0f}",
                "P&L $": "${:+,.2f}", "P&L bp": "{:+.1f}",
            }),
            hide_index=True, use_container_width=True, height=320,
        )
        total_pnl = trades["pnl_dollars"].sum()
        wins = (trades["pnl_dollars"] > 0).sum()
        hit = wins / len(trades) if len(trades) else 0
        st.markdown(
            f'<div style="color:#9ca3af;font-size:0.85rem;margin-top:0.4rem;">'
            f'<span style="color:#e5e7eb;font-weight:600;">{len(trades)}</span> trades · '
            f'Hit rate <span style="color:#e5e7eb;font-weight:600;">{hit*100:.1f}%</span> · '
            f'Cumulative <span style="color:{"#26a69a" if total_pnl>=0 else "#ef5350"};'
            f'font-weight:600;">${total_pnl:+.2f}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

with right:
    st.markdown("#### Feature Attribution — Latest Prediction")
    shap_df = d.shap_explain_latest()
    if shap_df is None:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
            "Waiting for fresh data to compute attribution."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(c.shap_bar(shap_df), use_container_width=True,
                        config={"displayModeBar": False}, key="shap_bar")


# ---------- Performance Stats + Risk panel ----------
st.markdown("<hr>", unsafe_allow_html=True)
all_fills = d.recent_fills(lookback_hours=24*30, symbol=symbol)
all_trades = d.pair_entries_exits(all_fills) if not all_fills.empty else pd.DataFrame()
stats = d.perf_stats(all_trades) if not all_trades.empty else {"n_trades": 0}

stats_left, stats_right = st.columns([1.4, 1.0], gap="large")

def _stat(label: str, value: str, color: str = "#e5e7eb") -> str:
    return (
        f'<div style="display:flex;flex-direction:column;flex:1;'
        f'padding:0 0.7rem;border-left:1px solid #1a1f2c;">'
        f'<div style="font-size:0.62rem;text-transform:uppercase;'
        f'letter-spacing:0.10em;color:#6b7280;font-weight:500;">{label}</div>'
        f'<div style="font-size:1.05rem;color:{color};font-weight:600;'
        f'font-variant-numeric:tabular-nums;line-height:1.4;">{value}</div>'
        f'</div>'
    )

with stats_left:
    st.markdown(
        '<div style="font-size:0.7rem;color:#6b7280;text-transform:uppercase;'
        'letter-spacing:0.10em;margin-bottom:0.4rem;">Performance — last 30 days</div>',
        unsafe_allow_html=True,
    )
    if stats.get("n_trades", 0) == 0:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
            "No closed trades yet. Stats populate after the loop completes round-trips."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        sharpe_color = "#26a69a" if stats["daily_sharpe"] >= 1 else "#fbbf24" if stats["daily_sharpe"] >= 0 else "#ef5350"
        pnl_color = "#26a69a" if stats["total_pnl"] >= 0 else "#ef5350"
        # Row 1
        st.markdown(
            '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
            'border-radius:4px;padding:0.75rem 0.4rem;">' +
            _stat("TOTAL P&L", f"${stats['total_pnl']:+,.2f}", pnl_color) +
            _stat("HIT RATE", f"{stats['hit_rate']*100:.1f}%") +
            _stat("SHARPE (D)", f"{stats['daily_sharpe']:.2f}", sharpe_color) +
            _stat("SHARPE (A)", f"{stats['sharpe_annualized']:.2f}", sharpe_color) +
            _stat("PROF FACTOR", f"{stats['profit_factor']:.2f}") +
            _stat("TRADES", f"{stats['n_trades']}") +
            '</div>',
            unsafe_allow_html=True,
        )
        # Row 2
        st.markdown(
            '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
            'border-radius:4px;padding:0.75rem 0.4rem;margin-top:0.4rem;">' +
            _stat("AVG TRADE", f"${stats['avg_trade']:+,.2f}") +
            _stat("BEST", f"${stats['best_trade']:+,.2f}", "#26a69a") +
            _stat("WORST", f"${stats['worst_trade']:+,.2f}", "#ef5350") +
            _stat("WIN/LOSS", f"{stats['win_loss_ratio']:.2f}") +
            _stat("MAX DD", f"${stats['max_dd']:+,.2f}", "#ef5350") +
            _stat("AVG/DAY", f"${stats['avg_per_day']:+,.2f}") +
            '</div>',
            unsafe_allow_html=True,
        )

with stats_right:
    st.markdown(
        '<div style="font-size:0.7rem;color:#6b7280;text-transform:uppercase;'
        'letter-spacing:0.10em;margin-bottom:0.4rem;">Risk — current</div>',
        unsafe_allow_html=True,
    )
    eq_now = float(acct.get("equity", 0))
    bp_now = float(acct.get("buying_power", 0))
    notional = abs(float(pos.get("market_value", 0))) if pos else 0
    leverage = notional / eq_now if eq_now else 0
    leverage_color = "#26a69a" if leverage < 1.5 else "#fbbf24" if leverage < 2.0 else "#ef5350"
    bp_used = (bp_now - notional) / bp_now if bp_now > 0 else 1
    bp_color = "#26a69a" if bp_used > 0.4 else "#fbbf24" if bp_used > 0.2 else "#ef5350"

    st.markdown(
        '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
        'border-radius:4px;padding:0.75rem 0.4rem;">' +
        _stat("EQUITY", f"${eq_now:,.0f}") +
        _stat("EXPOSURE", f"${notional:,.0f}") +
        _stat("LEVERAGE", f"{leverage:.2f}×", leverage_color) +
        '</div>',
        unsafe_allow_html=True,
    )
    pdt_status = "✓ ABOVE PDT" if eq_now > 25_000 else "⚠ PDT RISK" if eq_now > 24_500 else "✗ BELOW PDT"
    pdt_color = "#26a69a" if eq_now > 25_000 else "#fbbf24" if eq_now > 24_500 else "#ef5350"
    st.markdown(
        '<div style="display:flex;background:#0a0d14;border:1px solid #1a1f2c;'
        'border-radius:4px;padding:0.75rem 0.4rem;margin-top:0.4rem;">' +
        _stat("BP AVAIL", f"${bp_now - notional:,.0f}", bp_color) +
        _stat("BP USED", f"{(notional/bp_now*100 if bp_now else 0):.0f}%", bp_color) +
        _stat("PDT", pdt_status, pdt_color) +
        '</div>',
        unsafe_allow_html=True,
    )


# ---------- Daily P&L + Hour-of-day distribution ----------
st.markdown("<hr>", unsafe_allow_html=True)
pnl_left, pnl_right = st.columns([1.4, 1.0], gap="large")

with pnl_left:
    st.markdown("#### Daily P&L — last 30 days")
    if all_trades.empty:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
            "No daily P&L yet."
            "</div>", unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(c.daily_pnl_bars(all_trades, days=30),
                        use_container_width=True,
                        config={"displayModeBar": False}, key="daily_pnl")

with pnl_right:
    st.markdown("#### P&L by hour of day")
    if all_trades.empty:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.9rem;padding:1rem 0;">'
            "Awaiting closed trades."
            "</div>", unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(c.hourly_pnl_heatmap(all_trades),
                        use_container_width=True,
                        config={"displayModeBar": False}, key="hourly_heat")


# ---------- Live Equity Curve ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### Live equity curve")
st.plotly_chart(c.equity_curve(trades), use_container_width=True,
                config={"displayModeBar": False}, key="equity_curve")
