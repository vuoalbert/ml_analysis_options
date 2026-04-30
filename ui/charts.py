"""Plotly chart builders for the dashboard. TradingView-style dark theme."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DARK_BG = "#08090d"
GRID = "#1a1f2c"
UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"


def candlestick_with_trades(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    thresholds: tuple[float, float] = (0.57, 0.57),
    event_marks: pd.DataFrame | None = None,
    title: str = "SPY",
    timeframe: str = "1m",
    open_trade: dict | None = None,
) -> go.Figure:
    """Candlestick + entry/exit markers + volume + P(up)/P(down) signal panel.

    bars: index=UTC timestamp, cols open/high/low/close/volume
    trades: from data.pair_entries_exits, cols entry_at/exit_at/side/entry_px/exit_px/pnl_bp
    predictions: optional df with columns p_down, p_up
    """
    if bars.empty:
        return go.Figure()

    # Use ET for display (TradingView convention).
    bars = bars.copy()
    bars.index = bars.index.tz_convert("America/New_York")
    if predictions is not None and not predictions.empty:
        predictions = predictions.copy()
        predictions.index = predictions.index.tz_convert("America/New_York")
    if not trades.empty:
        trades = trades.copy()
        trades["entry_at"] = pd.to_datetime(trades["entry_at"]).dt.tz_convert("America/New_York")
        trades["exit_at"] = pd.to_datetime(trades["exit_at"]).dt.tz_convert("America/New_York")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.66, 0.12, 0.22],
        vertical_spacing=0.015,
    )

    # --- candlestick ---
    fig.add_trace(go.Candlestick(
        x=bars.index,
        open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"],
        increasing_line_color=UP_COLOR,
        decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR,
        decreasing_fillcolor=DOWN_COLOR,
        name="SPY",
        showlegend=False,
    ), row=1, col=1)

    # --- volume ---
    colors = [UP_COLOR if c >= o else DOWN_COLOR for o, c in zip(bars["open"], bars["close"])]
    fig.add_trace(go.Bar(
        x=bars.index, y=bars["volume"],
        marker_color=colors, marker_line_width=0,
        name="Volume", showlegend=False, opacity=0.6,
    ), row=2, col=1)

    # --- entries / exits ---
    if not trades.empty:
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]

        # Long entries
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs["entry_at"], y=longs["entry_px"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=13, color=UP_COLOR,
                            line=dict(color="white", width=1)),
                name="Long entry", hoverinfo="text",
                hovertext=[f"Long entry  ${p:.2f}" for p in longs["entry_px"]],
                showlegend=False,
            ), row=1, col=1)
            # Long exits
            fig.add_trace(go.Scatter(
                x=longs["exit_at"], y=longs["exit_px"],
                mode="markers",
                marker=dict(symbol="triangle-down-open", size=11, color=UP_COLOR,
                            line=dict(color=UP_COLOR, width=1.5)),
                name="Long exit", hoverinfo="text",
                hovertext=[f"Long exit  ${p:.2f}  ({pnl:+.1f} bp)"
                           for p, pnl in zip(longs["exit_px"], longs["pnl_bp"])],
                showlegend=False,
            ), row=1, col=1)

        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts["entry_at"], y=shorts["entry_px"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=13, color=DOWN_COLOR,
                            line=dict(color="white", width=1)),
                name="Short entry", hoverinfo="text",
                hovertext=[f"Short entry  ${p:.2f}" for p in shorts["entry_px"]],
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=shorts["exit_at"], y=shorts["exit_px"],
                mode="markers",
                marker=dict(symbol="triangle-up-open", size=11, color=DOWN_COLOR,
                            line=dict(color=DOWN_COLOR, width=1.5)),
                name="Short exit", hoverinfo="text",
                hovertext=[f"Short exit  ${p:.2f}  ({pnl:+.1f} bp)"
                           for p, pnl in zip(shorts["exit_px"], shorts["pnl_bp"])],
                showlegend=False,
            ), row=1, col=1)

        # Dashed line from entry to exit with pnl label.
        for _, t in trades.iterrows():
            color = UP_COLOR if t["pnl_bp"] >= 0 else DOWN_COLOR
            fig.add_trace(go.Scatter(
                x=[t["entry_at"], t["exit_at"]],
                y=[t["entry_px"], t["exit_px"]],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=1)

    # --- open-trade stop/target/entry levels ---
    # Drawn as full-width horizontal lines on the price panel so it's obvious
    # where the position will exit. Only render when the loop reports a live
    # vol-scaled position with both barriers set.
    if open_trade and open_trade.get("qty"):
        entry_px = open_trade.get("entry_price")
        stop_px = open_trade.get("stop_price")
        target_px = open_trade.get("target_price")
        for px, color, label in (
            (entry_px, "#9ca3af", "entry"),
            (stop_px,  "#ef5350", "stop"),
            (target_px, "#26a69a", "target"),
        ):
            if px is None:
                continue
            fig.add_hline(
                y=float(px),
                line=dict(color=color, width=1, dash="dot"),
                annotation_text=f"{label} ${px:,.2f}",
                annotation_position="top right",
                annotation_font=dict(color=color, size=10),
                row=1, col=1,
            )

    # --- event markers (vertical lines) ---
    # NOTE: plotly's add_vline crashes when annotation_text is set with a
    # non-numeric x (it calls (x+x)/2 internally). Split into add_shape +
    # add_annotation to avoid that broken code path entirely.
    if event_marks is not None and not event_marks.empty:
        for _, e in event_marks.iterrows():
            x_val = pd.Timestamp(e["ts"])
            fig.add_shape(
                type="line",
                x0=x_val, x1=x_val, y0=0, y1=1,
                yref="y domain",
                line=dict(color="#888", width=1, dash="dash"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=x_val, y=1, yref="y domain",
                text=e["label"], showarrow=False,
                xanchor="left", yanchor="top",
                font=dict(color="#aaa", size=9),
                row=1, col=1,
            )

    # --- signal panel ---
    up_thr, dn_thr = thresholds
    if predictions is not None and not predictions.empty:
        fig.add_trace(go.Scatter(
            x=predictions.index, y=predictions["p_up"],
            mode="lines", line=dict(color=UP_COLOR, width=1.5),
            name="P(up)", showlegend=True,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=predictions.index, y=predictions["p_down"],
            mode="lines", line=dict(color=DOWN_COLOR, width=1.5),
            name="P(down)", showlegend=True,
        ), row=3, col=1)
        fig.add_hline(y=up_thr, line=dict(color=UP_COLOR, width=1, dash="dash"),
                      row=3, col=1, opacity=0.5)
        fig.add_hline(y=dn_thr, line=dict(color=DOWN_COLOR, width=1, dash="dash"),
                      row=3, col=1, opacity=0.5)
        fig.add_hline(y=0.33, line=dict(color="#666", width=0.5, dash="dot"),
                      row=3, col=1, opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=0.30, x=0.01, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="#9ca3af")),
        hovermode="x unified",
        dragmode="zoom",
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                  color="#d1d4dc", size=11),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False,
                     tickfont=dict(size=10, color="#6b7280"))
    # Allow drag-zoom on the y-axis of the price/volume/signal panels.
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False,
                     fixedrange=False,
                     tickfont=dict(size=10, color="#6b7280"))
    fig.update_yaxes(title_text=None, row=1, col=1, autorange=True)
    fig.update_yaxes(title_text=None, row=2, col=1, showticklabels=False)
    fig.update_yaxes(title_text=None, row=3, col=1, range=[0, 1])

    # Range slider on the bottom subplot (signal panel) — drag to scrub
    # through history. shared_xaxes propagates the selection to row 1+2.
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04),
                     row=3, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=2, col=1)

    # Hide non-RTH gaps for INTRADAY only — daily/weekly bars don't have
    # intraday hours, and rangebreaks would erase legitimate trading days.
    if timeframe in ("1m", "5m", "15m", "1h"):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),        # no weekends
                dict(bounds=[16, 9.5], pattern="hour"),  # no after-hours / pre-market
            ],
        )
    elif timeframe == "1d":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def shap_bar(shap_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar of SHAP contributions for the latest prediction (class=up)."""
    if shap_df is None or shap_df.empty:
        return go.Figure()
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in shap_df["shap_up"]]
    labels = [f"{r['feature']} = {r['value']:.3g}" for _, r in shap_df.iterrows()]
    fig = go.Figure(go.Bar(
        x=shap_df["shap_up"],
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed", gridcolor=GRID, showline=False,
                   tickfont=dict(size=10, color="#9ca3af")),
        xaxis=dict(gridcolor=GRID, zerolinecolor="#374151",
                   tickfont=dict(size=10, color="#6b7280")),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                  color="#d1d4dc"),
        showlegend=False,
    )
    return fig


def equity_curve(trades: pd.DataFrame) -> go.Figure:
    if trades is None or trades.empty:
        return go.Figure()
    t = trades.sort_values("exit_at").copy()
    t["cum_pnl"] = t["pnl_dollars"].cumsum()
    fig = go.Figure(go.Scatter(
        x=t["exit_at"], y=t["cum_pnl"],
        mode="lines+markers",
        line=dict(color="#42a5f5", width=2),
        marker=dict(size=6),
        name="Cumulative PnL",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(gridcolor=GRID, zerolinecolor="#374151",
                   tickfont=dict(size=10, color="#6b7280")),
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=10, color="#6b7280")),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                  color="#d1d4dc"),
        showlegend=False,
    )
    return fig


def daily_pnl_bars(trades: pd.DataFrame, days: int = 30) -> go.Figure:
    """Bar chart of daily P&L over last N days. Green for up days, red for down."""
    if trades is None or trades.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG,
                            plot_bgcolor=DARK_BG, height=180,
                            margin=dict(l=10, r=10, t=10, b=10))
        return fig
    t = trades.copy()
    t["et_date"] = pd.to_datetime(t["exit_at"]).dt.tz_convert("America/New_York").dt.date
    daily = t.groupby("et_date")["pnl_dollars"].sum().reset_index()
    daily["et_date"] = pd.to_datetime(daily["et_date"])
    daily = daily.sort_values("et_date").tail(days)
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in daily["pnl_dollars"]]
    fig = go.Figure(go.Bar(
        x=daily["et_date"], y=daily["pnl_dollars"],
        marker_color=colors,
        hovertemplate="%{x|%b %d}<br>P&L: $%{y:+,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#374151", width=1))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=180, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(gridcolor=GRID, zerolinecolor="#374151",
                   tickfont=dict(size=9, color="#6b7280"),
                   tickformat="$,.0f"),
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=9, color="#6b7280"),
                   showgrid=False),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                  color="#d1d4dc"),
        showlegend=False, bargap=0.2,
    )
    return fig


def hourly_pnl_heatmap(trades: pd.DataFrame) -> go.Figure:
    """1×7 heatmap of hour-of-day P&L. Show which hours generate edge."""
    if trades is None or trades.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG,
                            plot_bgcolor=DARK_BG, height=130,
                            margin=dict(l=10, r=10, t=10, b=10))
        return fig
    t = trades.copy()
    t["entry_at"] = pd.to_datetime(t["entry_at"])
    t["et_hour"] = t["entry_at"].dt.tz_convert("America/New_York").dt.hour
    # Buckets: 9, 10, 11, 12, 13, 14, 15
    hours = list(range(9, 16))
    avg_pnl = []
    counts = []
    for h in hours:
        sub = t[t["et_hour"] == h]
        avg_pnl.append(sub["pnl_dollars"].mean() if not sub.empty else 0)
        counts.append(len(sub))
    text = [f"${p:+.0f}<br>n={n}" for p, n in zip(avg_pnl, counts)]
    fig = go.Figure(go.Heatmap(
        z=[avg_pnl],
        x=[f"{h}:30 ET" for h in hours],
        y=["Avg $/trade"],
        text=[text], texttemplate="%{text}",
        colorscale=[[0, DOWN_COLOR], [0.5, "#1a1f2c"], [1, UP_COLOR]],
        zmid=0,
        showscale=False,
        hovertemplate="%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=130, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#9ca3af")),
        xaxis=dict(showgrid=False, side="bottom",
                   tickfont=dict(size=10, color="#9ca3af")),
        font=dict(color="#d1d4dc"),
    )
    return fig


def calibration_plot(folds_df: pd.DataFrame) -> go.Figure:
    """Per-fold hit rate vs fold index — a proxy for calibration stability."""
    if folds_df is None or folds_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=folds_df["test_start"],
        y=folds_df["hit_rate"],
        marker_color=[UP_COLOR if h >= 0.5 else DOWN_COLOR for h in folds_df["hit_rate"]],
        name="Hit rate",
    ))
    fig.add_hline(y=0.5, line=dict(color="#888", dash="dash"), opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0.3, 0.75], gridcolor=GRID,
                   tickfont=dict(size=10, color="#6b7280"),
                   tickformat=".0%"),
        xaxis=dict(gridcolor=GRID, tickfont=dict(size=10, color="#6b7280")),
        font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                  color="#d1d4dc"),
        showlegend=False,
    )
    return fig
