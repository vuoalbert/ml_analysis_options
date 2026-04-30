"""Generate a 30-day rollup report from journal data.

Usage:
    python -m reports.summarize                # last 30 days
    python -m reports.summarize --days 60      # custom window
    python -m reports.summarize --since 2026-04-28
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.env import root
from model.artifact import load as load_artifact

REPORTS = root() / "reports"


def _load_trades(window_start: date) -> pd.DataFrame:
    p = REPORTS / "trades.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    df["entry_at_utc"] = pd.to_datetime(df["entry_at_utc"], utc=True)
    df["exit_at_utc"] = pd.to_datetime(df["exit_at_utc"], utc=True)
    df["et_date"] = df["entry_at_utc"].dt.tz_convert("America/New_York").dt.date
    df["et_hour"] = df["entry_at_utc"].dt.tz_convert("America/New_York").dt.hour
    return df[df["et_date"] >= window_start].copy()


def _load_predictions(window_start: date) -> pd.DataFrame:
    p = REPORTS / "predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["et_date"] = df["ts_utc"].dt.tz_convert("America/New_York").dt.date
    return df[df["et_date"] >= window_start].copy()


def _daily_sharpe(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    daily = trades.groupby("et_date")["pnl_dollars"].sum()
    if len(daily) < 5 or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(252))


def _max_dd_pct(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    eq = trades.sort_values("exit_at_utc")["pnl_dollars"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())


def _calibration(trades: pd.DataFrame) -> pd.DataFrame:
    """Bucket P(up)|P(down) into bins, report realized hit rate per bucket."""
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["max_p"] = df[["p_up_at_entry", "p_down_at_entry"]].max(axis=1)
    bins = [0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
    df["bucket"] = pd.cut(df["max_p"], bins=bins, right=False, include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(
        trades=("trade_id", "count"),
        hit_rate=("hit", "mean"),
        avg_pnl_bp=("pnl_bp", "mean"),
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    return out


def _by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("et_hour").agg(
        trades=("trade_id", "count"),
        hit_rate=("hit", "mean"),
        sum_pnl_bp=("pnl_bp", "sum"),
    ).reset_index()


def summarize(window_days: int = 30, since: date | None = None) -> dict:
    if since is None:
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).date()
    trades = _load_trades(since)
    preds = _load_predictions(since)

    art = load_artifact("latest")
    bt_holdout = art.metrics.get("holdout", {})
    bt_agg = art.metrics.get("aggregate", {})

    n = int(len(trades))
    wins = int((trades["hit"] == 1).sum()) if n else 0
    sum_pnl = float(trades["pnl_dollars"].sum()) if n else 0.0
    avg_bp = float(trades["pnl_bp"].mean()) if n else 0.0
    sum_bp = float(trades["pnl_bp"].sum()) if n else 0.0
    sharpe = _daily_sharpe(trades) if n else float("nan")
    hit = (wins / n) if n else float("nan")
    days_active = trades["et_date"].nunique() if n else 0

    summary = {
        "window_start": str(since),
        "window_end": str(date.today()),
        "trades": n,
        "winners": wins,
        "losers": n - wins,
        "active_days": int(days_active),
        "hit_rate": hit,
        "live_daily_sharpe": sharpe,
        "sum_pnl_dollars": round(sum_pnl, 2),
        "average_trade_bp": round(avg_bp, 2),
        "sum_trade_bp": round(sum_bp, 2),
        "max_drawdown_dollars": round(_max_dd_pct(trades), 2),
        "predictions_logged": int(len(preds)),
        "backtest_reference": {
            "holdout_daily_sharpe": bt_holdout.get("daily_sharpe"),
            "holdout_hit_rate": bt_holdout.get("hit_rate"),
            "holdout_avg_trade_bp": bt_holdout.get("avg_trade_bp"),
            "walk_forward_mean_sharpe": bt_agg.get("mean_daily_sharpe"),
        },
    }
    summary["calibration"] = _calibration(trades).to_dict(orient="records") if n else []
    summary["by_hour"] = _by_hour(trades).to_dict(orient="records") if n else []
    return summary


def to_markdown(summary: dict) -> str:
    bt = summary["backtest_reference"]
    sharpe = summary["live_daily_sharpe"]
    sharpe_str = f"{sharpe:.2f}" if not (sharpe is None or (isinstance(sharpe, float) and np.isnan(sharpe))) else "—"
    hit = summary["hit_rate"]
    hit_str = f"{hit*100:.1f}%" if hit and not (isinstance(hit, float) and np.isnan(hit)) else "—"
    bt_sharpe = bt.get("holdout_daily_sharpe")
    bt_hit = bt.get("holdout_hit_rate")

    lines = [
        f"# Live performance: {summary['window_start']} → {summary['window_end']}",
        "",
        "## Headline",
        "",
        f"| metric | live | backtest holdout | gap |",
        f"|---|---|---|---|",
        f"| Daily Sharpe | {sharpe_str} | {bt_sharpe:.2f} | "
        f"{(sharpe - bt_sharpe):+.2f}" if (sharpe is not None and bt_sharpe and not np.isnan(sharpe)) else "—",
        f"| Hit rate | {hit_str} | {bt_hit*100:.1f}% | "
        f"{(hit - bt_hit)*100:+.1f} pp" if (hit is not None and bt_hit and not np.isnan(hit)) else "—",
        f"| Avg trade (bp) | {summary['average_trade_bp']:+.2f} | "
        f"{bt.get('holdout_avg_trade_bp', 0):+.2f} | — |",
        "",
        "## Activity",
        "",
        f"- Trades: **{summary['trades']}** ({summary['winners']} wins / {summary['losers']} losses)",
        f"- Active days: **{summary['active_days']}**",
        f"- Sum P&L: **${summary['sum_pnl_dollars']:+,.2f}**",
        f"- Sum bp: **{summary['sum_trade_bp']:+.1f}**",
        f"- Max equity drawdown: **${summary['max_drawdown_dollars']:+,.2f}**",
        f"- Predictions logged: {summary['predictions_logged']}",
        "",
        "## Calibration",
        "",
    ]
    cal = summary.get("calibration") or []
    if cal:
        lines.append("| confidence bucket | trades | realized hit rate | avg bp |")
        lines.append("|---|---|---|---|")
        for r in cal:
            lines.append(
                f"| {r['bucket']} | {int(r['trades'])} | "
                f"{r['hit_rate']*100:.1f}% | {r['avg_pnl_bp']:+.2f} |"
            )
        lines.append("")
        lines.append("Well-calibrated model: realized hit rate climbs monotonically with the bucket.")
    else:
        lines.append("Not enough trades yet for calibration analysis.")

    lines += ["", "## Performance by hour of day", ""]
    by_hour = summary.get("by_hour") or []
    if by_hour:
        lines.append("| hour (ET) | trades | hit rate | sum bp |")
        lines.append("|---|---|---|---|")
        for r in by_hour:
            lines.append(
                f"| {int(r['et_hour']):02d}:00 | {int(r['trades'])} | "
                f"{r['hit_rate']*100:.1f}% | {r['sum_pnl_bp']:+.1f} |"
            )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--since", type=str, help="YYYY-MM-DD start date")
    args = ap.parse_args()
    since = date.fromisoformat(args.since) if args.since else None
    s = summarize(window_days=args.days, since=since)
    out = REPORTS / "summary.json"
    out.write_text(json.dumps(s, indent=2, default=str))
    md = REPORTS / "summary.md"
    md.write_text(to_markdown(s))
    print(json.dumps(s, indent=2, default=str))
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
