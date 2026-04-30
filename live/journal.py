"""Trade journal: append-only writers for predictions, trades, SHAP, and daily summaries.

Output layout (all under reports/):

    reports/predictions.csv          one row per tick (~12k after 30 days)
    reports/trades.csv               one row per closed round-trip
    reports/trade_shap.csv           top-K SHAP contributions per trade
    reports/daily/YYYY-MM-DD.json    end-of-day rollup

All files are append-only and forgive missing earlier writes — restart-safe.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.env import root
from utils.logging import get

log = get("live.journal")


REPORTS = root() / "reports"
DAILY = REPORTS / "daily"


PRED_HEADERS = [
    "ts_utc", "symbol", "price",
    "p_down", "p_flat", "p_up",
    "thr_up", "thr_dn",
    "decision", "position_qty",
    "equity", "in_window",
]

TRADE_HEADERS = [
    "trade_id", "symbol", "side",
    "entry_at_utc", "exit_at_utc", "hold_minutes",
    "qty", "entry_price", "exit_price",
    "pnl_dollars", "pnl_bp", "hit",
    "p_up_at_entry", "p_down_at_entry",
    "thr_up", "thr_dn",
    "stop_bps", "target_bps", "rv_bps",
    "exit_reason",
    "equity_at_entry", "equity_at_exit",
]

SHAP_HEADERS = [
    "trade_id", "feature", "value", "shap", "abs_shap",
]


def _ensure_csv(path: Path, headers: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(headers)


def _append_row(path: Path, headers: list[str], row: dict):
    _ensure_csv(path, headers)
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([row.get(h, "") for h in headers])


@dataclass
class OpenTrade:
    """Lightweight in-memory record of a position currently held."""
    trade_id: str
    symbol: str
    side: str
    qty: int
    entry_at_utc: datetime
    entry_price: float
    p_up_at_entry: float
    p_down_at_entry: float
    thr_up: float
    thr_dn: float
    equity_at_entry: float
    stop_bps: float = 0.0           # vol-scaled stop distance set at entry
    target_bps: float = 0.0         # vol-scaled target distance (= stop * rr)
    rv_bps: float = 0.0             # realized vol seen at entry (diagnostic)
    discord_message_id: str | None = None   # for live-update PATCH


class Journal:
    """Single instance owned by the live loop."""

    def __init__(self):
        REPORTS.mkdir(exist_ok=True)
        DAILY.mkdir(exist_ok=True)
        self._predictions = REPORTS / "predictions.csv"
        self._trades = REPORTS / "trades.csv"
        self._shap = REPORTS / "trade_shap.csv"

    # ---- per-tick ----

    def log_prediction(self, *, ts_utc: datetime, symbol: str, price: float,
                        p_down: float, p_flat: float, p_up: float,
                        thr_up: float, thr_dn: float,
                        decision: str, position_qty: int,
                        equity: float, in_window: bool):
        try:
            _append_row(self._predictions, PRED_HEADERS, {
                "ts_utc": ts_utc.isoformat(),
                "symbol": symbol,
                "price": f"{price:.4f}",
                "p_down": f"{p_down:.4f}",
                "p_flat": f"{p_flat:.4f}",
                "p_up": f"{p_up:.4f}",
                "thr_up": f"{thr_up:.4f}",
                "thr_dn": f"{thr_dn:.4f}",
                "decision": decision,
                "position_qty": int(position_qty),
                "equity": f"{equity:.2f}",
                "in_window": int(bool(in_window)),
            })
        except Exception as e:
            log.warning("prediction journal write failed: %s", e)

    # ---- trades ----

    def open_trade(self, symbol: str, side: str, qty: int, entry_price: float,
                   p_up: float, p_dn: float, thr_up: float, thr_dn: float,
                   equity: float,
                   stop_bps: float = 0.0, target_bps: float = 0.0,
                   rv_bps: float = 0.0) -> OpenTrade:
        ts = datetime.now(timezone.utc)
        tid = f"T{ts.strftime('%Y%m%d-%H%M%S')}-{symbol}"
        return OpenTrade(
            trade_id=tid,
            symbol=symbol,
            side=side,
            qty=int(qty),
            entry_at_utc=ts,
            entry_price=float(entry_price),
            p_up_at_entry=float(p_up),
            p_down_at_entry=float(p_dn),
            thr_up=float(thr_up),
            thr_dn=float(thr_dn),
            equity_at_entry=float(equity),
            stop_bps=float(stop_bps),
            target_bps=float(target_bps),
            rv_bps=float(rv_bps),
        )

    def close_trade(self, ot: OpenTrade, exit_price: float, equity_at_exit: float,
                     reason: str = "horizon"):
        ts = datetime.now(timezone.utc)
        hold_min = max(1, int((ts - ot.entry_at_utc).total_seconds() / 60))
        sign = 1 if ot.side == "long" else -1
        pnl_per_share = sign * (exit_price - ot.entry_price)
        pnl_dollars = pnl_per_share * ot.qty
        pnl_bp = (pnl_per_share / max(ot.entry_price, 1e-9)) * 1e4
        try:
            _append_row(self._trades, TRADE_HEADERS, {
                "trade_id": ot.trade_id,
                "symbol": ot.symbol,
                "side": ot.side,
                "entry_at_utc": ot.entry_at_utc.isoformat(),
                "exit_at_utc": ts.isoformat(),
                "hold_minutes": hold_min,
                "qty": ot.qty,
                "entry_price": f"{ot.entry_price:.4f}",
                "exit_price": f"{exit_price:.4f}",
                "pnl_dollars": f"{pnl_dollars:.4f}",
                "pnl_bp": f"{pnl_bp:.2f}",
                "hit": 1 if pnl_dollars > 0 else 0,
                "p_up_at_entry": f"{ot.p_up_at_entry:.4f}",
                "p_down_at_entry": f"{ot.p_down_at_entry:.4f}",
                "thr_up": f"{ot.thr_up:.4f}",
                "thr_dn": f"{ot.thr_dn:.4f}",
                "stop_bps": f"{ot.stop_bps:.2f}",
                "target_bps": f"{ot.target_bps:.2f}",
                "rv_bps": f"{ot.rv_bps:.2f}",
                "exit_reason": reason,
                "equity_at_entry": f"{ot.equity_at_entry:.2f}",
                "equity_at_exit": f"{equity_at_exit:.2f}",
            })
        except Exception as e:
            log.warning("trade journal write failed: %s", e)
        return {
            "trade_id": ot.trade_id,
            "pnl_dollars": pnl_dollars,
            "pnl_bp": pnl_bp,
            "hold_minutes": hold_min,
        }

    # ---- SHAP ----

    def log_shap(self, trade_id: str, feature_names: list[str],
                 feature_values: np.ndarray, shap_values: np.ndarray,
                 top_k: int = 10):
        """Persist top-K SHAP contributors for the trade's entry-time prediction."""
        try:
            order = np.argsort(np.abs(shap_values))[::-1][:top_k]
            for idx in order:
                _append_row(self._shap, SHAP_HEADERS, {
                    "trade_id": trade_id,
                    "feature": feature_names[idx],
                    "value": f"{float(feature_values[idx]):.6g}",
                    "shap": f"{float(shap_values[idx]):.6g}",
                    "abs_shap": f"{abs(float(shap_values[idx])):.6g}",
                })
        except Exception as e:
            log.warning("shap journal write failed: %s", e)

    # ---- daily rollup ----

    def write_daily_summary(self, *, et_date: str, starting_equity: float,
                              ending_equity: float):
        """Read today's trades from the journal and write a per-day summary JSON."""
        try:
            if not self._trades.exists():
                return None
            trades = pd.read_csv(self._trades)
            if trades.empty:
                return None
            trades["entry_at_utc"] = pd.to_datetime(trades["entry_at_utc"])
            trades["et_date"] = trades["entry_at_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
            today = trades[trades["et_date"] == et_date]
            n = int(len(today))
            wins = int((today["pnl_dollars"] > 0).sum())
            pnl_d = float(today["pnl_dollars"].sum()) if n else 0.0
            pnl_bp_avg = float(today["pnl_bp"].mean()) if n else 0.0
            pnl_bp_sum = float(today["pnl_bp"].sum()) if n else 0.0
            day_pct = ((ending_equity - starting_equity) / starting_equity * 100) if starting_equity else 0.0
            payload = {
                "date": et_date,
                "trades": n,
                "winners": wins,
                "losers": n - wins,
                "hit_rate": (wins / n) if n else None,
                "starting_equity": round(starting_equity, 2),
                "ending_equity": round(ending_equity, 2),
                "day_pnl_dollars": round(pnl_d, 2),
                "day_pnl_pct_on_portfolio": round(day_pct, 4),
                "average_trade_bp": round(pnl_bp_avg, 2),
                "sum_trade_bp": round(pnl_bp_sum, 2),
                "first_trade_utc": today["entry_at_utc"].min().isoformat() if n else None,
                "last_trade_utc": today["entry_at_utc"].max().isoformat() if n else None,
            }
            (DAILY / f"{et_date}.json").write_text(json.dumps(payload, indent=2))
            return payload
        except Exception as e:
            log.warning("daily summary write failed: %s", e)
            return None
