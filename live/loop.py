"""Live paper-trading loop against Alpaca.

Polls minute bars for SPY + cross-asset ETFs, computes features with the same
module used in training (features.build), predicts with the trained artifact,
and submits paper orders.

Risk rules (from config):
  - RTH only; skip first/last N minutes of the session
  - one position at a time
  - V2 EXIT MODE (default): stop_bps, target_bps, hold_min predicted per
    trade by the research_v2_exit_tight bundle from features at entry.
    Exit on stop OR target OR predicted-hold OR EOD-flat (whichever first).
  - Vol-scaled fallback (if v2 fails to load): stop = K × realized_vol_bps,
    target = 2 × stop, no horizon, EOD-flat catches anything else.
  - Risk-based qty = floor(equity × risk_pct / stop_distance_dollars), with
    risk_pct conviction-weighted (0.5%–1.0% based on entry probability)
  - Capped at 2.0× equity notional (Alpaca paper 2× day-trade BP)
  - Daily drawdown kill-switch halts new entries for the rest of the day

Recovered positions (read from Alpaca on boot) use the 15-min horizon as
a fallback since we don't have stop/target/hold levels for them — better
to exit quickly than hold an unknown-age position indefinitely.

State recovery: on boot we read existing positions from Alpaca and seed the
in-memory Position record. Without this, container restarts forget any open
position and the horizon exit silently never fires.

Usage:
    python -m live.loop                   # run until interrupted
    python -m live.loop --once            # one iteration then exit (smoke)
"""
from __future__ import annotations

import argparse
import json
import signal as _sig
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.common.exceptions import APIError

from utils.env import alpaca_keys, root
from utils.config import load as load_cfg
from utils.logging import get
from utils.calendar import NYSE
from features.build import build as build_features
from labels.build import forward_return  # unused live, but keeps parity
from model.artifact import load as load_artifact
from data_pull import yf_daily, fred
from data_pull.assemble import YF_DAILY_CROSS, ALPACA_CROSS, _daily_ffill, _align_minute
from live.journal import Journal
from live.discord import Discord
from research.vol_scaled_exits import compute_exit_plan, should_exit, ExitPlan
from research.state_recovery import recover_position

log = get("live.loop")
HALT_FLAG = root() / "halt.flag"
HEARTBEAT_FILE = root() / "logs" / "heartbeat.json"


@dataclass
class Position:
    qty: int = 0
    side: str | None = None  # "long" or "short"
    entry_ts: pd.Timestamp | None = None
    exit_due: pd.Timestamp | None = None
    entry_price: float | None = None
    stop_bps: float = 0.0       # vol-scaled barrier set at entry (stocks mode)
    target_bps: float = 0.0     # stocks mode
    open_trade: object | None = None   # journal.OpenTrade record for journaling at exit
    # Options-mode fields:
    option_symbol: str | None = None    # OCC symbol when trading options
    option_side: str | None = None      # "call" or "put"
    entry_premium: float = 0.0          # premium per share at entry (× 100 = $/contract)
    stop_pct: float = 0.0               # exit if premium drops by this fraction
    target_pct: float = 0.0             # exit if premium rises by this fraction


@dataclass
class State:
    position: Position = field(default_factory=Position)              # primary (stocks mode + legacy)
    option_positions: list[Position] = field(default_factory=list)    # multi-N options mode
    day_start_equity: float | None = None
    day_realized_pnl: float = 0.0
    killed_for_day: bool = False
    current_day: str | None = None


class TradeUpdater(threading.Thread):
    """Edits the open-trade Discord message every `interval` seconds with current P&L.

    Independent of the main 60s polling loop — fetches the latest SPY trade price
    directly from Alpaca so live unrealized P&L stays fresh between minute ticks.
    """

    def __init__(self, trader: "LiveTrader", message_id: str, ot, interval: int = 10):
        super().__init__(daemon=True, name=f"updater-{ot.trade_id}")
        self.trader = trader
        self.message_id = message_id
        self.ot = ot
        self.interval = interval
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _latest_price(self) -> float | None:
        """Get the freshest price for self.ot.symbol via Alpaca's latest-trade endpoint."""
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=self.ot.symbol, feed=DataFeed.IEX)
            resp = self.trader.data.get_stock_latest_trade(req)
            t = resp.get(self.ot.symbol)
            if t is None:
                return None
            return float(t.price)
        except Exception:
            # Fall back to latest quote mid if trades unavailable.
            try:
                qreq = StockLatestQuoteRequest(symbol_or_symbols=self.ot.symbol, feed=DataFeed.IEX)
                q = self.trader.data.get_stock_latest_quote(qreq).get(self.ot.symbol)
                if q is None:
                    return None
                bid = float(q.bid_price); ask = float(q.ask_price)
                return (bid + ask) / 2.0 if (bid > 0 and ask > 0) else None
            except Exception:
                return None

    def run(self):
        # First update fires almost immediately so the user sees motion.
        first = True
        while not self._stop.wait(2 if first else self.interval):
            first = False
            try:
                px = self._latest_price()
                if px is None:
                    continue
                now = pd.Timestamp.now(tz="UTC")
                mins_held = max(0, int((now - pd.Timestamp(self.ot.entry_at_utc)).total_seconds() / 60))
                # No 15-min horizon now — display minutes until EOD-flat instead
                # (16:00 ET close minus flat_by_minutes_before_close).
                now_et = now.tz_convert("America/New_York")
                flat_min_before = int(self.trader.cfg["risk"]["flat_by_minutes_before_close"])
                eod_flat = now_et.replace(hour=16, minute=0, second=0, microsecond=0) - pd.Timedelta(minutes=flat_min_before)
                mins_remaining = max(0, int((eod_flat - now_et).total_seconds() / 60))
                self.trader.discord.update_open_trade(
                    message_id=self.message_id,
                    side=self.ot.side, symbol=self.ot.symbol, qty=self.ot.qty,
                    entry_price=self.ot.entry_price, current_price=px,
                    p_up=self.ot.p_up_at_entry, p_dn=self.ot.p_down_at_entry,
                    mins_held=mins_held, mins_remaining=mins_remaining,
                    stop_bps=getattr(self.ot, "stop_bps", 0.0),
                    target_bps=getattr(self.ot, "target_bps", 0.0),
                )
            except Exception as e:
                log.warning("trade updater tick failed: %s", e)


class LiveTrader:
    def __init__(self, cfg_name: str = "v1"):
        self.cfg = load_cfg(cfg_name)
        self.symbol = self.cfg["universe"]["symbol"]
        self.cross = self.cfg["universe"]["cross_asset"]
        self.horizon = int(self.cfg["label"]["horizon_min"])
        key, sec = alpaca_keys()
        self.trading = TradingClient(key, sec, paper=bool(self.cfg["live"]["paper"]))
        self.data = StockHistoricalDataClient(key, sec)
        # Mode dispatch — "stocks" (legacy) or "options" (this repo's primary path)
        self.mode = self.cfg.get("strategy", {}).get("mode", "stocks")
        self.option_data = None
        if self.mode == "options":
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self.option_data = OptionHistoricalDataClient(key, sec)
            log.info("strategy mode: OPTIONS — entries will buy 0DTE contracts")
        else:
            log.info("strategy mode: STOCKS")
        self.art = load_artifact("latest")
        self.state = State()
        self._stop = False
        self.journal = Journal()
        self.discord = Discord()
        self._trade_updater: TradeUpdater | None = None
        # SHAP explainer is heavy to construct — do it once at startup.
        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.art.booster)
        except Exception as e:
            log.warning("SHAP explainer unavailable: %s", e)
            self._shap_explainer = None
        _sig.signal(_sig.SIGINT, self._on_stop)
        _sig.signal(_sig.SIGTERM, self._on_stop)
        log.info("loaded artifact: features=%d thresholds=%s", len(self.art.feature_cols), self.art.thresholds)

        # Optionally load v2 exit predictors (predicted stop/target/hold per trade).
        # If the bundle fails to load, fall through to vol-scaled exits.
        self.v2_exits = None
        if self.cfg["risk"].get("use_v2_exits", False):
            try:
                from research.v2_predictors import load as load_v2_exits
                v2_dir = root() / "artifacts" / self.cfg["risk"]["v2_exit_artifact"]
                self.v2_exits = load_v2_exits(art_dir=v2_dir)
                log.info("loaded v2 exit bundle from %s", v2_dir)
                log.info("  stop RMSE %.2f | target RMSE %.2f | hold RMSE %.1f min",
                         self.v2_exits.metrics.get("stop_rmse", float("nan")),
                         self.v2_exits.metrics.get("target_rmse", float("nan")),
                         self.v2_exits.metrics.get("hold_rmse", float("nan")))
            except Exception as e:
                log.warning("v2 exit bundle load failed (%s); falling back to vol-scaled", e)
                self.v2_exits = None
        if self.discord.enabled:
            try:
                eq = float(self.trading.get_account().equity)
                self.discord.startup(self.symbol, eq, paper=bool(self.cfg["live"]["paper"]))
            except Exception:
                pass

        # State recovery: if Alpaca already holds the symbol (e.g. after a
        # container restart) seed the in-memory Position so the exit logic
        # actually fires. Without this the loop silently bypasses horizon
        # exits on any position it didn't open itself.
        try:
            rec = recover_position(self.trading, self.symbol, self.horizon)
        except Exception as e:
            log.warning("state recovery failed: %s", e)
            rec = None
        if rec is not None:
            log.warning("STATE RECOVERY: %s", rec.note)
            self.state.position = Position(
                qty=rec.qty, side=rec.side,
                entry_ts=rec.entry_ts, exit_due=rec.exit_due,
                entry_price=rec.entry_price,
                # No vol-scaled barriers for recovered positions — we don't
                # know the realized vol at the original entry. Loop will exit
                # at horizon (or already-past exit_due → next iterate).
                stop_bps=0.0, target_bps=0.0,
                open_trade=None,
            )

    def _on_stop(self, *_):
        log.info("stop signal received")
        self._stop = True

    # ---- data ----

    def _fetch_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            adjustment="all",
        )
        try:
            resp = self.data.get_stock_bars(req).df
        except APIError as e:
            log.warning("bars fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()
        if resp is None or resp.empty:
            return pd.DataFrame()
        df = resp.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.drop(columns=[c for c in ("symbol",) if c in df.columns])
        return df

    def _build_live_dataframe(self) -> pd.DataFrame:
        """Build the assembled minute frame for the last ~60 trading days so features have enough warmup."""
        now = pd.Timestamp.now(tz="UTC")
        start = now - pd.Timedelta(days=90)

        # Minute grid from NYSE schedule.
        sched = NYSE.schedule(start_date=(now - pd.Timedelta(days=90)).date(), end_date=now.date())
        if sched.empty:
            return pd.DataFrame()
        import pandas_market_calendars as mcal
        minute_idx = mcal.date_range(sched, frequency="1min").tz_convert("UTC")
        minute_idx = minute_idx[minute_idx <= now]

        spy = self._fetch_bars(self.symbol, start, now)
        cols = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
        out = _align_minute(spy, minute_idx, cols, self.symbol.lower())

        for s in self.cross:
            if s in ALPACA_CROSS:
                df = self._fetch_bars(s, start, now)
                out = out.join(_align_minute(df, minute_idx, ["close", "volume"], s.lower()))

        for s, prefix in YF_DAILY_CROSS.items():
            if s in self.cross:
                try:
                    df = yf_daily.pull(s, str((now - pd.Timedelta(days=120)).date()), str(now.date()), use_cache=False)
                    if not df.empty and "close" in df.columns:
                        ff = _daily_ffill(df[["close"]].rename(columns={"close": f"{prefix}_close"}),
                                          minute_idx, lag_days=0)
                        out = out.join(ff)
                except Exception as e:
                    log.warning("live yf fetch failed for %s: %s", s, e)

        # FRED macro features (daily, 1-day-lagged). pull_many has its own
        # stale-cache fallback for transient API outages; this outer try/except
        # is belt-and-suspenders — even if fred returns something pathological,
        # the live loop should not crash. Model handles missing macro as NaN.
        try:
            fred_df = fred.pull_many(self.cfg.get("fred_series", []),
                                     str((now - pd.Timedelta(days=120)).date()),
                                     str(now.date()))
            if not fred_df.empty:
                ff = _daily_ffill(fred_df, minute_idx, lag_days=1)
                ff.columns = [f"fred_{c}" for c in ff.columns]
                out = out.join(ff)
        except Exception as e:
            log.warning("live FRED block failed entirely: %s — proceeding without macro features", e)

        # Event flags + session timing.
        from utils.calendar import is_fomc_day, is_zero_dte, minutes_into_session
        out["evt_fomc_day"] = is_fomc_day(minute_idx).astype(np.int8).values
        out["evt_zero_dte"] = is_zero_dte(minute_idx).astype(np.int8).values
        out["session_min"] = minutes_into_session(minute_idx).values

        out = out.dropna(subset=[f"{self.symbol.lower()}_close"])
        return out

    # ---- trading ----

    def _account_equity(self) -> float:
        try:
            acct = self.trading.get_account()
            return float(acct.equity)
        except Exception as e:
            log.warning("account fetch failed: %s", e)
            return 0.0

    def _reset_day_if_needed(self, now_et: pd.Timestamp):
        day = now_et.strftime("%Y-%m-%d")
        if self.state.current_day != day:
            # Write the previous day's summary before rolling over (only if we had a session).
            if self.state.current_day is not None and self.state.day_start_equity is not None:
                try:
                    summary = self.journal.write_daily_summary(
                        et_date=self.state.current_day,
                        starting_equity=self.state.day_start_equity,
                        ending_equity=self._account_equity(),
                    )
                    if self.discord.enabled and summary and summary["trades"]:
                        self.discord.session_summary(
                            date=summary["date"],
                            trades=summary["trades"],
                            wins=summary["winners"],
                            pnl_dollars=summary["day_pnl_dollars"],
                            pnl_bp=summary["sum_trade_bp"],
                            starting_equity=summary["starting_equity"],
                            ending_equity=summary["ending_equity"],
                        )
                except Exception as e:
                    log.warning("daily rollover failed: %s", e)
            self.state.current_day = day
            self.state.day_start_equity = self._account_equity()
            self.state.day_realized_pnl = 0.0
            self.state.killed_for_day = False
            log.info("new session %s start_equity=%.2f", day, self.state.day_start_equity)

    def _within_trading_window(self, now_et: pd.Timestamp) -> bool:
        open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        if now_et < open_t or now_et >= close_t:
            return False
        mins_from_open = (now_et - open_t).total_seconds() / 60
        mins_to_close = (close_t - now_et).total_seconds() / 60
        if mins_from_open < self.cfg["risk"]["skip_first_minutes"]:
            return False
        if mins_to_close < self.cfg["risk"]["skip_last_minutes"]:
            return False
        return True

    def _flat_all(self, reason: str = "horizon"):
        # Stop the live-updater thread first so it doesn't race with the final edit.
        if self._trade_updater is not None:
            try:
                self._trade_updater.stop()
                self._trade_updater.join(timeout=1.0)
            except Exception:
                pass
            self._trade_updater = None

        try:
            poss = self.trading.get_all_positions()
        except Exception as e:
            log.warning("get_all_positions failed: %s", e)
            return
        ot = self.state.position.open_trade if self.state.position else None
        last_px = None
        for p in poss:
            try:
                self.trading.close_position(p.symbol)
                last_px = float(p.current_price)
                log.info("closed %s qty=%s", p.symbol, p.qty)
            except Exception as e:
                log.warning("close failed for %s: %s", p.symbol, e)
        equity = self._account_equity()
        if ot is not None and last_px is not None:
            try:
                trade_summary = self.journal.close_trade(ot, exit_price=last_px,
                                                          equity_at_exit=equity, reason=reason)
                if self.discord.enabled:
                    # Prefer editing the existing live message into the final state.
                    if ot.discord_message_id:
                        self.discord.finalize_trade(
                            message_id=ot.discord_message_id,
                            side=ot.side, symbol=ot.symbol, qty=ot.qty,
                            entry_price=ot.entry_price, exit_price=last_px,
                            pnl_dollars=trade_summary["pnl_dollars"],
                            pnl_bp=trade_summary["pnl_bp"],
                            hold_minutes=trade_summary["hold_minutes"],
                        )
                    else:
                        # Fallback: post a new exit embed if we never got a message_id.
                        self.discord.trade_exit(
                            side=ot.side, symbol=ot.symbol, qty=ot.qty,
                            entry_px=ot.entry_price, exit_px=last_px,
                            pnl_dollars=trade_summary["pnl_dollars"],
                            pnl_bp=trade_summary["pnl_bp"],
                            hold_minutes=trade_summary["hold_minutes"],
                        )
            except Exception as e:
                log.warning("trade close journaling failed: %s", e)
        self.state.position = Position()
        # Multi-N options mode: also clear our internal list and journal each.
        if self.state.option_positions:
            try:
                from live.options import get_premium_quote
                for op in list(self.state.option_positions):
                    if op.open_trade is not None:
                        bid, _ = get_premium_quote(self.option_data, op.option_symbol)
                        exit_premium = bid if bid > 0 else op.entry_premium
                        self.journal.close_trade(op.open_trade,
                                                  exit_price=exit_premium,
                                                  equity_at_exit=equity, reason=reason)
            except Exception as e:
                log.warning("multi-N journal close failed: %s", e)
            self.state.option_positions.clear()

    def _submit(self, side: OrderSide, qty: int):
        if qty <= 0:
            return None
        req = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            o = self.trading.submit_order(order_data=req)
            log.info("ORDER %s %d %s status=%s", side.value, qty, self.symbol, getattr(o, "status", "?"))
            return o
        except Exception as e:
            log.warning("submit_order failed: %s", e)
            return None

    def _flat_options(self, reason: str = "exit", pos: Position | None = None):
        """Close one or all open option positions via market sell.

        If `pos` is given, close only that position and remove it from the list.
        If `pos` is None, close every open option position (flat-everything; used
        for EOD-flat or kill-switch).
        """
        targets: list[Position] = []
        if pos is not None:
            targets.append(pos)
        elif self.state.option_positions:
            targets.extend(self.state.option_positions)
        elif (self.state.position and self.state.position.option_symbol
              and self.state.position.qty != 0):
            # Legacy single-position fallback
            targets.append(self.state.position)
        if not targets:
            return

        from live.options import get_premium_quote
        equity = self._account_equity()
        for p in targets:
            if not p.option_symbol or p.qty == 0:
                continue
            try:
                req = MarketOrderRequest(
                    symbol=p.option_symbol, qty=p.qty,
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                )
                o = self.trading.submit_order(order_data=req)
                log.info("OPTION CLOSE %s qty=%d reason=%s status=%s",
                          p.option_symbol, p.qty, reason, getattr(o, "status", "?"))
            except Exception as e:
                log.warning("option close failed for %s: %s", p.option_symbol, e)

            # Best-effort journal close at latest bid
            try:
                bid, _ = get_premium_quote(self.option_data, p.option_symbol)
                exit_premium = bid if bid > 0 else p.entry_premium
                if p.open_trade is not None:
                    self.journal.close_trade(p.open_trade,
                                              exit_price=exit_premium,
                                              equity_at_exit=equity, reason=reason)
            except Exception as e:
                log.warning("option journal close failed: %s", e)

            if self.discord.enabled:
                try:
                    self.discord.embed(
                        title=f"OPT CLOSE {p.option_symbol}",
                        description=f"reason={reason}  qty={p.qty}",
                    )
                except Exception:
                    pass

            # Remove from state lists
            if p in self.state.option_positions:
                self.state.option_positions.remove(p)
            if self.state.position is p or self.state.position.option_symbol == p.option_symbol:
                self.state.position = Position()

    def _submit_option(self, occ_symbol: str, qty_contracts: int, side: OrderSide):
        """Submit an option market order. Same shape as _submit but for options."""
        if qty_contracts <= 0:
            return None
        req = MarketOrderRequest(
            symbol=occ_symbol,
            qty=qty_contracts,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            o = self.trading.submit_order(order_data=req)
            log.info("OPTION ORDER %s %d %s status=%s",
                      side.value, qty_contracts, occ_symbol,
                      getattr(o, "status", "?"))
            return o
        except Exception as e:
            log.warning("option submit_order failed: %s", e)
            return None

    def _plan_options_entry(self, last_price: float, model_side: str, p: float):
        """Pick an option contract per config and compute qty.

        model_side: "long" → buy a call, "short" → buy a put.
        Returns the OptionPlan from live.options or None if no contract available.
        Now respects:
          • options.expiration  (e.g., "7_business_days" / "same_day" / "next_friday")
          • options.conviction_min  (gate entry on max(p_up, p_dn))
          • options.max_concurrent_positions (checked by caller, not here)
        """
        from datetime import date
        from live.options import (
            pick_contract, get_premium_quote, plan_options_entry, resolve_expiration,
        )
        opt_cfg = self.cfg.get("strategy", {}).get("options", {})
        underlying = opt_cfg.get("underlying", "SPY")
        moneyness = opt_cfg.get("moneyness", "atm")
        itm_offset_pct = float(opt_cfg.get("itm_offset_pct", 0.005))
        max_qty = int(opt_cfg.get("max_qty_contracts", 10))
        risk_pct = float(opt_cfg.get("risk_pct_per_trade", 0.01))
        expiration_spec = opt_cfg.get("expiration", "same_day")
        conviction_min = float(opt_cfg.get("conviction_min", 0.0))

        # Hard conviction gate — refuse low-confidence signals entirely.
        if p < conviction_min:
            log.info("options: conviction p=%.3f < min=%.3f, skipping", p, conviction_min)
            return None

        # Conviction-weighted risk_pct (same conviction logic as stocks)
        risk = self.cfg.get("risk", {})
        if "conviction_lo" in risk and "conviction_hi" in risk:
            lo = float(risk["conviction_lo"])
            hi = float(risk["conviction_hi"])
            s = max(0.0, min(1.0, (p - lo) / max(hi - lo, 1e-9)))
            risk_pct = risk_pct + risk_pct * s   # up to 2× base when high conviction

        # Resolve expiration date from the config string (e.g. "7_business_days" → +7 BD from today)
        expiration_date = resolve_expiration(expiration_spec)

        side = "call" if model_side == "long" else "put"
        contract = pick_contract(
            trading=self.trading, underlying=underlying, side=side,
            last_price=last_price, expiration=expiration_date, moneyness=moneyness,
            itm_offset_pct=itm_offset_pct,
        )
        if contract is None:
            log.warning("options: no contract found for %s %s near %s exp=%s",
                         underlying, side, last_price, expiration_date)
            return None

        bid, ask = get_premium_quote(self.option_data, contract.occ_symbol)
        if ask <= 0:
            log.warning("options: bad quote for %s (bid=%s ask=%s)",
                         contract.occ_symbol, bid, ask)
            return None

        eq = self._account_equity()
        plan = plan_options_entry(
            contract=contract, bid=bid, ask=ask, equity=eq,
            risk_pct=risk_pct, max_qty=max_qty, min_qty=1,
        )
        return plan

    def _plan_exit(self, df: pd.DataFrame, last_price: float, side: str,
                    p: float | None = None, feature_row=None):
        """Compute stop, target, qty (and predicted hold) for a fresh entry.

        Two paths:
          • V2 path (if self.v2_exits is loaded AND feature_row provided):
            stop_bps, target_bps, hold_min are PREDICTED from features by the
            v2 exit bundle. Conviction sizing still applies on top.
          • Vol-scaled fallback: original K-based + 1:2 R/R + no horizon.

        Returns the ExitPlan with optional `_predicted_hold_min` attribute set
        when v2 path is taken. Caller checks plan.qty > 0.
        """
        risk = self.cfg["risk"]
        eq = self._account_equity()

        # Conviction-weighted risk_pct (defaults to legacy fixed value if config absent)
        if p is not None and "risk_pct_base" in risk and "risk_pct_max" in risk:
            lo = float(risk["conviction_lo"])
            hi = float(risk["conviction_hi"])
            base = float(risk["risk_pct_base"])
            top = float(risk["risk_pct_max"])
            s = max(0.0, min(1.0, (p - lo) / max(hi - lo, 1e-9)))
            risk_pct = base + (top - base) * s
        else:
            risk_pct = float(risk["risk_pct_per_trade"])

        # ---- V2 exit-prediction path ----
        if self.v2_exits is not None and feature_row is not None:
            try:
                # The entry model may use more features than the v2 exit
                # predictors were trained on (e.g. mtf_/vp_ extensions).
                # Slice the feature row to match the v2 exit feature schema.
                exit_cols = list(self.v2_exits.feature_cols)
                missing = [c for c in exit_cols if c not in feature_row.columns]
                if missing:
                    fr = feature_row.copy()
                    for c in missing:
                        fr[c] = np.nan
                    feat_arr = fr[exit_cols].values.reshape(1, -1)
                else:
                    feat_arr = feature_row[exit_cols].values.reshape(1, -1)
                stop_bps = float(np.clip(
                    self.v2_exits.stop_model.predict(feat_arr)[0],
                    float(risk["min_stop_bps"]),
                    float(risk["max_stop_bps"]),
                ))
                target_bps_raw = float(self.v2_exits.target_model.predict(feat_arr)[0])
                target_bps = float(np.clip(
                    target_bps_raw,
                    stop_bps * float(risk.get("v2_target_min_rr", 1.0)),
                    float(risk.get("v2_target_max_bps", 200.0)),
                ))
                hold_min = int(np.clip(
                    round(float(self.v2_exits.hold_model.predict(feat_arr)[0])),
                    int(risk.get("v2_hold_min", 5)),
                    int(risk.get("v2_hold_max", 390)),
                ))

                # Risk-based sizing using predicted stop
                risk_dollars = eq * risk_pct
                stop_distance_dollars = last_price * stop_bps / 1e4
                if stop_distance_dollars <= 0:
                    return None
                qty_risk = int(risk_dollars // stop_distance_dollars)
                qty_cap = int((eq * float(risk.get("max_notional_frac", 1.5))) //
                              max(last_price, 1e-9))
                qty = max(1, min(qty_risk, qty_cap))

                plan = ExitPlan(
                    stop_bps=stop_bps, target_bps=target_bps, qty=qty,
                    risk_dollars=risk_dollars, rv_bps=0.0,
                    note=(f"V2 stop={stop_bps:.1f} tgt={target_bps:.1f} "
                          f"hold={hold_min}m qty={qty} risk=${risk_dollars:.0f}"
                          f" | p={p:.3f} risk_pct={risk_pct*100:.2f}%"),
                )
                # Stash predicted hold for the caller (set Position.exit_due)
                plan._predicted_hold_min = hold_min
                return plan
            except Exception as e:
                log.warning("v2 prediction failed (%s); falling back to vol-scaled", e)

        # ---- Vol-scaled fallback ----
        col = f"{self.symbol.lower()}_close"
        closes = df[col].dropna().values[-(int(risk["vol_lookback_min"]) + 1):]
        plan = compute_exit_plan(
            bars_close=closes,
            entry_price=last_price,
            equity=eq,
            side=side,
            K=float(risk["vol_scaled_K"]),
            rr_ratio=float(risk["vol_scaled_rr"]),
            risk_pct=risk_pct,
            lookback_min=int(risk["vol_lookback_min"]),
            min_stop_bps=float(risk["min_stop_bps"]),
            max_stop_bps=float(risk["max_stop_bps"]),
            max_notional_frac=float(risk.get("max_notional_frac", 1.5)),
        )
        if p is not None:
            plan.note = f"{plan.note} | p={p:.3f} risk_pct={risk_pct*100:.2f}%"
        return plan

    def _check_dd_kill(self) -> bool:
        eq = self._account_equity()
        if self.state.day_start_equity and self.state.day_start_equity > 0:
            dd = (eq - self.state.day_start_equity) / self.state.day_start_equity
            if dd <= -float(self.cfg["risk"]["daily_dd_kill_frac"]):
                if not self.state.killed_for_day:
                    log.warning("daily DD kill-switch hit dd=%.4f; halting new entries", dd)
                    if self.discord.enabled:
                        self.discord.dd_kill(dd_pct=dd, equity=eq)
                self.state.killed_for_day = True
                return True
        return False

    # ---- main iteration ----

    def _write_heartbeat(self, now_utc: pd.Timestamp, last_price: float | None = None,
                          p_up: float | None = None, p_dn: float | None = None,
                          in_window: bool = False):
        """Write a small JSON the dashboard can read to check process health + last signal."""
        try:
            pos = self.state.position
            payload = {
                "ts_utc": now_utc.isoformat(),
                "in_window": bool(in_window),
                "halted": HALT_FLAG.exists(),
                "position_qty": int(pos.qty),
                "killed_for_day": bool(self.state.killed_for_day),
                "last_price": last_price,
                "p_up": p_up,
                "p_dn": p_dn,
                "thr_up": float(self.art.thresholds["up"]),
                "thr_dn": float(self.art.thresholds["down"]),
                # Open-trade detail so the dashboard can render stop/target/qty
                "open_trade": {
                    "qty": int(pos.qty),
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "stop_bps": float(pos.stop_bps),
                    "target_bps": float(pos.target_bps),
                    "stop_price": (pos.entry_price * (1 - pos.stop_bps / 1e4)
                                    if pos.side == "long"
                                    else pos.entry_price * (1 + pos.stop_bps / 1e4))
                                   if pos.entry_price and pos.stop_bps > 0 else None,
                    "target_price": (pos.entry_price * (1 + pos.target_bps / 1e4)
                                      if pos.side == "long"
                                      else pos.entry_price * (1 - pos.target_bps / 1e4))
                                     if pos.entry_price and pos.target_bps > 0 else None,
                    "entry_ts": pos.entry_ts.isoformat() if pos.entry_ts is not None else None,
                } if pos.qty != 0 else None,
            }
            HEARTBEAT_FILE.parent.mkdir(exist_ok=True)
            HEARTBEAT_FILE.write_text(json.dumps(payload))
        except Exception:
            pass  # never let heartbeat kill the loop

    def iterate(self, force: bool = False):
        now_utc = pd.Timestamp.now(tz="UTC")
        now_et = now_utc.tz_convert("America/New_York")
        self._reset_day_if_needed(now_et)

        # Kill switch: if the dashboard (or you) dropped a halt.flag, refuse new entries.
        # Existing positions still exit on their schedule / at close.
        halted = HALT_FLAG.exists()
        any_open = (self.state.position.qty != 0
                    or len(self.state.option_positions) > 0)
        if halted and not any_open:
            log.info("HALT flag present — refusing new entries")
            self._write_heartbeat(now_utc, in_window=False)
            return

        # Session window check.
        if not force and not self._within_trading_window(now_et):
            log.info("outside trading window (et=%s); idling", now_et.strftime("%Y-%m-%d %H:%M"))
            if now_et.hour == 15 and now_et.minute >= (60 - self.cfg["risk"]["flat_by_minutes_before_close"]):
                if self.state.position.qty != 0 or self.state.option_positions:
                    log.info("flattening before close")
                    self._flat_all()
            self._write_heartbeat(now_utc, in_window=False)
            return

        # Build frame + features.
        df = self._build_live_dataframe()
        if df.empty:
            log.warning("empty live frame, skipping")
            return
        feats = build_features(df, self.cfg)
        # If the artifact was trained with multi-timeframe / volume-profile
        # extensions (any feature starting with `mtf_` or `vp_`), compute
        # those at inference time too.
        wants_mtf = any(c.startswith("mtf_") for c in self.art.feature_cols)
        wants_vp = any(c.startswith("vp_") for c in self.art.feature_cols)
        if wants_mtf or wants_vp:
            try:
                from research.feature_extensions import add_extensions
                feats = add_extensions(feats, df, sym=self.symbol.lower(),
                                        add_volume=wants_vp, add_mtf=wants_mtf)
                ext_cols = [c for c in feats.columns if c.startswith(("mtf_", "vp_"))]
                if ext_cols:
                    feats[ext_cols] = feats[ext_cols].ffill().fillna(0.0)
            except Exception as e:
                log.warning("feature extension failed (%s); features may be incomplete", e)
        # Align to the training feature schema; fill missing columns with NaN so LightGBM
        # can still predict (native NaN split) rather than crashing with KeyError.
        missing = [c for c in self.art.feature_cols if c not in feats.columns]
        for c in missing:
            feats[c] = np.nan
        if missing:
            log.info("filled %d missing feature columns with NaN: %s", len(missing), missing[:5])
        feats = feats[self.art.feature_cols]
        last = feats.iloc[[-1]]
        # Require at least the SPY-based short-lookback features to be non-NaN.
        essential = [c for c in self.art.feature_cols
                     if c.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
        if last[essential].isna().any(axis=1).iloc[0]:
            log.info("last feature row missing essentials, skipping")
            return

        # Options exit check — premium-based, with theta-protection in last hour.
        # Multi-N: iterate ALL open option positions, exit each independently.
        if self.mode == "options" and self.state.option_positions:
            try:
                from live.options import get_premium_quote, check_options_exit
                flat_min_before = int(self.cfg["risk"]["flat_by_minutes_before_close"])
                eod_flat_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0) \
                              - pd.Timedelta(minutes=flat_min_before)
                mins_to_eod = max(0, int((eod_flat_et - now_et).total_seconds() / 60))
                # Iterate over a snapshot — _flat_options mutates the list
                for pos in list(self.state.option_positions):
                    if not pos.option_symbol or pos.qty == 0:
                        continue
                    bid, _ask = get_premium_quote(self.option_data, pos.option_symbol)
                    if bid <= 0:
                        continue
                    mins_held = max(0, int((now_utc - pos.entry_ts).total_seconds() / 60))
                    reason = check_options_exit(
                        side=pos.option_side,
                        entry_premium=pos.entry_premium,
                        current_premium=bid,
                        mins_held=mins_held,
                        mins_to_eod_flat=mins_to_eod,
                        stop_pct=pos.stop_pct,
                        target_pct=pos.target_pct,
                    )
                    if reason is not None:
                        log.info("options exit (%s) %s: premium %.2f→%.2f",
                                  reason, pos.option_symbol, pos.entry_premium, bid)
                        self._flat_options(reason=reason, pos=pos)
                # If no remaining positions, return so we re-evaluate next tick
                if not self.state.option_positions:
                    return
            except Exception as e:
                log.warning("options exit check failed: %s", e)

        # Per-bar barrier check: did the latest completed bar's intra-bar
        # range touch our vol-scaled stop or target? Tie-break favours stop.
        if self.state.position.qty != 0 and self.state.position.stop_bps > 0:
            sym_l = self.symbol.lower()
            try:
                last_high = float(df[f"{sym_l}_high"].iloc[-1])
                last_low = float(df[f"{sym_l}_low"].iloc[-1])
                barrier = should_exit(
                    side=self.state.position.side,
                    entry_price=self.state.position.entry_price,
                    last_high=last_high, last_low=last_low,
                    stop_bps=self.state.position.stop_bps,
                    target_bps=self.state.position.target_bps,
                )
                if barrier is not None:
                    log.info("barrier hit (%s); exiting", barrier)
                    self._flat_all(reason=barrier)
                    return
            except Exception as e:
                log.warning("barrier check failed: %s", e)

        # Horizon fallback (only fires for RECOVERED positions where we don't
        # have a stop/target — exit_due was set in __init__ from order history).
        # Live entries set exit_due=None and rely on barriers + EOD-flat instead.
        if self.state.position.qty != 0 and self.state.position.exit_due is not None:
            if now_utc >= self.state.position.exit_due:
                log.info("recovered-position horizon up; exiting")
                self._flat_all(reason="horizon")

        # DD kill.
        if self._check_dd_kill():
            return

        # Predict.
        p = self.art.booster.predict(last.values)[0]
        p_down, p_flat, p_up = float(p[0]), float(p[1]), float(p[2])
        up_t = float(self.art.thresholds["up"])
        dn_t = float(self.art.thresholds["down"])

        last_price = float(df[f"{self.symbol.lower()}_close"].iloc[-1])
        log.info("tick et=%s price=%.2f p_up=%.3f p_dn=%.3f thr_up=%.2f thr_dn=%.2f pos=%s",
                 now_et.strftime("%H:%M"), last_price, p_up, p_down, up_t, dn_t, self.state.position.qty)
        self._write_heartbeat(now_utc, last_price=last_price, p_up=p_up, p_dn=p_down, in_window=True)

        # Decision the journal will record (whether or not we trade).
        decision = "flat"
        if p_up >= up_t and self.state.position.qty == 0:
            decision = "long_signal"
        elif p_down >= dn_t and self.state.position.qty == 0:
            decision = "short_signal"
        elif self.state.position.qty > 0:
            decision = "in_long"
        elif self.state.position.qty < 0:
            decision = "in_short"

        equity_now = self._account_equity()
        self.journal.log_prediction(
            ts_utc=now_utc.to_pydatetime(),
            symbol=self.symbol, price=last_price,
            p_down=p_down, p_flat=p_flat, p_up=p_up,
            thr_up=up_t, thr_dn=dn_t,
            decision=decision, position_qty=int(self.state.position.qty),
            equity=equity_now, in_window=True,
        )

        # Respect the halt flag for new entries even after signal computation.
        if halted:
            return

        # Only act when flat. We're minute-horizon, one position at a time.
        if self.state.position.qty != 0:
            return

        feature_row = last  # for SHAP capture below

        def _enter(side: str, plan):
            qty = plan.qty
            o_side = OrderSide.BUY if side == "long" else OrderSide.SELL
            self._submit(o_side, qty)
            ot = self.journal.open_trade(
                symbol=self.symbol, side=side, qty=qty, entry_price=last_price,
                p_up=p_up, p_dn=p_down,
                thr_up=up_t, thr_dn=dn_t, equity=equity_now,
                stop_bps=plan.stop_bps, target_bps=plan.target_bps, rv_bps=plan.rv_bps,
            )
            # SHAP attribution at entry.
            if self._shap_explainer is not None:
                try:
                    sv = self._shap_explainer.shap_values(feature_row.values)
                    target_class = 2 if side == "long" else 0
                    if isinstance(sv, list):
                        sv_class = np.asarray(sv[target_class][0])
                    else:
                        sv_class = np.asarray(sv[0, :, target_class])
                    self.journal.log_shap(
                        trade_id=ot.trade_id,
                        feature_names=list(self.art.feature_cols),
                        feature_values=feature_row.values[0],
                        shap_values=sv_class,
                    )
                except Exception as e:
                    log.warning("SHAP capture failed for %s: %s", ot.trade_id, e)
            # Live-updating Discord message: post once, then a background thread
            # PATCHes it every 10s with the latest price + unrealized P&L.
            if self.discord.enabled:
                # Compute minutes-until-EOD-flat for the embed's "exit in" hint.
                flat_min_before = int(self.cfg["risk"]["flat_by_minutes_before_close"])
                eod_flat_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0) \
                              - pd.Timedelta(minutes=flat_min_before)
                mins_to_eod = max(0, int((eod_flat_et - now_et).total_seconds() / 60))
                msg_id = self.discord.open_trade_live(
                    side=side, symbol=self.symbol, qty=qty,
                    entry_price=last_price, p_up=p_up, p_dn=p_down,
                    horizon_min=mins_to_eod,
                    stop_bps=plan.stop_bps, target_bps=plan.target_bps,
                )
                ot.discord_message_id = msg_id
                if msg_id:
                    self._trade_updater = TradeUpdater(self, msg_id, ot, interval=10)
                    self._trade_updater.start()
                else:
                    # Couldn't get a message_id — fall back to the old two-message style
                    self.discord.trade_entry(side=side, symbol=self.symbol, qty=qty,
                                              price=last_price, p_up=p_up, p_dn=p_down)
            # exit_due: predicted hold time when v2 is active, else None (barriers+EOD)
            predicted_hold = getattr(plan, "_predicted_hold_min", None)
            exit_due = (now_utc + pd.Timedelta(minutes=predicted_hold)
                        if predicted_hold is not None else None)
            self.state.position = Position(
                qty=qty if side == "long" else -qty,
                side=side,
                entry_ts=now_utc,
                exit_due=exit_due,
                entry_price=last_price,
                stop_bps=plan.stop_bps,
                target_bps=plan.target_bps,
                open_trade=ot,
            )

        # Branch on strategy mode
        if self.mode == "options":
            # Options entry path — buy a 7DTE/0DTE/etc. call/put on signal.
            # Multi-N: respect max_concurrent_positions; reject if at capacity.
            opt_cfg = self.cfg.get("strategy", {}).get("options", {})
            max_concurrent = int(opt_cfg.get("max_concurrent_positions", 1))
            current_open = len(self.state.option_positions)
            if current_open >= max_concurrent:
                log.info("options: at capacity %d/%d, skipping new entry",
                          current_open, max_concurrent)
                return

            opt_side = None
            opt_p = None
            if p_up >= up_t:
                opt_side = "long"; opt_p = p_up
            elif p_down >= dn_t:
                opt_side = "short"; opt_p = p_down
            if opt_side is not None:
                opt_plan = self._plan_options_entry(last_price, opt_side, p=opt_p)
                if opt_plan is None:
                    log.warning("no options plan available; skipping")
                else:
                    log.info("options entry plan (%d/%d concurrent): %s",
                              current_open + 1, max_concurrent, opt_plan.note)
                    o_side = OrderSide.BUY  # always buying premium
                    self._submit_option(opt_plan.contract.occ_symbol,
                                          opt_plan.qty_contracts, o_side)
                    # Build a journal record (using the OpenTrade dataclass)
                    ot = self.journal.open_trade(
                        symbol=opt_plan.contract.occ_symbol,
                        side=opt_side, qty=opt_plan.qty_contracts,
                        entry_price=opt_plan.entry_premium,
                        p_up=p_up, p_dn=p_down, thr_up=up_t, thr_dn=dn_t,
                        equity=equity_now,
                    )
                    # New position appended to the multi-position list.
                    new_pos = Position(
                        qty=opt_plan.qty_contracts,
                        side=opt_side,
                        entry_ts=now_utc,
                        exit_due=None,
                        entry_price=opt_plan.entry_premium,
                        option_symbol=opt_plan.contract.occ_symbol,
                        option_side=opt_plan.contract.side,
                        entry_premium=opt_plan.entry_premium,
                        stop_pct=float(opt_cfg.get("stop_pct", 0.50)),
                        target_pct=float(opt_cfg.get("target_pct", 1.00)),
                        open_trade=ot,
                    )
                    self.state.option_positions.append(new_pos)
                    if self.discord.enabled:
                        try:
                            self.discord.embed(
                                title=(f"OPT {opt_side.upper()} "
                                       f"{opt_plan.contract.occ_symbol}"),
                                description=(f"qty={opt_plan.qty_contracts}  "
                                             f"premium=${opt_plan.entry_premium:.2f}  "
                                             f"max loss=${opt_plan.risk_dollars:.0f}  "
                                             f"({len(self.state.option_positions)}/{max_concurrent} concurrent)"),
                                fields=[
                                    {"name": "p", "value": f"{opt_p:.3f}", "inline": True},
                                    {"name": "Stop %", "value": f"-{new_pos.stop_pct*100:.0f}%", "inline": True},
                                    {"name": "Target %", "value": f"+{new_pos.target_pct*100:.0f}%", "inline": True},
                                ],
                            )
                        except Exception:
                            pass
            return  # don't run stocks-mode below

        # Stocks-mode entry path (legacy)
        if p_up >= up_t:
            plan = self._plan_exit(df, last_price, "long", p=p_up, feature_row=last)
            if plan is None:
                log.warning("plan_exit returned None for long signal; skipping")
            else:
                log.info("entry plan: %s", plan.note)
                if plan.qty > 0:
                    _enter("long", plan)
        elif p_down >= dn_t:
            plan = self._plan_exit(df, last_price, "short", p=p_down, feature_row=last)
            if plan is None:
                log.warning("plan_exit returned None for short signal; skipping")
            else:
                log.info("entry plan: %s", plan.note)
                if plan.qty > 0:
                    _enter("short", plan)

    def run_forever(self):
        # ── Watchdog: kill any single iteration that exceeds 5 minutes ────────
        # Prevents indefinite hangs on network calls (DNS, broker API stuck).
        # If an iteration goes silent for >5 min, SIGALRM fires → exception →
        # docker restart policy kicks in.
        import signal as _signal
        ITERATION_TIMEOUT_SECONDS = 300

        def _alarm_handler(signum, frame):
            raise TimeoutError(f"iteration exceeded {ITERATION_TIMEOUT_SECONDS}s — watchdog fired")

        # Set global socket timeout so any HTTP/socket call can't hang forever
        import socket as _socket
        _socket.setdefaulttimeout(60)  # 60s max per network call

        try:
            _signal.signal(_signal.SIGALRM, _alarm_handler)
        except (AttributeError, ValueError):
            # Windows or non-main thread doesn't support SIGALRM — soft-degrade
            pass

        poll = int(self.cfg["live"]["poll_seconds"])
        last_hb = 0.0
        hb_period = int(self.cfg["live"]["heartbeat_log_seconds"])
        log.info("live loop starting poll=%ds paper=%s watchdog=%ds",
                 poll, self.cfg["live"]["paper"], ITERATION_TIMEOUT_SECONDS)

        while not self._stop:
            t0 = time.time()
            try:
                # Arm watchdog before iteration; disarm after
                try:
                    _signal.alarm(ITERATION_TIMEOUT_SECONDS)
                except (AttributeError, OSError):
                    pass
                self.iterate()
                try:
                    _signal.alarm(0)  # disarm
                except (AttributeError, OSError):
                    pass
            except TimeoutError as e:
                log.error("WATCHDOG: %s — exiting so docker can restart cleanly", e)
                if self.discord.enabled:
                    self.discord.crash(f"Watchdog: iteration hung > {ITERATION_TIMEOUT_SECONDS}s. Container will restart.")
                # Exit with non-zero so docker `restart: unless-stopped` kicks in
                import sys
                sys.exit(2)
            except Exception as e:
                log.exception("iterate crashed: %s", e)
                if self.discord.enabled:
                    import traceback
                    self.discord.crash(traceback.format_exc())
            if time.time() - last_hb > hb_period:
                log.info("heartbeat ok")
                last_hb = time.time()
            # Sleep until the next minute boundary.
            elapsed = time.time() - t0
            time.sleep(max(1.0, poll - elapsed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one iteration and exit")
    ap.add_argument("--force", action="store_true",
                    help="bypass the trading-window check (for smoke testing outside market hours)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute signal but do not submit orders")
    ap.add_argument("--config", default="v1")
    args = ap.parse_args()
    t = LiveTrader(args.config)
    if args.dry_run:
        # Monkey-patch order submission to a no-op for dry runs.
        def _noop(side, qty, _orig=t._submit):
            log.info("DRY-RUN would submit %s %d %s", side.value, qty, t.symbol)
            return None
        t._submit = _noop
    if args.once:
        t.iterate(force=args.force)
    else:
        t.run_forever()


if __name__ == "__main__":
    main()
