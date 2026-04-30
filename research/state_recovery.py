"""Boot-time state recovery: read live positions from Alpaca and reconstruct
the in-memory `Position` record so a restarted loop doesn't forget what it
already holds.

The bug this fixes:
    On 2026-04-26 the GCP container restarted while holding 5 SPY shares.
    LiveTrader.__init__ initialises `self.state = State()` — i.e. position=0
    in memory — even though the broker still has the open position. The loop
    then never schedules an exit, the position sits indefinitely, and the
    horizon-based exit logic is silently bypassed.

The fix:
    On boot, ask Alpaca whether we already hold the symbol. If yes, reconstruct
    a `Position` object using `avg_entry_price` from the position record and
    the most recent filled entry order's timestamp from order history. Schedule
    `exit_due = entry_ts + horizon_min`. If that's already past, the loop's
    existing horizon check will exit on the next iteration.

This file is a prototype — to deploy, the function below is called from
`LiveTrader.__init__` after `self.state = State()`. Not modifying live/loop.py
during the 30-day lock window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide
from alpaca.common.exceptions import APIError


@dataclass
class RecoveredPosition:
    """Same shape as live.loop.Position, but as a plain dataclass for the
    prototype. To deploy, copy these fields onto LiveTrader's existing
    Position dataclass."""
    qty: int                                # signed
    side: str                               # "long" or "short"
    entry_ts: Optional[pd.Timestamp]        # UTC, from most recent fill
    exit_due: Optional[pd.Timestamp]        # entry_ts + horizon
    entry_price: float                      # broker's avg_entry_price
    note: str                               # human-readable summary


def _most_recent_entry_order(
    trading: TradingClient, symbol: str, side: str, qty_held: int,
) -> Optional[datetime]:
    """Walk the closed-orders history backwards to find the fill that opened
    the current position.

    "Entry" = a filled BUY for a long, filled SELL for a short. We just take
    the most recent filled order on the opening side; if the position was
    built up by multiple fills, this is conservative (latest fill = newest
    entry timestamp = furthest-out exit_due).
    """
    open_side = OrderSide.BUY if side == "long" else OrderSide.SELL
    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol],
            limit=50,
            direction="desc",
        )
        orders = trading.get_orders(filter=req)
    except APIError:
        return None
    for o in orders:
        if o.side == open_side and o.filled_at is not None and float(o.filled_qty or 0) > 0:
            return o.filled_at
    return None


def recover_position(
    trading: TradingClient,
    symbol: str,
    horizon_min: int,
    now_utc: Optional[pd.Timestamp] = None,
) -> Optional[RecoveredPosition]:
    """Return a RecoveredPosition if the broker reports an open position in
    `symbol`, else None.

    Failure modes (all return None — caller treats as "flat"):
      - No open position
      - Alpaca API error
      - Position quantity == 0 somehow
    """
    try:
        p = trading.get_open_position(symbol)
    except APIError:
        return None
    except Exception:
        return None

    qty_unsigned = int(float(p.qty))
    if qty_unsigned == 0:
        return None
    side_str = str(p.side).lower()
    side = "long" if "long" in side_str else "short"
    qty_signed = qty_unsigned if side == "long" else -qty_unsigned
    avg_entry = float(p.avg_entry_price)

    entry_dt = _most_recent_entry_order(trading, symbol, side, qty_unsigned)
    if entry_dt is not None:
        entry_ts = pd.Timestamp(entry_dt).tz_convert("UTC")
        exit_due = entry_ts + pd.Timedelta(minutes=horizon_min)
        note = (f"recovered {side} {qty_unsigned} {symbol} @ {avg_entry:.2f} — "
                f"entry {entry_ts.strftime('%Y-%m-%d %H:%M UTC')}, "
                f"exit_due {exit_due.strftime('%H:%M UTC')}")
    else:
        # No fill record found — schedule immediate exit on next iterate.
        # Better to flatten an unknown-age position than to hold it forever.
        now = now_utc or pd.Timestamp.now(tz="UTC")
        entry_ts = None
        exit_due = now - pd.Timedelta(seconds=1)
        note = (f"recovered {side} {qty_unsigned} {symbol} @ {avg_entry:.2f} — "
                f"no fill record found, scheduling immediate exit")

    return RecoveredPosition(
        qty=qty_signed, side=side, entry_ts=entry_ts,
        exit_due=exit_due, entry_price=avg_entry, note=note,
    )


def integration_patch_for_loop_py() -> str:
    """Returns the diff hint for LiveTrader.__init__. Print this from the
    runner so we have a copy-pasteable snippet for after the lock window."""
    return """
    # ---- in LiveTrader.__init__, AFTER self.state = State() ----
    from research.state_recovery import recover_position
    rec = recover_position(self.trading, self.symbol, self.horizon)
    if rec is not None:
        log.warning("STATE RECOVERY: %s", rec.note)
        self.state.position = Position(
            qty=rec.qty, side=rec.side,
            entry_ts=rec.entry_ts, exit_due=rec.exit_due,
            entry_price=rec.entry_price,
            open_trade=None,   # no journal record — recovered, not opened by us
        )
        if self.discord.enabled:
            try:
                self.discord.startup_recovered(rec.note)
            except Exception:
                pass
    # ------------------------------------------------------------
"""
