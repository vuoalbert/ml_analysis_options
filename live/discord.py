"""Discord webhook sender. Silent no-op if DISCORD_WEBHOOK_URL is unset.

Embed colors:
    green = profitable trade / system OK
    red   = losing trade / failure / halt
    amber = warning (DD kill, regime alert)
    gray  = informational

Usage:
    python -m live.discord --test         # send a test message to verify wiring
    python -m live.discord --message "hi"  # send arbitrary text
"""
from __future__ import annotations

import argparse
import json
import time

import requests

from utils.env import discord_webhook
from utils.logging import get

log = get("live.discord")

GREEN = 0x26a69a
RED = 0xef5350
AMBER = 0xffa726
GRAY = 0x6b7280
BLUE = 0x42a5f5


class Discord:
    def __init__(self, url: str | None = None):
        self.url = url or discord_webhook()
        self.enabled = bool(self.url)
        if not self.enabled:
            log.info("DISCORD_WEBHOOK_URL not set — alerts disabled")

    def _post(self, payload: dict, retries: int = 2, return_id: bool = False):
        """POST a message. If return_id=True, append ?wait=true and return the message ID."""
        if not self.enabled:
            return None if return_id else False
        url = f"{self.url}?wait=true" if return_id else self.url
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=8)
                if r.status_code == 204 or 200 <= r.status_code < 300:
                    if return_id:
                        try:
                            return r.json().get("id")
                        except Exception:
                            return None
                    return True
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 1))
                    time.sleep(min(wait, 5))
                    continue
                log.warning("discord webhook returned %d: %s", r.status_code, r.text[:200])
                return None if return_id else False
            except Exception as e:
                log.warning("discord post failed (attempt %d): %s", attempt + 1, e)
                time.sleep(0.5 * (attempt + 1))
        return None if return_id else False

    def _patch(self, message_id: str, payload: dict, retries: int = 2) -> bool:
        """Edit a previously-posted webhook message."""
        if not self.enabled or not message_id:
            return False
        url = f"{self.url}/messages/{message_id}"
        for attempt in range(retries + 1):
            try:
                r = requests.patch(url, json=payload, timeout=8)
                if 200 <= r.status_code < 300:
                    return True
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", 1))
                    time.sleep(min(wait, 5))
                    continue
                log.warning("discord PATCH returned %d: %s", r.status_code, r.text[:200])
                return False
            except Exception as e:
                log.warning("discord PATCH failed (attempt %d): %s", attempt + 1, e)
                time.sleep(0.5 * (attempt + 1))
        return False

    # ---- formatted senders ----

    def embed(self, title: str, description: str = "",
              color: int = GRAY, fields: list[dict] | None = None) -> bool:
        emb = {"title": title, "color": color}
        if description:
            emb["description"] = description
        if fields:
            emb["fields"] = fields
        return self._post({"embeds": [emb]})

    def trade_entry(self, side: str, symbol: str, qty: int, price: float,
                    p_up: float, p_dn: float) -> bool:
        return self.embed(
            title=f"Entry  {side.upper()}  {symbol}",
            color=GREEN if side == "long" else RED,
            fields=[
                {"name": "Price", "value": f"${price:,.2f}", "inline": True},
                {"name": "Quantity", "value": f"{qty:,}", "inline": True},
                {"name": "Notional", "value": f"${qty * price:,.2f}", "inline": True},
                {"name": "P(up)", "value": f"{p_up:.3f}", "inline": True},
                {"name": "P(down)", "value": f"{p_dn:.3f}", "inline": True},
            ],
        )

    # ---- live-updating trade messages ----

    def open_trade_live(self, side: str, symbol: str, qty: int, entry_price: float,
                        p_up: float, p_dn: float, horizon_min: int,
                        stop_bps: float = 0.0, target_bps: float = 0.0) -> str | None:
        """Post the initial 'OPEN' message with ?wait=true, return the message_id."""
        embed = self._open_trade_embed(
            side=side, symbol=symbol, qty=qty,
            entry_price=entry_price, current_price=entry_price,
            p_up=p_up, p_dn=p_dn,
            mins_held=0, mins_remaining=horizon_min,
            stop_bps=stop_bps, target_bps=target_bps,
        )
        return self._post({"embeds": [embed]}, return_id=True)

    def update_open_trade(self, message_id: str, side: str, symbol: str, qty: int,
                          entry_price: float, current_price: float,
                          p_up: float, p_dn: float,
                          mins_held: int, mins_remaining: int,
                          stop_bps: float = 0.0, target_bps: float = 0.0) -> bool:
        """PATCH the open-trade message with current unrealized P&L."""
        embed = self._open_trade_embed(
            side=side, symbol=symbol, qty=qty,
            entry_price=entry_price, current_price=current_price,
            p_up=p_up, p_dn=p_dn,
            mins_held=mins_held, mins_remaining=mins_remaining,
            stop_bps=stop_bps, target_bps=target_bps,
        )
        return self._patch(message_id, {"embeds": [embed]})

    def finalize_trade(self, message_id: str, side: str, symbol: str, qty: int,
                       entry_price: float, exit_price: float,
                       pnl_dollars: float, pnl_bp: float, hold_minutes: int) -> bool:
        """PATCH the trade message into its final 'CLOSED' state."""
        win = pnl_dollars >= 0
        embed = {
            "title": f"Closed  {side.upper()}  {symbol}  ({'WIN' if win else 'LOSS'})",
            "color": GREEN if win else RED,
            "fields": [
                {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
                {"name": "Exit", "value": f"${exit_price:,.2f}", "inline": True},
                {"name": "Held", "value": f"{hold_minutes} min", "inline": True},
                {"name": "P&L", "value": f"${pnl_dollars:+,.2f}", "inline": True},
                {"name": "bp", "value": f"{pnl_bp:+.1f}", "inline": True},
                {"name": "Qty", "value": f"{qty:,}", "inline": True},
            ],
        }
        return self._patch(message_id, {"embeds": [embed]})

    def _open_trade_embed(self, *, side: str, symbol: str, qty: int,
                          entry_price: float, current_price: float,
                          p_up: float, p_dn: float,
                          mins_held: int, mins_remaining: int,
                          stop_bps: float = 0.0, target_bps: float = 0.0) -> dict:
        sign = 1 if side == "long" else -1
        pnl_per_share = sign * (current_price - entry_price)
        pnl_dollars = pnl_per_share * qty
        pnl_bp = (pnl_per_share / max(entry_price, 1e-9)) * 1e4
        # Color: green if winning, red if losing, amber if flat.
        if pnl_bp > 0.5:
            color = GREEN
        elif pnl_bp < -0.5:
            color = RED
        else:
            color = AMBER
        # Compute absolute stop/target prices for human-readable display.
        if side == "long":
            stop_px = entry_price * (1 - stop_bps / 1e4)
            target_px = entry_price * (1 + target_bps / 1e4)
        else:
            stop_px = entry_price * (1 + stop_bps / 1e4)
            target_px = entry_price * (1 - target_bps / 1e4)
        fields = [
            {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
            {"name": "Current", "value": f"${current_price:,.2f}", "inline": True},
            {"name": "Held", "value": f"{mins_held} min", "inline": True},
            {"name": "Unrealized", "value": f"${pnl_dollars:+,.2f}", "inline": True},
            {"name": "bp", "value": f"{pnl_bp:+.1f}", "inline": True},
            {"name": "Qty", "value": f"{qty:,}", "inline": True},
            {"name": "P(up) at entry", "value": f"{p_up:.3f}", "inline": True},
            {"name": "P(dn) at entry", "value": f"{p_dn:.3f}", "inline": True},
        ]
        if stop_bps > 0:
            fields.extend([
                {"name": "Stop", "value": f"${stop_px:,.2f} ({stop_bps:.0f}bp)", "inline": True},
                {"name": "Target", "value": f"${target_px:,.2f} ({target_bps:.0f}bp)", "inline": True},
            ])
        return {
            "title": f"Open  {side.upper()}  {symbol}  •  exit in {mins_remaining}m",
            "color": color,
            "fields": fields,
        }

    def trade_exit(self, side: str, symbol: str, qty: int,
                   entry_px: float, exit_px: float,
                   pnl_dollars: float, pnl_bp: float,
                   hold_minutes: int) -> bool:
        win = pnl_dollars >= 0
        return self.embed(
            title=f"Exit  {side.upper()}  {symbol}  ({'WIN' if win else 'LOSS'})",
            color=GREEN if win else RED,
            fields=[
                {"name": "Entry", "value": f"${entry_px:,.2f}", "inline": True},
                {"name": "Exit", "value": f"${exit_px:,.2f}", "inline": True},
                {"name": "Held", "value": f"{hold_minutes} min", "inline": True},
                {"name": "P&L", "value": f"${pnl_dollars:+,.2f}", "inline": True},
                {"name": "bp", "value": f"{pnl_bp:+.1f}", "inline": True},
                {"name": "Qty", "value": f"{qty:,}", "inline": True},
            ],
        )

    def halt(self, on: bool, reason: str = "") -> bool:
        if on:
            return self.embed(
                title="Trading HALTED",
                description=reason or "Manual halt — no new entries until cleared.",
                color=RED,
            )
        return self.embed(
            title="Trading RESUMED",
            description="Halt cleared. New entries allowed.",
            color=GREEN,
        )

    def dd_kill(self, dd_pct: float, equity: float) -> bool:
        return self.embed(
            title="Daily drawdown kill-switch triggered",
            description="No new entries for the rest of the session.",
            color=AMBER,
            fields=[
                {"name": "Drawdown", "value": f"{dd_pct*100:+.2f}%", "inline": True},
                {"name": "Equity", "value": f"${equity:,.2f}", "inline": True},
            ],
        )

    def session_summary(self, date: str, trades: int, wins: int,
                         pnl_dollars: float, pnl_bp: float,
                         starting_equity: float, ending_equity: float) -> bool:
        hit = (wins / trades * 100) if trades else 0
        win = pnl_dollars >= 0
        return self.embed(
            title=f"Session recap  {date}",
            color=GREEN if win else RED if pnl_dollars < 0 else GRAY,
            fields=[
                {"name": "Trades", "value": str(trades), "inline": True},
                {"name": "Hit rate", "value": f"{hit:.1f}%" if trades else "—", "inline": True},
                {"name": "Net P&L", "value": f"${pnl_dollars:+,.2f}", "inline": True},
                {"name": "Net bp", "value": f"{pnl_bp:+.1f}", "inline": True},
                {"name": "Equity", "value": f"${starting_equity:,.2f} → ${ending_equity:,.2f}", "inline": True},
            ],
        )

    def startup(self, symbol: str, equity: float, paper: bool) -> bool:
        return self.embed(
            title=f"Live loop online  ({'paper' if paper else 'LIVE'})",
            color=BLUE,
            fields=[
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Equity", "value": f"${equity:,.2f}", "inline": True},
            ],
        )

    def crash(self, exc: str) -> bool:
        return self.embed(
            title="Live loop crashed",
            description=f"```\n{exc[:1500]}\n```",
            color=RED,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--message", help="send a custom plaintext message")
    args = ap.parse_args()
    d = Discord()
    if not d.enabled:
        print("DISCORD_WEBHOOK_URL not set — add it to .env")
        return
    if args.test:
        ok = d.embed(
            title="Test message",
            description="If you see this, the webhook is wired correctly.",
            color=BLUE,
            fields=[
                {"name": "From", "value": "ml_analysis live loop", "inline": True},
                {"name": "Status", "value": "online", "inline": True},
            ],
        )
        print("sent:" if ok else "failed:", "test message")
    elif args.message:
        ok = d._post({"content": args.message})
        print("sent:" if ok else "failed:", args.message)
    else:
        print("nothing to do; pass --test or --message")


if __name__ == "__main__":
    main()
