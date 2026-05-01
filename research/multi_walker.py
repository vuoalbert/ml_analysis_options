"""Multi-position walker — like backtest_compare.walk_with_blocking but
allows up to N concurrent options positions, since options qty is just
"number of contracts" with no margin restriction at small size.

Each signal still gets a single sim() call. The walker maintains a list
of currently-open trades (those whose exit_ts is in the future relative
to the new signal). If max_concurrent is reached, the new signal is skipped.

Two cap modes:
  • walk_multi          — cap by number of concurrent positions
  • walk_capital_capped — cap by % of equity tied up in open premium
                          (more realistic for long-premium retail accounts)
"""
from __future__ import annotations

import pandas as pd


def walk_multi(signals: list, bars: pd.DataFrame, simulator,
               max_concurrent: int = 1, max_per_symbol: int = 0,
               cancel_on_flip: bool = False) -> list[dict]:
    """Walk signals in order; allow up to `max_concurrent` open positions.

    Each call to simulator returns a single trade dict (with exit_ts).
    A signal is skipped if `max_concurrent` open positions already cover
    its entry timestamp.

    Optional features:
      • max_per_symbol: cap number of open positions on the same OCC symbol
        (prevents stacking 15 entries on a single strike). 0 = no cap.
      • cancel_on_flip: when a new signal's side opposes the side of any
        currently-open position, force-exit ALL opposite-side positions before
        attempting the entry. Reduces "hedged" basket exposure that the basic
        multi-walker accumulates.
    """
    trades = []
    # Track open positions as list of dicts so we can inspect symbol/side
    open_positions: list[dict] = []
    for s in signals:
        ts = pd.Timestamp(s["ts"])
        # Drop trades that have already exited by this signal's time
        open_positions = [p for p in open_positions if p["exit_ts"] > ts]

        # Cancel-on-flip: close opposite-side positions before processing entry
        if cancel_on_flip and open_positions:
            new_side = s.get("side")           # "long" or "short"
            opposite = "short" if new_side == "long" else "long"
            for p in list(open_positions):
                if p.get("side") == opposite:
                    # Build a synthetic exit trade record (premium frozen at the moment)
                    # Mark the original trade's exit at this signal's timestamp.
                    p["forced_exit_ts"] = ts
                    p["forced_exit_reason"] = "cancel_on_flip"
                    open_positions.remove(p)

        if len(open_positions) >= max_concurrent:
            continue

        # Per-symbol cap check (estimated — we don't know the new trade's symbol
        # until after sim runs, so this is a soft check via simulation).
        if max_per_symbol > 0:
            # Pre-walk to inspect what symbol this entry would pick
            trial = simulator(bars, s)
            if trial is None:
                continue
            sym = trial.get("option_symbol") or trial.get("symbol")
            sym_count = sum(1 for p in open_positions if p.get("option_symbol") == sym)
            if sym_count >= max_per_symbol:
                continue
            trade = trial
        else:
            trade = simulator(bars, s)
            if trade is None:
                continue

        trades.append(trade)
        open_positions.append({
            "exit_ts": pd.Timestamp(trade["exit_ts"]),
            "side": trade.get("side"),
            "option_symbol": trade.get("option_symbol") or trade.get("symbol"),
        })
    return trades


def walk_capital_capped(signals: list, bars: pd.DataFrame, simulator,
                         equity: float = 30_000.0,
                         max_capital_pct: float = 0.50) -> list[dict]:
    """Walk signals; skip if next trade would push deployed premium over cap.

    Tracks (exit_ts, premium_outlay) of every open trade. A new signal at
    time T is rejected when sum(premium_outlay for open trades at T) +
    new_trade_outlay > equity × max_capital_pct.

    This more accurately models a long-premium retail account where each
    option contract requires upfront cash and you can't exceed the account
    balance with simultaneous positions.
    """
    cap_dollars = equity * max_capital_pct
    trades = []
    open_pos = []   # list of (exit_ts, outlay_$)
    for s in signals:
        ts = pd.Timestamp(s["ts"])
        # Drop trades that have already exited
        open_pos = [(e, o) for (e, o) in open_pos if e > ts]
        deployed = sum(o for (_, o) in open_pos)

        trade = simulator(bars, s)
        if trade is None:
            continue
        new_outlay = trade["entry_premium"] * 100.0 * trade["qty"]
        if deployed + new_outlay > cap_dollars:
            continue   # would exceed cap, skip
        trades.append(trade)
        open_pos.append((pd.Timestamp(trade["exit_ts"]), new_outlay))
    return trades
