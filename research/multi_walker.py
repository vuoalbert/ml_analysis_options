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
               max_concurrent: int = 1) -> list[dict]:
    """Walk signals in order; allow up to `max_concurrent` open positions.

    Each call to simulator returns a single trade dict (with exit_ts).
    A signal is skipped if `max_concurrent` open positions already cover
    its entry timestamp.
    """
    trades = []
    open_exits: list[pd.Timestamp] = []   # exit_ts of currently-open trades
    for s in signals:
        ts = pd.Timestamp(s["ts"])
        # Drop trades that have already exited by this signal's time
        open_exits = [e for e in open_exits if e > ts]
        if len(open_exits) >= max_concurrent:
            continue
        trade = simulator(bars, s)
        if trade is None:
            continue
        trades.append(trade)
        open_exits.append(pd.Timestamp(trade["exit_ts"]))
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
