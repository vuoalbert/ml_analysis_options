"""Cross-sectional assembler: pulls multiple target symbols, applies the same
feature pipeline to each, tags rows with `ticker`, and concatenates.

Existing single-symbol code is reused unchanged: `assemble()` pulls a target
symbol's bars + the same cross-asset/macro features. We rename the target's
columns to `spy_*` so the existing `features.build` (which is hardcoded to
that prefix) works without modification — the prefix is just a name, the
math doesn't care which underlying symbol it represents.
"""
from __future__ import annotations

import pandas as pd

from utils.config import load as load_cfg
from utils.logging import get
from .assemble import assemble

log = get("data.assemble_multi")


def assemble_for_target(cfg: dict, target_symbol: str) -> pd.DataFrame:
    """Assemble a target symbol with cross-asset features, then rename target columns
    to `spy_*` so the SPY-prefixed feature builder works unmodified."""
    sub_cfg = dict(cfg)
    sub_cfg["universe"] = {**cfg["universe"], "symbol": target_symbol}
    df = assemble(sub_cfg)
    target_lower = target_symbol.lower()
    if target_lower != "spy":
        rename_map = {c: c.replace(f"{target_lower}_", "spy_")
                      for c in df.columns if c.startswith(f"{target_lower}_")}
        df = df.rename(columns=rename_map)
    return df


def assemble_multi(cfg: dict, targets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return per-ticker assembled frames keyed by symbol.

    Returning a dict (not a concat) so feature/label computation can run
    independently per ticker — many features (VWAP, opening range, overnight gap)
    are session-grouped and would corrupt across tickers if mixed first.
    """
    targets = targets or cfg["universe"].get("targets", [cfg["universe"]["symbol"]])
    out = {}
    for sym in targets:
        log.info("assembling %s …", sym)
        out[sym] = assemble_for_target(cfg, sym)
    return out
