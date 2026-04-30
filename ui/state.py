"""Shared state between dashboard and live loop: halt flag + heartbeat reader."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from utils.env import root

HALT_FLAG = root() / "halt.flag"
HEARTBEAT_FILE = root() / "logs" / "heartbeat.json"
LIVE_LOG = root() / "logs" / "live_loop.log"


def halted() -> bool:
    return HALT_FLAG.exists()


def set_halt(on: bool):
    if on:
        HALT_FLAG.write_text(datetime.now(timezone.utc).isoformat())
    else:
        HALT_FLAG.unlink(missing_ok=True)


def read_heartbeat() -> dict | None:
    if not HEARTBEAT_FILE.exists():
        return None
    try:
        data = json.loads(HEARTBEAT_FILE.read_text())
        return data
    except Exception:
        return None


def heartbeat_age_seconds() -> float | None:
    if not HEARTBEAT_FILE.exists():
        return None
    mtime = HEARTBEAT_FILE.stat().st_mtime
    return (datetime.now(timezone.utc).timestamp() - mtime)


def log_mtime_seconds() -> float | None:
    if not LIVE_LOG.exists():
        return None
    return datetime.now(timezone.utc).timestamp() - LIVE_LOG.stat().st_mtime
