import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def alpaca_keys() -> tuple[str, str]:
    k = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    s = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not k or not s:
        raise RuntimeError("Alpaca keys missing. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env")
    return k, s


def fred_key() -> str | None:
    return os.environ.get("FRED_API_KEY")


def gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY")


def discord_webhook() -> str | None:
    """Discord webhook URL for trade alerts. Optional — silent if unset."""
    return os.environ.get("DISCORD_WEBHOOK_URL")


def root() -> Path:
    return ROOT
