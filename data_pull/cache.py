from pathlib import Path
import pandas as pd
from utils.env import root


def cache_dir() -> Path:
    d = root() / "cache"
    d.mkdir(exist_ok=True)
    return d


def save(df: pd.DataFrame, name: str) -> Path:
    p = cache_dir() / f"{name}.parquet"
    df.to_parquet(p)
    return p


def load(name: str) -> pd.DataFrame | None:
    p = cache_dir() / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def exists(name: str) -> bool:
    return (cache_dir() / f"{name}.parquet").exists()
