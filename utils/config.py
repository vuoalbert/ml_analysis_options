from pathlib import Path
import yaml
from .env import root


def load(name: str = "v1") -> dict:
    path = root() / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
