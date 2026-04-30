"""Model artifact: saves model + feature spec + thresholds + config snapshot.

The live loop loads this to guarantee identical features and thresholds as training.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import lightgbm as lgb

from utils.env import root


@dataclass
class Artifact:
    booster: lgb.Booster
    feature_cols: list[str]
    thresholds: dict   # {"up": float, "down": float}
    cfg: dict
    train_window: tuple[str, str]
    metrics: dict

    def to_dict(self) -> dict:
        return {
            "feature_cols": self.feature_cols,
            "thresholds": self.thresholds,
            "cfg": self.cfg,
            "train_window": list(self.train_window),
            "metrics": self.metrics,
        }


def artifact_dir() -> Path:
    d = root() / "artifacts"
    d.mkdir(exist_ok=True)
    return d


def save(art: Artifact, name: str = "latest") -> Path:
    d = artifact_dir() / name
    d.mkdir(exist_ok=True)
    art.booster.save_model(str(d / "model.lgb"))
    with open(d / "meta.json", "w") as f:
        json.dump(art.to_dict(), f, indent=2, default=str)
    return d


def load(name: str = "latest") -> Artifact:
    d = artifact_dir() / name
    booster = lgb.Booster(model_file=str(d / "model.lgb"))
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return Artifact(
        booster=booster,
        feature_cols=meta["feature_cols"],
        thresholds=meta["thresholds"],
        cfg=meta["cfg"],
        train_window=tuple(meta["train_window"]),
        metrics=meta["metrics"],
    )
