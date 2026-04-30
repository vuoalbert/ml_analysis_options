"""End-to-end-style joint exit prediction for the v2 research model.

Trains 3 LightGBM regressors, all sharing the v2 entry model's 61-feature
input vector, each predicting one exit decision per trade:

  • v2_stop    — ideal stop distance in bps (label: max adverse excursion + 5 bps margin)
  • v2_target  — ideal target distance in bps (label: max favorable excursion to EOD)
  • v2_hold    — ideal hold time in minutes (label: bars from entry to peak favorable price)

Inference path during a backtest:
    direction   = v2_entry model says long/short
    stop_bps    = clip(v2_stop.predict(features), 5, 100)
    target_bps  = clip(v2_target.predict(features), stop_bps × 1.0, 200)
    hold_min    = clip(v2_hold.predict(features), 5, 390)

Deviation from approaches_v2.py: this version trains on entries the v2 entry
model would have actually fired, not synthetic entries. Labels are clipped to
prevent extreme bets.

Save to artifacts/research_v2_exit/ as a 3-model bundle.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import lightgbm as lgb

from utils.config import load as load_cfg
from utils.env import root
from utils.logging import get
from features.build import build as build_features
from model.artifact import load as load_artifact
from research.backtest_original import build_frame
from research.vol_scaled_exits import realized_vol_bps

log = get("research.v2_predictors")
ARTIFACT_DIR_DEFAULT = root() / "artifacts" / "research_v2_exit"


@dataclass
class V2ExitBundle:
    stop_model: object
    target_model: object
    hold_model: object
    feature_cols: list[str]
    train_window: tuple[str, str]
    metrics: dict


# ---------- training data construction ----------

def harvest_v2_training_data(
    *,
    train_start: str,
    train_end: str,
    entry_artifact: str = "research_v2_entry",
    flat_minutes_before_close: int = 5,
    cfg_name: str = "v1",
):
    """For every entry the v2_entry model fires in [train_start, train_end), compute:
        - features at entry (61-feature vector aligned to v2_entry.feature_cols)
        - stop_label   = clip(max_adverse_excursion + 5, 5, 100)        [bps]
        - target_label = clip(max_favorable_excursion, 5, 200)           [bps]
        - hold_label   = clip(bar_idx_of_peak_favorable, 5, 390)         [minutes]

    Returns dict with X, y_stop, y_target, y_hold, feature_cols.

    No-horizon, same-day cap (matches live exit policy)."""
    c = load_cfg(cfg_name)
    art = load_artifact(entry_artifact)
    sym = c["universe"]["symbol"].lower()
    thr_up = float(art.thresholds["up"])
    thr_dn = float(art.thresholds["down"])

    log.info("building frame %s..%s", train_start, train_end)
    out = build_frame(train_start, train_end, c)
    if out.empty:
        return None

    feats = build_features(out, c)
    for col in art.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[art.feature_cols]
    essential = [col for col in art.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    feats = feats[feats[essential].notna().all(axis=1)]
    if feats.empty:
        return None

    proba = art.booster.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    pred["close"] = out.loc[pred.index, f"{sym}_close"]

    rth_mask = pred.index.to_series().apply(
        lambda t: 13 * 60 + 30 + c["risk"]["skip_first_minutes"]
                  <= t.hour * 60 + t.minute
                  < 20 * 60 - c["risk"]["skip_last_minutes"]
    )
    pred = pred[rth_mask.values]

    bars = out[[f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close"]].rename(columns={
        f"{sym}_open": "open", f"{sym}_high": "high",
        f"{sym}_low": "low", f"{sym}_close": "close",
    }).dropna()
    bars_et = bars.index.tz_convert("America/New_York").date

    rows = []
    for ts, row in pred.iterrows():
        side = None
        if row["p_up"] >= thr_up:
            side = "long"
        elif row["p_down"] >= thr_dn:
            side = "short"
        if side is None:
            continue
        if ts not in bars.index:
            continue
        ei = bars.index.get_loc(ts)
        if ei < 31:
            continue
        entry_price = float(bars.iloc[ei]["close"])

        # Same-day window
        entry_date = bars_et[ei]
        same_day_mask = bars_et == entry_date
        same_day_idx = np.where(same_day_mask)[0]
        last_same_day = int(same_day_idx[-1])
        end_idx = max(ei + 1, last_same_day - flat_minutes_before_close + 1)
        window = bars.iloc[ei + 1: end_idx + 1]
        if window.empty:
            continue

        if side == "long":
            adv_dollars = max(0.0, entry_price - float(window["low"].min()))
            fav_dollars = max(0.0, float(window["high"].max()) - entry_price)
            peak_idx = int(window["high"].argmax()) + 1
        else:
            adv_dollars = max(0.0, float(window["high"].max()) - entry_price)
            fav_dollars = max(0.0, entry_price - float(window["low"].min()))
            peak_idx = int(window["low"].argmin()) + 1

        adv_bps = (adv_dollars / entry_price) * 1e4
        fav_bps = (fav_dollars / entry_price) * 1e4

        stop_label = float(np.clip(adv_bps + 5.0, 5.0, 100.0))
        target_label = float(np.clip(fav_bps, 5.0, 200.0))
        hold_label = float(np.clip(peak_idx, 5, 390))

        feat_row = feats.loc[ts].values
        rows.append({
            "ts": ts,
            "side": side,
            "stop_label": stop_label,
            "target_label": target_label,
            "hold_label": hold_label,
            **{f"f_{i}": feat_row[i] for i in range(len(feat_row))},
        })

    log.info("v2 training rows: %d", len(rows))
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("ts")
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    return {
        "X": df[feature_cols].values,
        "y_stop": df["stop_label"].values,
        "y_target": df["target_label"].values,
        "y_hold": df["hold_label"].values,
        "feature_cols": list(art.feature_cols),
        "n_rows": len(rows),
    }


# ---------- training ----------

def train_one_lgb(X, y, X_val, y_val, params_extra: dict | None = None,
                    tight: bool = False) -> object:
    if tight:
        # Aggressive regularization to combat overfitting:
        # half the leaves, 5× the min-data-in-leaf, 5× the L2 penalty.
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.02,
            num_leaves=7,
            min_data_in_leaf=100,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            lambda_l2=5.0,
            verbosity=-1,
        )
    else:
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=15,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.85,
            bagging_freq=5,
            lambda_l2=1.0,
            verbosity=-1,
        )
    if params_extra:
        params.update(params_extra)
    booster = lgb.train(
        params,
        lgb.Dataset(X, label=y),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(X, label=y), lgb.Dataset(X_val, label=y_val)],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
    )
    return booster


def train(train_start: str, train_end: str,
           valid_start: str, valid_end: str,
           entry_artifact: str = "research_v2_entry",
           tight: bool = False,
           out_dir: Path = None) -> V2ExitBundle:
    log.info("=" * 60)
    log.info(" Training V2 exit-predictor bundle (stop, target, hold)")
    log.info(f"  entry model: {entry_artifact}")
    log.info(f"  train: {train_start} → {train_end}")
    log.info(f"  valid: {valid_start} → {valid_end}")
    log.info("=" * 60)

    tr = harvest_v2_training_data(
        train_start=train_start, train_end=train_end, entry_artifact=entry_artifact)
    va = harvest_v2_training_data(
        train_start=valid_start, train_end=valid_end, entry_artifact=entry_artifact)
    if tr is None or va is None:
        raise RuntimeError("not enough rows to train v2 exit bundle")

    metrics = {"n_train": tr["n_rows"], "n_valid": va["n_rows"]}
    log.info(f"train rows: {tr['n_rows']}  valid rows: {va['n_rows']}")

    metrics["tight"] = tight
    log.info("training stop predictor… tight=%s", tight)
    stop_model = train_one_lgb(tr["X"], tr["y_stop"], va["X"], va["y_stop"], tight=tight)
    pred_tr = stop_model.predict(tr["X"])
    pred_va = stop_model.predict(va["X"])
    rmse_tr = float(np.sqrt(np.mean((pred_tr - tr["y_stop"]) ** 2)))
    rmse_va = float(np.sqrt(np.mean((pred_va - va["y_stop"]) ** 2)))
    metrics["stop_rmse_train"] = rmse_tr
    metrics["stop_rmse"] = rmse_va
    metrics["stop_pred_range"] = [float(pred_va.min()), float(pred_va.max())]
    log.info(f"  stop   RMSE train={rmse_tr:.2f}  valid={rmse_va:.2f}  gap={rmse_va/rmse_tr:.2f}x")

    log.info("training target predictor…")
    target_model = train_one_lgb(tr["X"], tr["y_target"], va["X"], va["y_target"], tight=tight)
    pred_tr = target_model.predict(tr["X"])
    pred_va = target_model.predict(va["X"])
    rmse_tr = float(np.sqrt(np.mean((pred_tr - tr["y_target"]) ** 2)))
    rmse_va = float(np.sqrt(np.mean((pred_va - va["y_target"]) ** 2)))
    metrics["target_rmse_train"] = rmse_tr
    metrics["target_rmse"] = rmse_va
    metrics["target_pred_range"] = [float(pred_va.min()), float(pred_va.max())]
    log.info(f"  target RMSE train={rmse_tr:.2f}  valid={rmse_va:.2f}  gap={rmse_va/rmse_tr:.2f}x")

    log.info("training hold predictor…")
    hold_model = train_one_lgb(tr["X"], tr["y_hold"], va["X"], va["y_hold"], tight=tight)
    pred_tr = hold_model.predict(tr["X"])
    pred_va = hold_model.predict(va["X"])
    rmse_tr = float(np.sqrt(np.mean((pred_tr - tr["y_hold"]) ** 2)))
    rmse_va = float(np.sqrt(np.mean((pred_va - va["y_hold"]) ** 2)))
    metrics["hold_rmse_train"] = rmse_tr
    metrics["hold_rmse"] = rmse_va
    metrics["hold_pred_range"] = [float(pred_va.min()), float(pred_va.max())]
    log.info(f"  hold   RMSE train={rmse_tr:.1f}  valid={rmse_va:.1f}  gap={rmse_va/rmse_tr:.2f}x")

    bundle = V2ExitBundle(
        stop_model=stop_model, target_model=target_model, hold_model=hold_model,
        feature_cols=tr["feature_cols"],
        train_window=(train_start, train_end),
        metrics=metrics,
    )
    save(bundle, out_dir=out_dir)
    return bundle


# ---------- save / load ----------

def save(b: V2ExitBundle, out_dir: Path = None):
    out_dir = out_dir or ARTIFACT_DIR_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)
    b.stop_model.save_model(str(out_dir / "stop.lgb"))
    b.target_model.save_model(str(out_dir / "target.lgb"))
    b.hold_model.save_model(str(out_dir / "hold.lgb"))
    meta = {
        "feature_cols": b.feature_cols,
        "train_window": list(b.train_window),
        "metrics": b.metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("saved → %s", out_dir)


def load(art_dir: Path = None) -> V2ExitBundle:
    art_dir = art_dir or ARTIFACT_DIR_DEFAULT
    meta = json.loads((art_dir / "meta.json").read_text())
    return V2ExitBundle(
        stop_model=lgb.Booster(model_file=str(art_dir / "stop.lgb")),
        target_model=lgb.Booster(model_file=str(art_dir / "target.lgb")),
        hold_model=lgb.Booster(model_file=str(art_dir / "hold.lgb")),
        feature_cols=meta["feature_cols"],
        train_window=tuple(meta["train_window"]),
        metrics=meta["metrics"],
    )


def main():
    train(
        train_start="2024-04-29", train_end="2026-02-01",   # 21 months
        valid_start="2026-02-01", valid_end="2026-03-01",   # 1 month valid
    )


if __name__ == "__main__":
    main()
