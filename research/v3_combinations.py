"""V3 combination experiment — train and backtest every combination of:

  • Multi-model ensemble (5 LightGBM models with different seeds, averaged)
  • Volume profile features (5 new columns)
  • Multi-timeframe features (5 new columns)

8 variants total (2^3 combinations including baseline). Each tested on
5 windows; compared against the current live model.

Workflow:
  1. Build training dataset 4 times (one per feature set: base / +vol / +mtf / +both)
  2. For each feature set, train 5 LightGBM models with different seeds
  3. At inference: single model OR ensemble of 5
  4. Combined with v2 exit predictors (unchanged)
  5. Backtest on 5 windows × 8 variants = 40 runs

Run:
    python -m research.v3_combinations  --train          # train all variants
    python -m research.v3_combinations  --backtest       # run backtests
    python -m research.v3_combinations  --all            # both
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import lightgbm as lgb

from utils.config import load as load_cfg
from utils.env import root
from utils.logging import get
from features.build import build as build_features, feature_columns
from labels.build import build as build_labels
from data_pull.assemble import assemble
from model.artifact import Artifact, save as save_artifact, load as load_artifact
from model.train import (
    DEFAULT_PARAMS, train_lgb, predict_proba, tune_thresholds,
    simulate_one_position, prepare_dataset,
)
from backtest.costs import round_trip_cost_bp
from research.feature_extensions import add_extensions
from research.backtest_compare import walk_with_blocking
from research.run_v2_combined import (
    get_signals_features_bars, make_variant_C, metrics, print_table,
)
from research.v2_predictors import load as load_v2_exits

log = get("research.v3")
ARTIFACT_BASE = root() / "artifacts"


WINDOWS = [
    ("THIS MONTH (Apr 2026)",                   "2026-04-01", "2026-04-29"),
    ("LAST MONTH (Mar 2026, holdout)",          "2026-03-01", "2026-04-01"),
    ("LAST 2 MONTHS (Mar+Apr 2026)",            "2026-03-01", "2026-04-29"),
    ("LAST 2 YEARS",                            "2024-04-29", "2026-04-29"),
    ("PRE-TRAIN OOS (Sep 2020 - Apr 2023)",     "2020-09-01", "2023-04-01"),
]


# All 8 combinations: (ensemble, vol, mtf)
COMBINATIONS = [
    # tag      ens   vol    mtf
    ("base",        False, False, False),  # = current live (single model, no extras)
    ("ens",         True,  False, False),
    ("vol",         False, True,  False),
    ("mtf",         False, False, True),
    ("ens_vol",     True,  True,  False),
    ("ens_mtf",     True,  False, True),
    ("vol_mtf",     False, True,  True),
    ("all",         True,  True,  True),
]


# ---------- Training ----------

def build_train_data(*, add_vol: bool, add_mtf: bool):
    """Build training dataset with optional feature extensions.

    Returns dict with X (extended features), y, fwd, feat_cols, holdout_start.
    """
    cfg = load_cfg("v1")
    cfg["splits"] = {**cfg["splits"], "train_months": 35, "val_months": 1}

    log.info("Building dataset (add_vol=%s, add_mtf=%s)…", add_vol, add_mtf)
    X, y, fwd = prepare_dataset(cfg)
    base_feat_cols = feature_columns(X)
    X = X[base_feat_cols].copy()

    # Bring in raw bars for volume/MTF feature computation
    raw = assemble(cfg)
    sym = cfg["universe"]["symbol"].lower()
    X = add_extensions(X, raw, sym=sym, add_volume=add_vol, add_mtf=add_mtf)

    # Fill any new-feature NaNs that arose at the boundary
    new_cols = [c for c in X.columns if c not in base_feat_cols]
    if new_cols:
        X[new_cols] = X[new_cols].ffill().fillna(0.0)

    feat_cols = list(X.columns)
    holdout_start = pd.Timestamp(cfg["window"]["holdout_start"], tz="UTC")
    return {
        "X": X, "y": y, "fwd": fwd, "feat_cols": feat_cols,
        "holdout_start": holdout_start, "cfg": cfg,
    }


def train_one_model(data: dict, seed: int = 42, dropout_features: float = 0.0) -> object:
    """Train a single LightGBM with the given seed and optional feature dropout."""
    X = data["X"]
    y = data["y"]
    holdout_start = data["holdout_start"]
    mask_nonholdout = X.index < holdout_start
    Xn = X[mask_nonholdout]
    last = Xn.index[-1]
    val_start = last - pd.DateOffset(months=1)
    train_start = last - pd.DateOffset(months=36)
    mask_tr = (X.index >= train_start) & (X.index < val_start) & mask_nonholdout
    mask_va = (X.index >= val_start) & mask_nonholdout
    X_tr = X[mask_tr]
    X_va = X[mask_va]

    if dropout_features > 0:
        rng = np.random.default_rng(seed)
        keep = rng.random(len(X.columns)) >= dropout_features
        cols_kept = [c for c, k in zip(X.columns, keep) if k]
        # Always keep at least 30 features
        if len(cols_kept) < 30:
            cols_kept = list(X.columns)
        X_tr = X_tr[cols_kept]
        X_va = X_va[cols_kept]
    else:
        cols_kept = list(X.columns)

    y_tr = y.loc[X_tr.index]
    y_va = y.loc[X_va.index]

    params = dict(DEFAULT_PARAMS)
    params["seed"] = seed
    booster = train_lgb(X_tr, y_tr, X_va, y_va, params, use_weights=False)
    return {"booster": booster, "feature_cols": cols_kept}


def train_variant(tag: str, ensemble: bool, add_vol: bool, add_mtf: bool):
    """Train (or ensemble of) entry models for one combination."""
    log.info("=" * 70)
    log.info(" Training variant: %s (ensemble=%s, vol=%s, mtf=%s)",
             tag, ensemble, add_vol, add_mtf)
    log.info("=" * 70)

    out_dir = ARTIFACT_BASE / f"v3_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_train_data(add_vol=add_vol, add_mtf=add_mtf)
    log.info(f"  total features: {len(data['feat_cols'])} "
             f"({sum(c.startswith('vp_') for c in data['feat_cols'])} vol + "
             f"{sum(c.startswith('mtf_') for c in data['feat_cols'])} mtf + "
             f"{sum(not c.startswith(('vp_','mtf_')) for c in data['feat_cols'])} base)")

    # Train models
    if ensemble:
        seeds = [42, 123, 456, 789, 1000]
        models = []
        for i, seed in enumerate(seeds):
            log.info(f"  ensemble model {i+1}/5 (seed={seed})")
            m = train_one_model(data, seed=seed,
                                dropout_features=0.10 if i > 0 else 0.0)
            models.append(m)
        ensemble_data = [{
            "feature_cols": m["feature_cols"],
            "booster_path": str(out_dir / f"model_{i}.lgb"),
        } for i, m in enumerate(models)]
        for i, m in enumerate(models):
            m["booster"].save_model(ensemble_data[i]["booster_path"])
    else:
        log.info(f"  single model (seed=42)")
        m = train_one_model(data, seed=42)
        m["booster"].save_model(str(out_dir / "model_0.lgb"))
        ensemble_data = [{
            "feature_cols": m["feature_cols"],
            "booster_path": str(out_dir / "model_0.lgb"),
        }]

    # Tune thresholds on validation set using ENSEMBLE PREDICTION
    cfg = data["cfg"]
    fwd = data["fwd"]
    last = data["X"][data["X"].index < data["holdout_start"]].index[-1]
    val_start = last - pd.DateOffset(months=1)
    val_mask = (data["X"].index >= val_start) & (data["X"].index < data["holdout_start"])
    X_va = data["X"][val_mask]
    fwd_va = fwd.loc[X_va.index]

    p_va_avg = np.zeros((len(X_va), 3))
    for entry in ensemble_data:
        booster = lgb.Booster(model_file=entry["booster_path"])
        p_va_avg += booster.predict(X_va[entry["feature_cols"]].values)
    p_va_avg /= len(ensemble_data)
    p_va_df = pd.DataFrame(p_va_avg, index=X_va.index, columns=["p_down", "p_flat", "p_up"])
    up_t, dn_t = tune_thresholds(p_va_df, fwd_va, cfg)
    log.info(f"  tuned thresholds: up={up_t:.2f} dn={dn_t:.2f}")

    # Save bundle metadata
    meta = {
        "tag": tag,
        "ensemble": ensemble,
        "add_volume": add_vol,
        "add_mtf": add_mtf,
        "n_models": len(ensemble_data),
        "feature_cols": data["feat_cols"],
        "thresholds": {"up": float(up_t), "down": float(dn_t)},
        "models": ensemble_data,
        "train_window": [str(data["X"].index[0]), str(data["X"].index[-1])],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info(f"  saved → {out_dir}")
    return meta


def train_all():
    """Train all 8 variants."""
    for tag, ens, vol, mtf in COMBINATIONS:
        try:
            train_variant(tag, ens, vol, mtf)
        except Exception as e:
            log.exception(f"variant {tag} failed: {e}")


# ---------- Backtest ----------

class V3Bundle:
    """Lightweight wrapper to make a v3 variant look like an Artifact for the
    existing backtest pipeline."""

    def __init__(self, tag: str):
        self.tag = tag
        meta = json.loads((ARTIFACT_BASE / f"v3_{tag}" / "meta.json").read_text())
        self.thresholds = meta["thresholds"]
        self.feature_cols = meta["feature_cols"]
        self.models = meta["models"]
        self.boosters = []
        self.boosters_feature_cols = []
        for entry in self.models:
            self.boosters.append(lgb.Booster(model_file=entry["booster_path"]))
            self.boosters_feature_cols.append(entry["feature_cols"])
        # Mimic Artifact for compatibility
        self.booster = self  # so .booster.predict() works
        self.cfg = {}

    def predict(self, X_array: np.ndarray) -> np.ndarray:
        """Average ensemble predictions across all models."""
        # X_array has shape (n, len(feature_cols))
        # We need to slice by each booster's expected columns
        # But X_array doesn't carry column names; assume caller passes the
        # full feature_cols array. We'll need a DF for slicing.
        if len(self.boosters) == 1:
            # Single model — drop extra columns it didn't train on
            cols_kept = self.boosters_feature_cols[0]
            keep_idx = [self.feature_cols.index(c) for c in cols_kept]
            return self.boosters[0].predict(X_array[:, keep_idx])
        # Ensemble
        avg = np.zeros((X_array.shape[0], 3))
        for booster, cols_kept in zip(self.boosters, self.boosters_feature_cols):
            keep_idx = [self.feature_cols.index(c) for c in cols_kept]
            avg += booster.predict(X_array[:, keep_idx])
        return avg / len(self.boosters)


def get_v3_signals_and_bars(start: str, end: str, bundle: V3Bundle,
                              add_vol: bool, add_mtf: bool):
    """Pull signals using a v3 bundle (with optional extended features)."""
    from research.backtest_original import build_frame
    cfg = load_cfg("v1")
    sym = cfg["universe"]["symbol"].lower()

    out = build_frame(start, end, cfg)
    if out.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    feats = build_features(out, cfg)
    feats = add_extensions(feats, out, sym=sym, add_volume=add_vol, add_mtf=add_mtf)
    new_cols = [c for c in feats.columns if c.startswith(("vp_", "mtf_"))]
    if new_cols:
        feats[new_cols] = feats[new_cols].ffill().fillna(0.0)

    # Make sure all bundle.feature_cols exist
    for col in bundle.feature_cols:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[bundle.feature_cols]
    essential = [col for col in bundle.feature_cols
                 if col.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_"))]
    feats = feats[feats[essential].notna().all(axis=1)]
    if feats.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    proba = bundle.predict(feats.values)
    pred = pd.DataFrame(proba, index=feats.index, columns=["p_down", "p_flat", "p_up"])
    pred["close"] = out.loc[pred.index, f"{sym}_close"]

    # Vectorized RTH mask — the .apply() version was looping per-timestamp
    # in Python (250k+ ts × Timestamp.hour access = 5+ min per window).
    minutes_of_day = pred.index.hour * 60 + pred.index.minute
    rth_lo = 13 * 60 + 30 + cfg["risk"]["skip_first_minutes"]
    rth_hi = 20 * 60 - cfg["risk"]["skip_last_minutes"]
    rth_mask = (minutes_of_day >= rth_lo) & (minutes_of_day < rth_hi)
    pred = pred[rth_mask]

    bars = out[[f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close"]].rename(columns={
        f"{sym}_open": "open", f"{sym}_high": "high",
        f"{sym}_low": "low", f"{sym}_close": "close",
    }).dropna()

    thr_up = float(bundle.thresholds["up"])
    thr_dn = float(bundle.thresholds["down"])
    signals = []
    for ts, row in pred.iterrows():
        side = None
        if row["p_up"] >= thr_up:
            side = "long"
        elif row["p_down"] >= thr_dn:
            side = "short"
        if side is None:
            continue
        signals.append({"ts": ts, "side": side,
                         "p_up": float(row["p_up"]),
                         "p_dn": float(row["p_down"])})
    return signals, bars, feats


def backtest_all():
    """Run backtest on all 8 variants × 5 windows."""
    print("=" * 110)
    print(" V3 combinations × 5 windows backtest")
    print("=" * 110)

    # Load v2 exit bundle (shared across all variants)
    v2_exits = load_v2_exits(art_dir=root() / "artifacts/research_v2_exit_tight")
    # Subset of features that v2_exits expects (61 base features from v2_entry)
    v2_exit_feature_cols = v2_exits.feature_cols

    # Load all 8 v3 bundles
    bundles = {}
    for tag, ens, vol, mtf in COMBINATIONS:
        try:
            bundles[tag] = V3Bundle(tag)
        except Exception as e:
            print(f"  skipping {tag}: {e}")

    # Also load current live (research_v2_entry + tight v2 exits) as the comparison baseline
    print(f"  loaded {len(bundles)} v3 variants: {list(bundles.keys())}")
    print()

    summary = {}  # {window_label: {tag: metrics}}

    for label, start, end in WINDOWS:
        print("=" * 110)
        print(f" {label}: {start} → {end}")
        print("=" * 110)

        rows = []
        for tag, ens, vol, mtf in COMBINATIONS:
            if tag not in bundles:
                continue
            bundle = bundles[tag]
            try:
                signals, bars, feats = get_v3_signals_and_bars(
                    start, end, bundle, add_vol=vol, add_mtf=mtf)
            except Exception as e:
                print(f"  {tag}: signal harvest failed: {e}")
                continue
            if not signals:
                rows.append({"name": tag, "n": 0})
                continue
            # Use v2_exit_tight for exits (shared across all variants).
            # Restrict feature row to the v2_exit feature set only.
            v2_feats = feats[v2_exit_feature_cols] if all(
                c in feats.columns for c in v2_exit_feature_cols) else feats
            try:
                sim = make_variant_C(v2_exits, v2_feats, max_notional_frac=2.0)
                trades = walk_with_blocking(signals, bars, sim)
            except Exception as e:
                print(f"  {tag}: backtest failed: {e}")
                continue
            tag_label = f"{tag} ({'ENS' if ens else '   '} {'VOL' if vol else '   '} {'MTF' if mtf else '   '})"
            rec = metrics(tag_label, trades)
            rows.append(rec)
            summary.setdefault(label, {})[tag] = {
                "n": rec.get("n", 0),
                "win": rec.get("win", 0),
                "total_dollars": rec.get("total_dollars", 0),
                "total_bps_net": rec.get("total_bps_net", 0),
                "daily_sharpe": rec.get("daily_sharpe", float("nan")),
                "max_dd_dollars": rec.get("max_dd_dollars", 0),
            }

            # Save trades CSV
            if trades:
                out_dir = Path(__file__).parent / "outputs"
                out_dir.mkdir(exist_ok=True)
                safe_w = f"{start}_{end}".replace("-", "")
                pd.DataFrame(trades).to_csv(
                    out_dir / f"v3_{tag}_{safe_w}.csv", index=False)
        print_table(rows)

    # Save full summary
    summary_path = Path(__file__).parent / "outputs" / "v3_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nFull summary saved to {summary_path}")
    print_winner_summary(summary)


def print_winner_summary(summary: dict):
    """Highlight the best variant per window and overall."""
    print("\n" + "=" * 110)
    print(" Per-window WINNERS")
    print("=" * 110)
    for window, by_tag in summary.items():
        valid = [(t, m) for t, m in by_tag.items() if m.get("n", 0) > 0]
        if not valid:
            continue
        best_dollars = max(valid, key=lambda x: x[1]["total_dollars"])
        best_sharpe = max(valid, key=lambda x: x[1].get("daily_sharpe", -1e9)
                            if x[1].get("daily_sharpe") == x[1].get("daily_sharpe") else -1e9)
        print(f"  {window:<48}")
        print(f"    best $:      {best_dollars[0]:<12}  ${best_dollars[1]['total_dollars']:+12,.0f}")
        print(f"    best Sharpe: {best_sharpe[0]:<12}  {best_sharpe[1]['daily_sharpe']:+5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train all variants")
    ap.add_argument("--backtest", action="store_true", help="backtest all variants")
    ap.add_argument("--all", action="store_true", help="train + backtest")
    args = ap.parse_args()

    if args.all or args.train:
        train_all()
    if args.all or args.backtest:
        backtest_all()


if __name__ == "__main__":
    main()
