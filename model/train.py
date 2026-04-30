"""Train LightGBM with walk-forward validation.

Usage:
    python -m model.train              # assemble data, tune, walk-forward, save artifact
    python -m model.train --no-tune    # skip optuna, use defaults
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils.config import load as load_cfg
from utils.env import root
from utils.logging import get
from data_pull.assemble import assemble
from features.build import build as build_features, feature_columns
from labels.build import build as build_labels, forward_return
from backtest.walk_forward import fold_iter, tuning_split
from backtest.costs import compute_metrics, round_trip_cost_bp
from model.artifact import Artifact, save as save_artifact

warnings.filterwarnings("ignore", category=UserWarning)

log = get("model.train")


DEFAULT_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    verbosity=-1,
)


def prepare_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = assemble(cfg)
    X = build_features(df, cfg)
    y = build_labels(df, cfg)
    fwd = forward_return(df, cfg["label"]["horizon_min"], cfg)
    # Trim warmup + tail (label invalid for last `horizon` rows).
    mask = y != -1
    X = X.loc[mask]
    y = y.loc[mask]
    fwd = fwd.loc[mask]
    # LightGBM handles NaN natively for missing cross-asset / macro features.
    # Only drop rows where essential SPY-based features are unavailable.
    essential = [c for c in X.columns if c.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_", "vwap_dev", "session_"))]
    essential = [c for c in essential if c in X.columns]
    good = X.index[X[essential].notna().all(axis=1)]
    X = X.loc[good]
    y = y.loc[good]
    fwd = fwd.loc[good]
    log.info("dataset ready shape=%s classes=%s", X.shape, y.value_counts().to_dict())
    return X, y, fwd


def class_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().sort_index()
    n = len(y)
    weights_map = {c: n / (len(counts) * counts[c]) for c in counts.index}
    return y.map(weights_map).astype(float).values


def train_lgb(X_tr, y_tr, X_va, y_va, params: dict, num_boost: int = 2000, esr: int = 50,
              use_weights: bool = False) -> lgb.Booster:
    w = class_weights(y_tr) if use_weights else None
    d_tr = lgb.Dataset(X_tr, label=y_tr, weight=w)
    d_va = lgb.Dataset(X_va, label=y_va, reference=d_tr)
    booster = lgb.train(
        params,
        d_tr,
        num_boost_round=num_boost,
        valid_sets=[d_tr, d_va],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def tune_hyperparams(X_tr, y_tr, X_va, y_va, n_trials: int, cfg: dict) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = dict(DEFAULT_PARAMS)
        params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 1000, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        )
        booster = train_lgb(X_tr, y_tr, X_va, y_va, params, num_boost=500, esr=25,
                            use_weights=cfg["model"].get("use_class_weights", False))
        preds = booster.predict(X_va, num_iteration=booster.best_iteration)
        return _multi_logloss(y_va, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(DEFAULT_PARAMS)
    best.update(study.best_params)
    log.info("optuna best value=%.4f params=%s", study.best_value, study.best_params)
    return best


def _multi_logloss(y, p) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    n = len(y)
    return -np.log(p[np.arange(n), y.values]).mean()


def predict_proba(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    p = booster.predict(X, num_iteration=booster.best_iteration or booster.num_trees())
    return pd.DataFrame(p, index=X.index, columns=["p_down", "p_flat", "p_up"])


def build_signal(proba: pd.DataFrame, up_thr: float, down_thr: float) -> pd.Series:
    sig = pd.Series(0, index=proba.index, dtype=int)
    sig[proba["p_up"] >= up_thr] = 1
    sig[proba["p_down"] >= down_thr] = -1
    return sig


def simulate_one_position(proba: pd.DataFrame, fwd: pd.Series, up_thr: float, down_thr: float,
                          horizon: int, cost: float) -> tuple[float, int, pd.Series]:
    """Simulate a one-position-at-a-time policy with fixed `horizon`-minute holds.

    Returns (total_pnl, num_trades, pnl_series).
    This matches the live loop's behavior and the disjoint-Sharpe assumption.
    """
    p_up = proba["p_up"].values
    p_dn = proba["p_down"].values
    fwd_v = fwd.reindex(proba.index).values
    n = len(proba)
    pnl_arr = np.zeros(n, dtype=float)
    in_pos_until = -1
    trades = 0
    for i in range(n):
        if i < in_pos_until:
            continue
        if p_up[i] >= up_thr:
            side = 1
        elif p_dn[i] >= down_thr:
            side = -1
        else:
            continue
        if np.isnan(fwd_v[i]):
            continue
        pnl_arr[i] = side * fwd_v[i] - cost
        in_pos_until = i + horizon
        trades += 1
    return float(pnl_arr.sum()), trades, pd.Series(pnl_arr, index=proba.index)


def tune_thresholds(proba: pd.DataFrame, fwd: pd.Series, cfg: dict) -> tuple[float, float]:
    """Sweep (up_thr, down_thr), pick pair maximizing realistic (non-overlapping) net PnL.

    Constraints:
      - up_thr, down_thr >= threshold_floor (no grid-floor degenerate solutions)
      - at least min_trades fire on the val set (no overfit to a handful of lucky bars)
    """
    tcfg = cfg.get("threshold_tuning", {"threshold_floor": 0.55, "min_trades": 100, "grid_step": 0.02})
    floor = float(tcfg["threshold_floor"])
    min_trades = int(tcfg["min_trades"])
    step = float(tcfg["grid_step"])
    horizon = int(cfg["label"]["horizon_min"])
    cost = round_trip_cost_bp(cfg) / 1e4

    grid = np.arange(floor, 0.90, step)
    best = (floor, floor)
    best_pnl = -np.inf
    for up in grid:
        for down in grid:
            pnl, n, _ = simulate_one_position(proba, fwd, up, down, horizon, cost)
            if n < min_trades:
                continue
            if pnl > best_pnl:
                best_pnl = pnl
                best = (float(up), float(down))
    log.info("threshold tune best up=%.2f down=%.2f pnl_bp=%.1f (floor=%.2f min_trades=%d)",
             best[0], best[1], best_pnl * 1e4, floor, min_trades)
    return best


def run(cfg: dict, tune: bool) -> Artifact:
    X, y, fwd = prepare_dataset(cfg)
    feat_cols = feature_columns(X)
    X = X[feat_cols]

    # Guard against holdout contamination.
    holdout_start = pd.Timestamp(cfg["window"]["holdout_start"], tz="UTC")
    mask_nonholdout = X.index < holdout_start
    Xn, yn, fwdn = X[mask_nonholdout], y[mask_nonholdout], fwd[mask_nonholdout]

    # Hyperparam tuning on first `tuning_frac` of non-holdout data.
    tuning_frac = float(cfg.get("tuning_frac", 0.5))
    if tune:
        tr_idx, va_idx = tuning_split(Xn.index, frac=tuning_frac, val_frac=0.25)
        params = tune_hyperparams(Xn.loc[tr_idx], yn.loc[tr_idx], Xn.loc[va_idx], yn.loc[va_idx],
                                  n_trials=cfg["model"]["optuna_trials"], cfg=cfg)
    else:
        params = _params_from_meta(cfg.get("run_name", "v1")) or DEFAULT_PARAMS
        log.info("no-tune: using params=%s", params)

    # Walk-forward: from the post-tuning portion through the end of non-holdout data.
    # Bounded to Xn so folds never touch the final holdout month.
    fold_metrics = []
    wf_start = Xn.index[int(len(Xn) * tuning_frac)]
    Xw = Xn[Xn.index >= wf_start]
    folds = list(fold_iter(Xw.index,
                           train_months=cfg["splits"]["train_months"],
                           val_months=cfg["splits"]["val_months"],
                           test_months=cfg["splits"]["test_months"]))
    log.info("walk-forward folds=%d", len(folds))
    horizon = int(cfg["label"]["horizon_min"])
    cost_frac = round_trip_cost_bp(cfg) / 1e4
    use_weights = cfg["model"].get("use_class_weights", False)
    for i, f in enumerate(folds):
        tr_mask, va_mask, te_mask = f["train_mask"], f["val_mask"], f["test_mask"]
        X_tr = Xw[tr_mask]; y_tr = y.loc[X_tr.index]
        X_va = Xw[va_mask]; y_va = y.loc[X_va.index]
        X_te = Xw[te_mask]; y_te = y.loc[X_te.index]
        if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
            continue
        booster = train_lgb(X_tr, y_tr, X_va, y_va, params, use_weights=use_weights)
        p_va = predict_proba(booster, X_va)
        up_t, dn_t = tune_thresholds(p_va, fwd.loc[X_va.index], cfg)
        p_te = predict_proba(booster, X_te)

        # Per-fold eval: simulate the live one-position-at-a-time policy.
        pnl_total, n_trades, pnl_series = simulate_one_position(
            p_te, fwd.loc[p_te.index], up_t, dn_t, horizon, cost_frac)
        trade_rets = pnl_series[pnl_series != 0]
        # Daily Sharpe: aggregate PnL per trading day, annualize by sqrt(252).
        # This is the defensible Sharpe number; per-trade Sharpe with period_per_year=6552
        # overstates because you don't actually trade every 15-min slot.
        daily_pnl = pnl_series.groupby(pnl_series.index.tz_convert("America/New_York").date).sum()
        daily_pnl = daily_pnl[daily_pnl != 0]  # only trading days
        if len(daily_pnl) >= 5 and daily_pnl.std() > 0:
            daily_sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
        else:
            daily_sharpe = float("nan")
        hit = float((trade_rets > 0).mean()) if len(trade_rets) else float("nan")
        m = {
            "fold": i,
            "test_start": str(f["test"][0]),
            "daily_sharpe": daily_sharpe,
            "net_pnl_bp": float(pnl_total * 1e4),
            "trades": int(n_trades),
            "active_days": int(len(daily_pnl)),
            "hit_rate": hit,
            "avg_trade_bp": float(trade_rets.mean() * 1e4) if len(trade_rets) else float("nan"),
            "up_thr": up_t, "down_thr": dn_t,
        }
        fold_metrics.append(m)
        log.info("fold %d test=%s dailySharpe=%.2f pnl_bp=%+.1f trades=%d days=%d hit=%.2f avg_bp=%+.1f",
                 i, m["test_start"], m["daily_sharpe"], m["net_pnl_bp"], m["trades"],
                 m["active_days"], m["hit_rate"], m["avg_trade_bp"])

    # Final artifact: train on the last train_months months of non-holdout data.
    # Use the non-holdout end, not the full data end — otherwise val window collapses to 0 rows.
    last = Xn.index[-1]
    train_start = last - pd.DateOffset(months=cfg["splits"]["train_months"] + cfg["splits"]["val_months"])
    val_start = last - pd.DateOffset(months=cfg["splits"]["val_months"])
    mask_tr = (X.index >= train_start) & (X.index < val_start) & mask_nonholdout
    mask_va = (X.index >= val_start) & mask_nonholdout
    X_tr, y_tr = X[mask_tr], y.loc[X[mask_tr].index]
    X_va, y_va = X[mask_va], y.loc[X[mask_va].index]
    if len(X_tr) == 0 or len(X_va) == 0:
        raise RuntimeError(f"final fit windows empty: train={len(X_tr)} val={len(X_va)}")
    log.info("final fit: train=%d val=%d", len(X_tr), len(X_va))
    final_booster = train_lgb(X_tr, y_tr, X_va, y_va, params, use_weights=use_weights)
    p_va = predict_proba(final_booster, X_va)
    up_t, dn_t = tune_thresholds(p_va, fwd.loc[X_va.index], cfg)

    agg = _aggregate_metrics(fold_metrics)
    log.info("walk-forward aggregate: %s", json.dumps(agg, indent=2))

    # --- Untouched holdout evaluation ---
    # Use the model we just trained (final_booster) to predict on the holdout window
    # that was never seen during tuning or walk-forward.
    holdout_mask = (X.index >= holdout_start)
    X_ho = X[holdout_mask]
    fwd_ho = fwd.loc[X_ho.index]
    holdout = {"n_rows": int(len(X_ho))}
    if len(X_ho) > 0:
        p_ho = predict_proba(final_booster, X_ho)
        pnl_total, n_trades, pnl_series = simulate_one_position(
            p_ho, fwd_ho, up_t, dn_t, horizon, cost_frac)
        trade_rets = pnl_series[pnl_series != 0]
        daily_pnl = pnl_series.groupby(
            pnl_series.index.tz_convert("America/New_York").date).sum()
        daily_pnl = daily_pnl[daily_pnl != 0]
        if len(daily_pnl) >= 5 and daily_pnl.std() > 0:
            ds = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
        else:
            ds = float("nan")
        holdout.update({
            "net_pnl_bp": float(pnl_total * 1e4),
            "daily_sharpe": ds,
            "trades": int(n_trades),
            "active_days": int(len(daily_pnl)),
            "hit_rate": float((trade_rets > 0).mean()) if len(trade_rets) else float("nan"),
            "avg_trade_bp": float(trade_rets.mean() * 1e4) if len(trade_rets) else float("nan"),
            "pct_on_portfolio_10pct_size": float(pnl_total * 0.10 * 100),
        })
    log.info("HOLDOUT (untouched %s-onwards): %s", holdout_start.date(), json.dumps(holdout, indent=2))

    art = Artifact(
        booster=final_booster,
        feature_cols=feat_cols,
        thresholds={"up": up_t, "down": dn_t},
        cfg=cfg,
        train_window=(str(X_tr.index[0]), str(X_tr.index[-1])),
        metrics={"walk_forward": fold_metrics, "aggregate": agg, "holdout": holdout, "params": params},
    )
    # Each run goes into artifacts/{run_name}/ so different experiments don't overwrite each other.
    # The 'latest' pointer is what the live loop reads — only update it if the config explicitly says so.
    run_name = cfg.get("run_name", "latest")
    out_dir = save_artifact(art, run_name)
    log.info("artifact saved -> %s", out_dir)
    if cfg.get("update_latest", run_name == "v1"):
        latest_dir = save_artifact(art, "latest")
        log.info("'latest' pointer updated -> %s", latest_dir)
    else:
        log.info("'latest' pointer NOT updated (this is a sandbox run)")
    return art


def _params_from_meta(run_name: str) -> dict | None:
    """Reuse tuned LightGBM params saved by a previous run, so --no-tune
    can reproduce the prior model without re-running optuna.

    Looks first under the configured run_name, then falls back to 'latest' —
    artifacts/latest is the canonical pointer the live loop reads, and the
    only one guaranteed to exist if the run_name dir was cleaned up.
    """
    artifacts_dir = root() / "artifacts"
    for candidate in (run_name, "latest"):
        meta_path = artifacts_dir / candidate / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        params = meta.get("metrics", {}).get("params")
        if not params:
            continue
        log.info("loaded tuned params from artifacts/%s/meta.json", candidate)
        return {k: v for k, v in params.items() if k != "verbosity"} | {"verbosity": -1}
    return None


def _aggregate_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    df = pd.DataFrame(fold_metrics)
    total_pnl = float(df["net_pnl_bp"].sum())
    winners = int((df["net_pnl_bp"] > 0).sum())
    return {
        "n_folds": len(df),
        "winning_folds": winners,
        "losing_folds": int((df["net_pnl_bp"] < 0).sum()),
        "cumulative_pnl_bp_on_notional": total_pnl,
        "cumulative_pnl_pct_on_portfolio_10pct_size": total_pnl / 1e4 * 0.10 * 100,
        "mean_pnl_bp": float(df["net_pnl_bp"].mean()),
        "median_pnl_bp": float(df["net_pnl_bp"].median()),
        "mean_daily_sharpe": float(df["daily_sharpe"].mean(skipna=True)),
        "median_daily_sharpe": float(df["daily_sharpe"].median(skipna=True)),
        "std_daily_sharpe": float(df["daily_sharpe"].std(skipna=True)),
        "mean_hit_rate": float(df["hit_rate"].mean(skipna=True)),
        "mean_trades_per_fold": float(df["trades"].mean()),
        "mean_avg_trade_bp": float(df["avg_trade_bp"].mean(skipna=True)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--config", default="v1")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    run(cfg, tune=not args.no_tune)


if __name__ == "__main__":
    main()
