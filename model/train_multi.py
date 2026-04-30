"""Cross-sectional training: train one model on multiple target symbols
(e.g., SPY+QQQ+IWM) with a `ticker` categorical feature.

Usage:
    python -m model.train_multi --config cross
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from utils.config import load as load_cfg
from utils.logging import get
from data_pull.assemble_multi import assemble_multi
from features.build import build as build_features
from labels.build import build as build_labels, forward_return
from backtest.costs import round_trip_cost_bp
from model.artifact import Artifact, save as save_artifact
from model.train import (DEFAULT_PARAMS, _multi_logloss, predict_proba,
                          simulate_one_position, _aggregate_metrics)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = get("model.train_multi")

TICKER_FEATURE = "ticker_id"


# ---- dataset builder: ONE combined frame, no separate Series ----

def prepare_dataset(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build a single DataFrame with features + label/fwd/ticker columns appended.

    Returning ONE frame avoids the duplicate-timestamp `.loc` blowup we'd hit if
    X, y, and fwd were independent Series — boolean masks on the combined frame
    guarantee positional alignment of everything.
    """
    targets = cfg["universe"]["targets"]
    sym_to_id = {s: i for i, s in enumerate(targets)}
    frames = assemble_multi(cfg, targets)

    rows = []
    for sym, df in frames.items():
        feats = build_features(df, cfg)
        feats["_y"] = build_labels(df, cfg).astype("int8")
        feats["_fwd"] = forward_return(df, cfg["label"]["horizon_min"])
        feats["_ticker"] = sym
        feats[TICKER_FEATURE] = sym_to_id[sym]
        feats = feats[feats["_y"] != -1]
        essential = [c for c in feats.columns
                     if c.startswith(("ret_", "rsi_", "macd", "bb_pctb_", "rvol_",
                                      "vwap_dev", "session_"))]
        feats = feats[feats[essential].notna().all(axis=1)]
        rows.append(feats)
        log.info("  %s rows=%d", sym, len(feats))

    combined = pd.concat(rows).sort_index(kind="mergesort")
    combined[TICKER_FEATURE] = combined[TICKER_FEATURE].astype("category")

    feature_cols = sorted([c for c in combined.columns
                           if c not in ("_y", "_fwd", "_ticker")])
    log.info("combined shape=%s tickers=%s class_dist=%s",
             combined.shape, list(sym_to_id.keys()),
             combined["_y"].value_counts().to_dict())
    return combined, feature_cols


# ---- LightGBM helpers ----

def train_lgb(data_tr: pd.DataFrame, data_va: pd.DataFrame, feature_cols: list[str],
              params: dict, num_boost: int = 2000, esr: int = 50) -> lgb.Booster:
    X_tr = data_tr[feature_cols]; y_tr = data_tr["_y"]
    X_va = data_va[feature_cols]; y_va = data_va["_y"]
    cat = [TICKER_FEATURE]
    d_tr = lgb.Dataset(X_tr, label=y_tr.values, categorical_feature=cat,
                       free_raw_data=False)
    d_va = lgb.Dataset(X_va, label=y_va.values, reference=d_tr,
                       categorical_feature=cat, free_raw_data=False)
    booster = lgb.train(
        params, d_tr,
        num_boost_round=num_boost,
        valid_sets=[d_tr, d_va], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def tune_hyperparams(data_tr, data_va, feature_cols, n_trials: int) -> dict:
    def objective(trial):
        p = dict(DEFAULT_PARAMS)
        p.update(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 1000, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        )
        bst = train_lgb(data_tr, data_va, feature_cols, p, num_boost=500, esr=25)
        preds = bst.predict(data_va[feature_cols], num_iteration=bst.best_iteration)
        return _multi_logloss(data_va["_y"], preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(DEFAULT_PARAMS)
    best.update(study.best_params)
    log.info("optuna best value=%.4f params=%s", study.best_value, study.best_params)
    return best


# ---- threshold tuning + metrics (operate per-ticker since duplicates would corrupt the simulator) ----

def simulate_combined(data_te: pd.DataFrame, proba: pd.DataFrame,
                      up_thr: float, dn_thr: float, horizon: int, cost: float
                      ) -> tuple[float, int, pd.Series]:
    """One-position-at-a-time simulator applied per-ticker, then summed."""
    pnl_total = 0.0
    n_total = 0
    pieces = []
    for sym in data_te["_ticker"].unique():
        sub = data_te[data_te["_ticker"] == sym]
        if len(sub) == 0:
            continue
        # Align proba to the same rows.
        sub_proba = proba.loc[sub.index]
        if sub_proba.index.has_duplicates:
            # Recover positionally using sub.index alignment trick: use values directly.
            sub_proba = pd.DataFrame(
                proba.values[data_te["_ticker"].values == sym],
                index=sub.index, columns=proba.columns,
            )
        sub_fwd = sub["_fwd"]
        pnl, n, series = simulate_one_position(sub_proba, sub_fwd, up_thr, dn_thr,
                                                horizon, cost)
        pnl_total += pnl
        n_total += n
        pieces.append(series)
    if pieces:
        full = pd.concat(pieces).sort_index(kind="mergesort")
    else:
        full = pd.Series(dtype=float)
    return pnl_total, n_total, full


def tune_thresholds(data: pd.DataFrame, proba: pd.DataFrame, cfg: dict
                    ) -> tuple[float, float]:
    tcfg = cfg["threshold_tuning"]
    floor = float(tcfg["threshold_floor"])
    min_trades = int(tcfg["min_trades"])
    step = float(tcfg["grid_step"])
    horizon = int(cfg["label"]["horizon_min"])
    cost = round_trip_cost_bp(cfg) / 1e4

    grid = np.arange(floor, 0.90, step)
    best, best_pnl = (floor, floor), -np.inf
    for up in grid:
        for down in grid:
            pnl, n, _ = simulate_combined(data, proba, up, down, horizon, cost)
            if n < min_trades:
                continue
            if pnl > best_pnl:
                best_pnl, best = pnl, (float(up), float(down))
    log.info("threshold tune up=%.2f down=%.2f pnl_bp=%.1f trades_min=%d",
             best[0], best[1], best_pnl * 1e4, min_trades)
    return best


def per_ticker_metrics(data: pd.DataFrame, proba: pd.DataFrame,
                       up_t: float, dn_t: float, horizon: int, cost: float) -> dict:
    out = {}
    for sym in data["_ticker"].unique():
        sub = data[data["_ticker"] == sym]
        sub_proba = pd.DataFrame(
            proba.values[data["_ticker"].values == sym],
            index=sub.index, columns=proba.columns,
        )
        sub_fwd = sub["_fwd"]
        pnl, n, series = simulate_one_position(sub_proba, sub_fwd, up_t, dn_t,
                                                horizon, cost)
        trade_rets = series[series != 0]
        daily = series.groupby(series.index.tz_convert("America/New_York").date).sum()
        daily = daily[daily != 0]
        ds = (float(daily.mean() / daily.std() * np.sqrt(252))
              if len(daily) >= 5 and daily.std() > 0 else float("nan"))
        out[sym] = {
            "trades": int(n),
            "net_pnl_bp": float(pnl * 1e4),
            "daily_sharpe": ds,
            "hit_rate": float((trade_rets > 0).mean()) if len(trade_rets) else float("nan"),
            "avg_trade_bp": float(trade_rets.mean() * 1e4) if len(trade_rets) else float("nan"),
        }
    return out


# ---- run ----

def run(cfg: dict, tune: bool) -> Artifact:
    data, feature_cols = prepare_dataset(cfg)
    holdout_start = pd.Timestamp(cfg["window"]["holdout_start"], tz="UTC")
    nh = data[data.index < holdout_start]

    tuning_frac = float(cfg.get("tuning_frac", 0.5))
    if tune:
        cutoff = int(len(nh) * tuning_frac)
        sub = nh.iloc[:cutoff]
        val_n = int(len(sub) * 0.25)
        params = tune_hyperparams(sub.iloc[:-val_n], sub.iloc[-val_n:], feature_cols,
                                  n_trials=cfg["model"]["optuna_trials"])
    else:
        params = DEFAULT_PARAMS

    horizon = int(cfg["label"]["horizon_min"])
    cost_frac = round_trip_cost_bp(cfg) / 1e4

    # Walk-forward by month, but operate on the combined dataframe.
    et_idx = nh.index.tz_convert("America/New_York")
    months = pd.PeriodIndex(et_idx, freq="M")
    unique_months = sorted(months.unique())
    wf_start_idx = int(len(nh) * tuning_frac)
    wf_start_month = pd.PeriodIndex(nh.index[wf_start_idx:wf_start_idx + 1].tz_convert("America/New_York"),
                                    freq="M")[0]
    nh_months = list(unique_months)

    train_months = cfg["splits"]["train_months"]
    val_months = cfg["splits"]["val_months"]

    fold_metrics = []
    for i in range(nh_months.index(wf_start_month), len(nh_months) - 0):
        test_m = nh_months[i]
        if i - val_months < 0 or i - val_months - train_months < 0:
            continue
        val_m = nh_months[i - val_months]
        train_first = nh_months[i - val_months - train_months]
        train_last = nh_months[i - val_months - 1]

        m_arr = pd.PeriodIndex(nh.index.tz_convert("America/New_York"), freq="M")
        tr_mask = (m_arr >= train_first) & (m_arr <= train_last)
        va_mask = m_arr == val_m
        te_mask = m_arr == test_m

        d_tr = nh.iloc[tr_mask]; d_va = nh.iloc[va_mask]; d_te = nh.iloc[te_mask]
        if len(d_tr) == 0 or len(d_va) == 0 or len(d_te) == 0:
            continue

        bst = train_lgb(d_tr, d_va, feature_cols, params)

        # Predict on val for threshold tuning.
        p_va = predict_proba(bst, d_va[feature_cols])
        up_t, dn_t = tune_thresholds(d_va, p_va, cfg)

        # Eval on test.
        p_te = predict_proba(bst, d_te[feature_cols])
        pnl_total, n_trades, pnl_series = simulate_combined(
            d_te, p_te, up_t, dn_t, horizon, cost_frac)
        trade_rets = pnl_series[pnl_series != 0]
        daily = pnl_series.groupby(
            pnl_series.index.tz_convert("America/New_York").date).sum()
        daily = daily[daily != 0]
        ds = (float(daily.mean() / daily.std() * np.sqrt(252))
              if len(daily) >= 5 and daily.std() > 0 else float("nan"))
        per_t = per_ticker_metrics(d_te, p_te, up_t, dn_t, horizon, cost_frac)
        m = {
            "fold": i,
            "test_start": str(test_m),
            "daily_sharpe": ds,
            "net_pnl_bp": float(pnl_total * 1e4),
            "trades": int(n_trades),
            "hit_rate": float((trade_rets > 0).mean()) if len(trade_rets) else float("nan"),
            "avg_trade_bp": float(trade_rets.mean() * 1e4) if len(trade_rets) else float("nan"),
            "active_days": int(len(daily)),
            "up_thr": up_t, "down_thr": dn_t,
            "per_ticker": per_t,
        }
        fold_metrics.append(m)
        per_t_str = ", ".join(f"{k}:{v['trades']}t/{v['net_pnl_bp']:+.0f}" for k, v in per_t.items())
        log.info("fold %d test=%s sharpe=%.2f pnl_bp=%+.1f trades=%d hit=%.2f [%s]",
                 i, str(test_m), ds, m["net_pnl_bp"], n_trades, m["hit_rate"], per_t_str)

    # Final fit on the most recent train+val months in non-holdout.
    last_idx = len(nh_months) - 1
    val_first = nh_months[last_idx - val_months]
    train_first = nh_months[last_idx - val_months - train_months]
    m_arr = pd.PeriodIndex(nh.index.tz_convert("America/New_York"), freq="M")
    tr_mask = (m_arr >= train_first) & (m_arr < val_first)
    va_mask = m_arr >= val_first
    d_tr = nh.iloc[tr_mask]; d_va = nh.iloc[va_mask]
    log.info("final fit: train=%d val=%d", len(d_tr), len(d_va))
    final = train_lgb(d_tr, d_va, feature_cols, params)
    p_va = predict_proba(final, d_va[feature_cols])
    up_t, dn_t = tune_thresholds(d_va, p_va, cfg)

    agg = _aggregate_metrics(fold_metrics)
    log.info("walk-forward aggregate: %s", json.dumps(agg, indent=2, default=str))

    # Holdout.
    d_ho = data[data.index >= holdout_start]
    holdout = {"n_rows": int(len(d_ho))}
    if len(d_ho) > 0:
        p_ho = predict_proba(final, d_ho[feature_cols])
        pnl_total, n_trades, pnl_series = simulate_combined(
            d_ho, p_ho, up_t, dn_t, horizon, cost_frac)
        trade_rets = pnl_series[pnl_series != 0]
        daily = pnl_series.groupby(
            pnl_series.index.tz_convert("America/New_York").date).sum()
        daily = daily[daily != 0]
        ds = (float(daily.mean() / daily.std() * np.sqrt(252))
              if len(daily) >= 5 and daily.std() > 0 else float("nan"))
        holdout.update({
            "trades": int(n_trades),
            "net_pnl_bp": float(pnl_total * 1e4),
            "daily_sharpe": ds,
            "hit_rate": float((trade_rets > 0).mean()) if len(trade_rets) else float("nan"),
            "avg_trade_bp": float(trade_rets.mean() * 1e4) if len(trade_rets) else float("nan"),
            "per_ticker": per_ticker_metrics(d_ho, p_ho, up_t, dn_t, horizon, cost_frac),
        })
    log.info("HOLDOUT: %s", json.dumps(holdout, indent=2, default=str))

    art = Artifact(
        booster=final,
        feature_cols=feature_cols,
        thresholds={"up": up_t, "down": dn_t},
        cfg=cfg,
        train_window=(str(d_tr.index[0]), str(d_tr.index[-1])),
        metrics={"walk_forward": fold_metrics, "aggregate": agg, "holdout": holdout, "params": params},
    )
    run_name = cfg.get("run_name", "cross")
    out_dir = save_artifact(art, run_name)
    log.info("artifact saved -> %s", out_dir)
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cross")
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    run(cfg, tune=not args.no_tune)


if __name__ == "__main__":
    main()
