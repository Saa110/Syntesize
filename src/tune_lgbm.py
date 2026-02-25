from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import optuna
import pandas as pd

from src.data.features import build_features
from src.data.load_data import load_train_test
from src.validation.cv import FoldConfig, make_stratified_folds, get_fold_indices
from src.validation.metrics import roc_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _objective_lgbm(trial: optuna.Trial) -> float:
    import lightgbm as lgb

    train_df, _ = load_train_test(PROJECT_ROOT)
    features_df, feature_cols, categorical_features = build_features(train_df)
    features_df["TARGET"] = train_df["TARGET"].values

    cfg = FoldConfig(n_splits=3, shuffle=True, random_state=42)
    df_folds = make_stratified_folds(features_df, target_col="TARGET", cfg=cfg)
    fold_indices = get_fold_indices(df_folds, n_splits=cfg.n_splits)

    X = df_folds[feature_cols].values
    y = df_folds["TARGET"].values

    params: Dict = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "n_estimators": 1500,
        "n_jobs": -1,
    }

    # optional imbalance handling via scale_pos_weight
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos > 0:
        base_spw = neg / pos
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight", base_spw * 0.5, base_spw * 1.5
        )

    fold_scores: List[float] = []
    for fold, (train_idx, valid_idx) in enumerate(fold_indices):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
        )
        preds = clf.predict_proba(X_valid)[:, 1]
        fold_scores.append(roc_auc(y_valid, preds))

    return float(sum(fold_scores) / len(fold_scores))


def main() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(_objective_lgbm, n_trials=50)

    best_params = study.best_params
    print("Best LightGBM params:", best_params)

    # Persist best params to a tuned config file
    out_path = PROJECT_ROOT / "config" / "config_lgbm_tuned.yaml"
    import yaml

    cfg = {
        "model_version": "lgbm_tuned",
        "folds": {"n_splits": 5, "shuffle": True, "random_state": 42},
        "params": best_params,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


if __name__ == "__main__":
    main()

