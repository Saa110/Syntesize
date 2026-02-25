from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import optuna

from src.data.features import build_features
from src.data.load_data import load_train_test
from src.validation.cv import FoldConfig, make_stratified_folds, get_fold_indices
from src.validation.metrics import roc_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _objective_cat(trial: optuna.Trial) -> float:
    from catboost import CatBoostClassifier

    train_df, _ = load_train_test(PROJECT_ROOT)
    features_df, feature_cols, categorical_features = build_features(train_df)
    features_df["TARGET"] = train_df["TARGET"].values

    cfg = FoldConfig(n_splits=3, shuffle=True, random_state=42)
    df_folds = make_stratified_folds(features_df, target_col="TARGET", cfg=cfg)
    fold_indices = get_fold_indices(df_folds, n_splits=cfg.n_splits)

    X = df_folds[feature_cols]
    y = df_folds["TARGET"].values

    params: Dict = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "iterations": 1500,
        "verbose": False,
        "task_type": "CPU",
    }

    cat_indices: List[int] = []
    for c in categorical_features:
        if c in feature_cols:
            cat_indices.append(feature_cols.index(c))

    fold_scores: List[float] = []
    for fold, (train_idx, valid_idx) in enumerate(fold_indices):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        clf = CatBoostClassifier(**params)
        clf.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_indices,
        )
        preds = clf.predict_proba(X_valid)[:, 1]
        fold_scores.append(roc_auc(y_valid, preds))
        print(f"Fold {fold} AUC: {fold_scores[-1]}")
    
    return float(sum(fold_scores) / len(fold_scores))


def main() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(_objective_cat, n_trials=50)

    best_params = study.best_params
    print("Best CatBoost params:", best_params)

    out_path = PROJECT_ROOT / "config" / "config_catboost_tuned.yaml"
    import yaml

    cfg = {
        "model_version": "cat_tuned",
        "folds": {"n_splits": 5, "shuffle": True, "random_state": 42},
        "params": best_params,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


if __name__ == "__main__":
    main()

