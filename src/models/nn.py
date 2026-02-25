from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.validation.cv import FoldConfig, get_fold_indices, make_stratified_folds
from src.validation.metrics import fold_metrics_summary


def train_mlp_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str = "TARGET",
    fold_cfg: FoldConfig | None = None,
    model_params: Dict | None = None,
    model_version: str = "mlp_v1",
    artifacts_dir: Path | str = Path("artifacts"),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simple MLP with CV. This is optional and mainly adds diversity to the ensemble.
    """
    if fold_cfg is None:
        fold_cfg = FoldConfig()
    if model_params is None:
        model_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-3,
            "batch_size": 256,
            "learning_rate": "adaptive",
            "max_iter": 100,
            "random_state": 42,
        }

    artifacts_dir = Path(artifacts_dir)
    models_dir = artifacts_dir / "models"
    oof_dir = artifacts_dir / "oof"
    preds_dir = artifacts_dir / "predictions"
    metrics_dir = artifacts_dir / "metrics"
    for d in (models_dir, oof_dir, preds_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_with_folds = make_stratified_folds(train_df, target_col=target_col, cfg=fold_cfg)
    fold_indices = get_fold_indices(train_with_folds, n_splits=fold_cfg.n_splits)

    X_all = train_with_folds[features].values.astype("float32")
    y_all = train_with_folds[target_col].values
    X_test_all = test_df[features].values.astype("float32")

    oof_pred = np.zeros(len(train_with_folds), dtype=float)
    test_pred_folds = np.zeros((fold_cfg.n_splits, len(test_df)), dtype=float)
    fold_scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(fold_indices):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_valid, y_valid = X_all[valid_idx], y_all[valid_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test_all)

        clf = MLPClassifier(**model_params)
        clf.fit(X_train_scaled, y_train)

        valid_pred = clf.predict_proba(X_valid_scaled)[:, 1]
        oof_pred[valid_idx] = valid_pred
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)

        test_pred_folds[fold] = clf.predict_proba(X_test_scaled)[:, 1]

        dump(
            {"scaler": scaler, "model": clf},
            models_dir / f"{model_version}_fold{fold}.pkl",
        )

    test_pred = test_pred_folds.mean(axis=0)

    oof_df = train_with_folds[["SK_ID_CURR", target_col]].copy()
    oof_df[f"pred_{model_version}"] = oof_pred
    oof_df.to_csv(oof_dir / f"oof_{model_version}.csv", index=False)

    test_pred_df = pd.DataFrame(
        {"SK_ID_CURR": test_df["SK_ID_CURR"], f"pred_{model_version}": test_pred}
    )
    test_pred_df.to_csv(preds_dir / f"pred_{model_version}_test.csv", index=False)

    metrics = {
        "fold_auc": fold_scores,
        **fold_metrics_summary(fold_scores),
    }
    with (metrics_dir / f"run_{model_version}.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(metrics, f, indent=2)

    return oof_pred, test_pred, metrics

