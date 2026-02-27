from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier

from src.validation.cv import FoldConfig, get_fold_indices, make_stratified_folds
from src.validation.metrics import roc_auc, fold_metrics_summary


def _ensure_dirs(artifacts_dir: Path) -> Dict[str, Path]:
    models_dir = artifacts_dir / "models"
    oof_dir = artifacts_dir / "oof"
    preds_dir = artifacts_dir / "predictions"
    metrics_dir = artifacts_dir / "metrics"

    for d in (models_dir, oof_dir, preds_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "models": models_dir,
        "oof": oof_dir,
        "preds": preds_dir,
        "metrics": metrics_dir,
    }


def train_xgb_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str = "TARGET",
    fold_cfg: FoldConfig | None = None,
    model_params: Dict | None = None,
    model_version: str = "xgb_v1",
    artifacts_dir: Path | str = Path("artifacts"),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train XGBoost with stratified CV, returning OOF and test predictions.
    """
    if fold_cfg is None:
        fold_cfg = FoldConfig()
    if model_params is None:
        model_params = {
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_jobs": -1,
        }

    artifacts_dir = Path(artifacts_dir)
    dirs = _ensure_dirs(artifacts_dir)

    train_with_folds = make_stratified_folds(train_df, target_col=target_col, cfg=fold_cfg)
    fold_indices = get_fold_indices(train_with_folds, n_splits=fold_cfg.n_splits)

    X = train_with_folds[features].values
    y = train_with_folds[target_col].values
    X_test = test_df[features].values

    oof_pred = np.zeros(len(train_with_folds), dtype=float)
    test_pred_folds = np.zeros((fold_cfg.n_splits, len(test_df)), dtype=float)
    fold_scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(fold_indices):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        clf = XGBClassifier(**model_params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        valid_pred = clf.predict_proba(X_valid)[:, 1]
        oof_pred[valid_idx] = valid_pred
        fold_auc = roc_auc(y_valid, valid_pred)
        fold_scores.append(fold_auc)

        test_pred_folds[fold] = clf.predict_proba(X_test)[:, 1]

        model_path = dirs["models"] / f"{model_version}_fold{fold}.pkl"
        dump(clf, model_path)

    test_pred = test_pred_folds.mean(axis=0)

    oof_df = train_with_folds[["SK_ID_CURR", target_col]].copy()
    oof_df[f"pred_{model_version}"] = oof_pred
    oof_path = dirs["oof"] / f"oof_{model_version}.csv"
    oof_df.to_csv(oof_path, index=False)

    test_pred_df = pd.DataFrame(
        {
            "SK_ID_CURR": test_df["SK_ID_CURR"],
            f"pred_{model_version}": test_pred,
        }
    )
    test_pred_path = dirs["preds"] / f"pred_{model_version}_test.csv"
    test_pred_df.to_csv(test_pred_path, index=False)

    metrics = {
        "fold_auc": fold_scores,
        **fold_metrics_summary(fold_scores),
    }
    metrics_path = dirs["metrics"] / f"run_{model_version}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return oof_pred, test_pred, metrics

