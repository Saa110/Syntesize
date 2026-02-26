from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import BayesianRidge, LogisticRegression

from src.validation.cv import FoldConfig, make_stratified_folds, get_fold_indices
from src.validation.metrics import roc_auc, fold_metrics_summary


def load_oof_files(oof_paths: Iterable[Path]) -> pd.DataFrame:
    """
    Merge multiple OOF files on SK_ID_CURR and TARGET.
    Each OOF file is expected to contain SK_ID_CURR, TARGET, and one prediction column.
    """
    oof_paths = list(oof_paths)
    if not oof_paths:
        raise ValueError("No OOF paths provided for stacking.")

    merged: pd.DataFrame | None = None
    for path in oof_paths:
        df = pd.read_csv(path)
        if merged is None:
            merged = df
        else:
            pred_cols = [c for c in df.columns if c not in {"SK_ID_CURR", "TARGET"}]
            merged = merged.merge(df[["SK_ID_CURR"] + pred_cols], on="SK_ID_CURR", how="inner")
    if merged is None:
        raise ValueError("Failed to load any OOF data.")
    return merged


def _make_meta_features(df_preds: pd.DataFrame, base_pred_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    X_list: List[pd.Series] = []
    meta_feature_names: List[str] = []
    for col in base_pred_cols:
        p = df_preds[col].clip(1e-6, 1 - 1e-6).reset_index(drop=True)
        X_list.append(p)
        meta_feature_names.append(col)

        # logits
        logit = (p / (1 - p)).apply(np.log)
        logit_name = f"{col}_logit"
        X_list.append(logit)
        meta_feature_names.append(logit_name)

        # ranks
        rank = p.rank(method="average") / len(p)
        rank_name = f"{col}_rank"
        X_list.append(rank)
        meta_feature_names.append(rank_name)

    if not X_list:
        return np.array([]), []

    X_df = pd.concat(X_list, axis=1)
    X_df.columns = meta_feature_names
    return X_df.values, meta_feature_names


def train_stacking_model(
    oof_df: pd.DataFrame,
    target_col: str = "TARGET",
    model_version: str = "stack_ridge_v1",
    artifacts_dir: Path | str = Path("artifacts"),
    fold_cfg: FoldConfig | None = None,
    stacker_type: str = "bayes",
) -> Tuple[np.ndarray, dict]:
    """
    Train a meta-model on OOF predictions with CV on the OOF table.
    stacker_type: 'bayes' (BayesianRidge) or 'logreg' (LogisticRegression).
    """
    if fold_cfg is None:
        fold_cfg = FoldConfig()

    artifacts_dir = Path(artifacts_dir)
    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_pred_cols = [c for c in oof_df.columns if c not in {"SK_ID_CURR", target_col}]

    y = oof_df[target_col].values

    df_with_folds = make_stratified_folds(oof_df[["SK_ID_CURR", target_col]].copy(), target_col=target_col, cfg=fold_cfg)
    fold_indices = get_fold_indices(df_with_folds, n_splits=fold_cfg.n_splits)

    oof_meta = np.zeros(len(oof_df), dtype=float)
    fold_scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(fold_indices):
        X_train, _ = _make_meta_features(oof_df.iloc[train_idx], base_pred_cols)
        y_train = y[train_idx]
        X_valid, _ = _make_meta_features(oof_df.iloc[valid_idx], base_pred_cols)
        y_valid = y[valid_idx]

        if stacker_type == "logreg":
            model = LogisticRegression(max_iter=1000)
        else:
            model = BayesianRidge()

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            valid_pred = model.predict_proba(X_valid)[:, 1]
        else:
            valid_pred = model.predict(X_valid)
        oof_meta[valid_idx] = valid_pred

        score = roc_auc(y_valid, valid_pred)
        fold_scores.append(score)

    # Fit final model on all OOF data
    X, meta_feature_names = _make_meta_features(oof_df, base_pred_cols)
    if stacker_type == "logreg":
        final_model = LogisticRegression(max_iter=1000)
    else:
        final_model = BayesianRidge()
    final_model.fit(X, y)

    dump(
        {"model": final_model, "features": meta_feature_names},
        models_dir / f"{model_version}.pkl",
    )

    metrics = {
        "fold_auc": fold_scores,
        **fold_metrics_summary(fold_scores),
    }
    with (metrics_dir / f"run_{model_version}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return oof_meta, metrics


def apply_stacking_to_test(
    test_pred_paths: Iterable[Path],
    stacking_model_path: Path,
    output_path: Path,
) -> None:
    """
    Apply trained stacking model to per-model test prediction files and save final predictions.
    """
    test_pred_paths = list(test_pred_paths)
    if not test_pred_paths:
        raise ValueError("No test prediction paths provided for stacking.")

    merged: pd.DataFrame | None = None
    for path in test_pred_paths:
        df = pd.read_csv(path)
        if merged is None:
            merged = df
        else:
            pred_cols = [c for c in df.columns if c != "SK_ID_CURR"]
            merged = merged.merge(df[["SK_ID_CURR"] + pred_cols], on="SK_ID_CURR", how="inner")
    if merged is None:
        raise ValueError("Failed to load any test prediction data.")

    bundle = load(stacking_model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    base_pred_cols = [c for c in merged.columns if c != "SK_ID_CURR"]
    X_test, generated_features = _make_meta_features(merged, base_pred_cols)
    
    # Ensure generated features match model features
    if generated_features != feature_cols:
        raise ValueError(f"Generated meta-features mismatch with trained stacker model features.\nExpected: {feature_cols}\nGot: {generated_features}")

    if hasattr(model, "predict_proba"):
        merged["TARGET"] = model.predict_proba(X_test)[:, 1]
    else:
        merged["TARGET"] = model.predict(X_test)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["SK_ID_CURR", "TARGET"]].to_csv(output_path, index=False)

