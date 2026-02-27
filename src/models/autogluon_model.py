from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


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


def train_autogluon_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str = "TARGET",
    model_params: Dict | None = None,
    model_version: str = "autogluon_v1",
    artifacts_dir: Path | str = Path("artifacts"),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train AutoGluon, returning OOF and test predictions.
    AutoGluon handles its own internal CV if bagging is enabled.
    """
    if model_params is None:
        model_params = {
            "presets": "best_quality",
            "time_limit": 900,
        }

    artifacts_dir = Path(artifacts_dir)
    dirs = _ensure_dirs(artifacts_dir)
    model_path = dirs["models"] / model_version

    # AutoGluon needs the features + target in one dataframe
    train_data = train_df[features + [target_col]].copy()

    predictor = TabularPredictor(
        label=target_col,
        path=str(model_path),
        problem_type="binary",
        eval_metric="roc_auc",
    ).fit(
        train_data,
        **model_params
    )

    # Get OOF predictions (if bagging was enabled in presets/params)
    try:
        oof_pred = predictor.get_oof_pred_proba().iloc[:, 1].values
    except Exception:
        # Fallback if OOF is not available
        oof_pred = predictor.predict_proba(train_data).iloc[:, 1].values

    # Test predictions
    test_pred = predictor.predict_proba(test_df[features]).iloc[:, 1].values

    # Save OOF and test predictions
    oof_df = train_df[["SK_ID_CURR", target_col]].copy()
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

    # Save metrics
    leaderboard = predictor.leaderboard(train_data, silent=True)
    best_score = predictor.evaluate(train_data)["roc_auc"]
    
    metrics = {
        "mean_auc": best_score,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    metrics_path = dirs["metrics"] / f"run_{model_version}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return oof_pred, test_pred, metrics
