# save as: src/feature_importance_summary.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

from src.train import PROJECT_ROOT, CONFIG_DIR, ARTIFACTS_DIR, load_yaml_config
from src.data.load_data import load_train_test
from src.data.features import build_features


def load_fold_importances(model_name: str, config_file: str) -> np.ndarray | None:
    cfg = load_yaml_config(CONFIG_DIR / config_file)
    model_version: str = cfg["model_version"]
    n_splits: int = cfg["folds"]["n_splits"]

    models_dir = ARTIFACTS_DIR / "models"
    imps: list[np.ndarray] = []

    for fold in range(n_splits):
        model_path = models_dir / f"{model_version}_fold{fold}.pkl"
        if not model_path.exists():
            continue
        clf = load(model_path)

        if model_name == "cat":
            fi = np.asarray(clf.get_feature_importance())  # CatBoost
        else:
            fi = np.asarray(clf.feature_importances_)      # LGBM / XGB

        imps.append(fi)

    if not imps:
        return None

    return np.mean(imps, axis=0)


def main() -> None:
    train_df, _ = load_train_test(PROJECT_ROOT)
    train_features_df, feature_cols, _ = build_features(train_df)

    # Load per-model mean importances
    lgbm_imp = load_fold_importances("lgbm", "config_lgbm.yaml")
    xgb_imp = load_fold_importances("xgb", "config_xgb.yaml")
    cat_imp = load_fold_importances("cat", "config_catboost.yaml")

    df = pd.DataFrame({"feature": feature_cols})

    model_cols = []
    if lgbm_imp is not None:
        df["imp_lgbm"] = lgbm_imp
        model_cols.append("imp_lgbm")
    if xgb_imp is not None:
        df["imp_xgb"] = xgb_imp
        model_cols.append("imp_xgb")
    if cat_imp is not None:
        df["imp_cat"] = cat_imp
        model_cols.append("imp_cat")

    if not model_cols:
        raise SystemExit("No model importances found. Train models first.")

    # Normalize each model’s importances so they’re comparable
    for col in model_cols:
        s = df[col].sum()
        if s > 0:
            df[col] = df[col] / s

    df["importance_mean"] = df[model_cols].mean(axis=1)

    top20 = df.sort_values("importance_mean", ascending=False).head(20)
    print("Top 20 features across models:")
    print(top20[["feature", "importance_mean"] + model_cols].to_string(index=False))


if __name__ == "__main__":
    main()