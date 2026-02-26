from __future__ import annotations

from pathlib import Path
import argparse
import time

import yaml
from sklearn.impute import SimpleImputer

from src.data.features import build_features
from src.data.load_data import load_train_test
from src.models.catboost import train_catboost_cv
from src.models.lgbm import train_lgbm_cv
from src.models.nn import train_mlp_cv
from src.models.stacking import load_oof_files, train_stacking_model
from src.models.xgb import train_xgb_cv
from src.utility.logger import get_logger, setup_logging
from src.validation.cv import FoldConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Home Credit models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "lgbm", "xgb", "cat", "mlp", "stack"],
        help="Which components to train: all, or any of lgbm xgb cat mlp stack",
    )
    parser.add_argument(
        "--stacker",
        choices=["bayes", "logreg"],
        default="logreg",
        help="Stacking meta-model type: bayes (BayesianRidge) or logreg (LogisticRegression)",
    )
    args = parser.parse_args()
    selected = set(args.models)
    if "all" in selected:
        # Default to tree-based models plus stacking; MLP is opt-in only.
        selected = {"lgbm", "xgb", "cat", "stack"}

    start_time = time.time()

    # Initialize centralized logging (console + file)
    logs_dir = PROJECT_ROOT / "artifacts" / "logs"
    setup_logging(log_dir=str(logs_dir), experiment_name="train")
    logger = get_logger(__name__)
    logger.info("Selected components to train: %s", ", ".join(sorted(selected)))

    logger.info("Loading data...")
    train_df, test_df = load_train_test(PROJECT_ROOT)

    # Class imbalance stats (for potential loss shaping)
    pos = int((train_df["TARGET"] == 1).sum())
    neg = int((train_df["TARGET"] == 0).sum())
    if pos > 0:
        scale_pos_weight_est = neg / pos
        logger.info(
            "Class counts -> pos=%d, neg=%d, estimated scale_pos_weight≈%.2f",
            pos,
            neg,
            scale_pos_weight_est,
        )

    logger.info("Building features...")
    train_features_df, feature_cols, categorical_features = build_features(train_df)
    test_features_df, _, _ = build_features(test_df)

    # Ensure TARGET present in training features
    train_features_df["TARGET"] = train_df["TARGET"].values

    # Global numeric imputation (median) so all models see finite numeric values.
    # Exclude categorical/bin features from numeric imputation to keep them as integers
    # for CatBoost and other tree-based models.
    # logger.info("Imputing missing numeric values with column medians (fitted on train)...")
    # imputer = SimpleImputer(strategy="median")
    # base_numeric_cols = (
    #     train_features_df[feature_cols]
    #     .select_dtypes(include=["number", "bool"])
    #     .columns.tolist()
    # )
    # numeric_cols = [c for c in base_numeric_cols if c not in categorical_features]
    # if numeric_cols:
    #     train_features_df[numeric_cols] = imputer.fit_transform(train_features_df[numeric_cols])
    #     test_features_df[numeric_cols] = imputer.transform(test_features_df[numeric_cols])

    logger.info("Features ready: %d columns", len(feature_cols))

    # LightGBM
    if "lgbm" in selected:
        t0 = time.time()
        logger.info("Training LightGBM...")
        lgb_cfg = load_yaml_config(CONFIG_DIR / "config_lgbm.yaml")
        lgb_fold_cfg = FoldConfig(**lgb_cfg["folds"])
        _, _, lgb_metrics = train_lgbm_cv(
            train_features_df,
            test_features_df,
            features=feature_cols,
            target_col="TARGET",
            categorical_features=categorical_features,
            fold_cfg=lgb_fold_cfg,
            model_params=lgb_cfg["params"],
            model_version=lgb_cfg["model_version"],
            artifacts_dir=ARTIFACTS_DIR,
        )
        logger.info(
            "LightGBM done. CV AUC=%.5f (took %.1fs)",
            lgb_metrics.get("mean_auc", float("nan")),
            time.time() - t0,
        )

    # XGBoost
    if "xgb" in selected:
        t0 = time.time()
        logger.info("Training XGBoost...")
        xgb_cfg = load_yaml_config(CONFIG_DIR / "config_xgb.yaml")
        xgb_fold_cfg = FoldConfig(**xgb_cfg["folds"])
        _, _, xgb_metrics = train_xgb_cv(
            train_features_df,
            test_features_df,
            features=feature_cols,
            target_col="TARGET",
            fold_cfg=xgb_fold_cfg,
            model_params=xgb_cfg["params"],
            model_version=xgb_cfg["model_version"],
            artifacts_dir=ARTIFACTS_DIR,
        )
        logger.info(
            "XGBoost done. CV AUC=%.5f (took %.1fs)",
            xgb_metrics.get("mean_auc", float("nan")),
            time.time() - t0,
        )

    # CatBoost
    if "cat" in selected:
        t0 = time.time()
        logger.info("Training CatBoost...")
        cat_cfg = load_yaml_config(CONFIG_DIR / "config_catboost.yaml")
        cat_fold_cfg = FoldConfig(**cat_cfg["folds"])
        _, _, cat_metrics = train_catboost_cv(
            train_features_df,
            test_features_df,
            features=feature_cols,
            target_col="TARGET",
            categorical_features=categorical_features,
            fold_cfg=cat_fold_cfg,
            model_params=cat_cfg["params"],
            model_version=cat_cfg["model_version"],
            artifacts_dir=ARTIFACTS_DIR,
        )
        logger.info(
            "CatBoost done. CV AUC=%.5f (took %.1fs)",
            cat_metrics.get("mean_auc", float("nan")),
            time.time() - t0,
        )

    # Optional MLP (disabled by default; kept for experimentation)
    if "mlp" in selected:
        logger.warning(
            "MLP training is currently disabled. Remove this guard if you want to re-enable it."
        )

    # Stacking
    if "stack" in selected:
        logger.info("Training stacking model...")
        oof_dir = ARTIFACTS_DIR / "oof"
        # Exclude any legacy MLP OOF files from stacking by default.
        oof_paths = sorted(
            p for p in oof_dir.glob("oof_*.csv") if "mlp" not in p.name.lower()
        )
        if not oof_paths:
            logger.warning(
                "No OOF files found in %s; skipping stacking. Train base models first.",
                oof_dir,
            )
        else:
            oof_df = load_oof_files(oof_paths)
            _, stack_metrics = train_stacking_model(
                oof_df=oof_df,
                target_col="TARGET",
                model_version="stack_ridge_v1" if args.stacker == "bayes" else "stack_logreg_v1",
                artifacts_dir=ARTIFACTS_DIR,
                stacker_type=args.stacker,
            )
            logger.info(
                "Stacking done. CV AUC=%.5f",
                stack_metrics.get("mean_auc", float("nan")),
            )

    logger.info(
        "All training finished in %.1f minutes.",
        (time.time() - start_time) / 60.0,
    )


if __name__ == "__main__":
    main()

