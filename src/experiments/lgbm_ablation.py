import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.features import build_features
from src.data.load_data import load_train_test
from src.models.lgbm import train_lgbm_cv
from src.utility.logger import get_logger, setup_logging
from src.validation.cv import FoldConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "ablation"

def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_experiment(
    train_df, 
    test_df, 
    features, 
    categorical_features, 
    run_name, 
    lgb_cfg, 
    logger
):
    """Run a single LightGBM experiment and return the mean AUC."""
    fold_cfg = FoldConfig(**lgb_cfg["folds"])
    
    # We suppress logs for individual folds to avoid clutter
    oof_pred, test_pred, metrics = train_lgbm_cv(
        train_df,
        test_df,
        features=features,
        target_col="TARGET",
        categorical_features=categorical_features,
        fold_cfg=fold_cfg,
        model_params=lgb_cfg["params"],
        model_version=run_name,
        artifacts_dir=ARTIFACTS_DIR,
    )
    
    return metrics.get("mean_auc", 0.0), metrics.get("feature_importances_path")

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Feature-Out Ablation Study for LightGBM")
    parser.add_argument("--subsample", type=float, default=1.0, help="Fraction of training data to use (for speed)")
    parser.add_argument("--n-folds", type=int, default=None, help="Override number of folds (e.g. 2 for speed)")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Override n_estimators (e.g. 500 for speed)")
    parser.add_argument("--skip-zero-gain", action="store_true", help="Skip ablating features that have 0 gain in baseline")
    parser.add_argument("--resume", action="store_true", help="Resume from previous ablation results")
    args = parser.parse_args()

    # Setup Logging
    logs_dir = ARTIFACTS_DIR / "logs"
    setup_logging(log_dir=str(logs_dir), experiment_name="lgbm_ablation")
    logger = get_logger(__name__)

    logger.info("Initializing LOFO Feature Ablation Study for LightGBM...")
    
    # Load and subset data
    train_raw, test_raw = load_train_test(PROJECT_ROOT)
    if args.subsample < 1.0:
        logger.info(f"Subsampling training data to {args.subsample * 100}%")
        train_raw = train_raw.sample(frac=args.subsample, random_state=42).reset_index(drop=True)
    
    # Build features
    logger.info("Building features...")
    train_features_df, feature_cols, cat_features = build_features(train_raw)
    test_features_df, _, _ = build_features(test_raw)
    train_features_df["TARGET"] = train_raw["TARGET"].values
    
    logger.info(f"Total features baseline: {len(feature_cols)}")

    # Load and override config
    lgb_cfg = load_yaml_config(CONFIG_DIR / "config_lgbm_tuned.yaml")
    if args.n_folds is not None:
        lgb_cfg["folds"]["n_splits"] = args.n_folds
    if args.n_estimators is not None:
        lgb_cfg["params"]["n_estimators"] = args.n_estimators
    
    logger.info(f"Using CV config: {lgb_cfg['folds']}")
    logger.info(f"Using estimator count: {lgb_cfg['params']['n_estimators']}")

    results_path = ARTIFACTS_DIR / "ablation_results.csv"
    results = []
    
    if args.resume and results_path.exists():
        existing_results = pd.read_csv(results_path)
        results = existing_results.to_dict('records')
        logger.info(f"Resumed from {len(results)} previously completed runs.")
        completed_features = set([r["ablated_feature"] for r in results if r["ablated_feature"] != "baseline"])
    else:
        completed_features = set()
        # RUN BASELINE
        logger.info("--- RUNNING BASELINE ---")
        t0 = time.time()
        baseline_auc, fi_path = run_experiment(
            train_features_df, test_features_df, feature_cols, cat_features, "baseline", lgb_cfg, logger
        )
        t_elapsed = time.time() - t0
        logger.info(f"Baseline AUC: {baseline_auc:.5f} (took {t_elapsed:.1f}s)")
        results.append({
            "ablated_feature": "baseline",
            "auc": baseline_auc,
            "auc_diff": 0.0,
            "time_seconds": t_elapsed
        })
        pd.DataFrame(results).to_csv(results_path, index=False)

    # Get Baseline AUC
    baseline_auc = next(r["auc"] for r in results if r["ablated_feature"] == "baseline")
    
    # Check baseline feature importances if skipping 0 gain
    features_to_ablate = feature_cols.copy()
    if args.skip_zero_gain:
        try:
            baseline_fi = pd.read_csv(ARTIFACTS_DIR / "metrics" / "feature_importance_baseline.csv")
            zero_gain_features = baseline_fi[baseline_fi["mean_gain"] == 0.0]["feature"].tolist()
            features_to_ablate = [f for f in features_to_ablate if f not in zero_gain_features]
            logger.info(f"Skipping {len(zero_gain_features)} features with 0 gain in baseline. Remaining to ablate: {len(features_to_ablate)}")
        except Exception as e:
            logger.warning(f"Could not load baseline feature importances for zero-gain filtering: {e}")

    # Remove already completed features
    features_to_ablate = [f for f in features_to_ablate if f not in completed_features]
    logger.info(f"Features pending ablation: {len(features_to_ablate)}")

    # RUN LOFO
    for i, feature in enumerate(features_to_ablate):
        logger.info(f"--- Ablating [{i+1}/{len(features_to_ablate)}]: {feature} ---")
        
        ablated_features = [f for f in feature_cols if f != feature]
        ablated_cats = [f for f in cat_features if f != feature]
        
        t0 = time.time()
        auc, _ = run_experiment(
            train_features_df, test_features_df, ablated_features, ablated_cats, f"ablation_{feature}", lgb_cfg, logger
        )
        t_elapsed = time.time() - t0
        
        auc_diff = auc - baseline_auc
        # A negative difference means AUC went DOWN when the feature was removed (Feature is GOOD)
        # A positive difference means AUC went UP when the feature was removed (Feature is NOISY / BAD)
        
        logger.info(f"Feature '{feature}' ablated -> AUC={auc:.5f} (Diff: {auc_diff:+.5f}, Time: {t_elapsed:.1f}s)")
        
        results.append({
            "ablated_feature": feature,
            "auc": auc,
            "auc_diff": auc_diff,
            "time_seconds": t_elapsed
        })
        
        # Save incrementally
        results_df = pd.DataFrame(results)
        # Sort by auc_diff descending (highest diff = most helpful to remove)
        results_df.sort_values(by="auc_diff", ascending=False, inplace=True)
        results_df.to_csv(results_path, index=False)

    logger.info(f"Ablation sweep complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()
