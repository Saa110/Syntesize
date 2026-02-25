import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

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

# Define mappings from substrings/prefixes to Groups
FEATURE_GROUPS_RULES = {
    "EXT_SOURCES": ["EXT_SOURCE", "EXT_1_", "EXT_2_", "EXT_3_", "EXT_SOURCES_"],
    "CREDIT_BUREAU": ["AMT_REQ_CREDIT_BUREAU", "TOTAL_BUREAU_ENQ", "RECENT_BUREAU_ENQ", "OLDER_BUREAU_ENQ"],
    "BUILDING_INFO": [
        "_AVG", "_MODE", "_MEDI", "FONDKAPREMONT_", "HOUSETYPE_", "WALLSMATERIAL_", "EMERGENCYSTATE_", 
        "BUILDING_MISSING_COUNT", "APARTMENTS_", "BASEMENTAREA_", "YEARS_BEGINEXPLUATATION_",
        "YEARS_BUILD_", "COMMONAREA_", "ELEVATORS_", "ENTRANCES_", "FLOORSMAX_", "FLOORSMIN_", 
        "LANDAREA_", "LIVINGAPARTMENTS_", "LIVINGAREA_", "NONLIVINGAPARTMENTS_", "NONLIVINGAREA_", "TOTALAREA_"
    ],
    "CONTACT_INFO": ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "CONTACT_FLAGS_SUM", "DAYS_LAST_PHONE_CHANGE", "PHONE_TO_"],
    "DEMOGRAPHICS": ["DAYS_BIRTH", "AGE_", "CNT_FAM_MEMBERS", "CNT_CHILDREN", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE", "ID_TO_BIRTH_RATIO", "DAYS_ID_PUBLISH"],
    "EMPLOYMENT": ["DAYS_EMPLOYED", "EMPLOYED_", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"],
    "APP_BEHAVIOR": ["HOUR_APPR_PROCESS_START", "WEEKDAY_APPR_PROCESS_START"],
    "FINANCIALS": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "LOG_AMT_", "CREDIT_TERM", "PAYMENT_TO_INCOME", "CREDIT_TO_INCOME", "GOODS_TO_CREDIT", "ANNUITY_TO_CREDIT", "GOODS_TO_INCOME", "INCOME_PER_PERSON", "INCOME_TO_EMPLOYED", "INCOME_MONTHLY_MINUS_ANNUITY", "INCOME_BIN"],
    "SOCIAL_CIRCLE": ["_SOCIAL_CIRCLE", "SOCIAL_DEFAULT_RATIO"],
    "DOCUMENTS": ["FLAG_DOCUMENT", "DOCUMENT_COUNT", "DOCUMENT_KURT"],
    "CAR_INFO": ["OWN_CAR_AGE", "FLAG_OWN_CAR", "CAR_TO_"],
    "REGION_RATING": ["REGION_RATING_CLIENT", "REG_CITY_", "REG_REGION_", "LIVE_CITY_", "LIVE_REGION_"],
    "MISSINGNESS": ["TOTAL_MISSING"],
    "RISK_GROUPANIZER": ["_high_risk", "_medium_risk", "_low_risk"],
}

def get_group_for_feature(feature_name: str) -> str:
    for group, rules in FEATURE_GROUPS_RULES.items():
        if any(rule in feature_name for rule in rules):
            return group
    return "OTHER"

def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_experiment(train_df, test_df, features, categorical_features, run_name, lgb_cfg, logger):
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
    parser = argparse.ArgumentParser(description="Leave-One-Group-Out (LOGO) Ablation Study for LightGBM")
    parser.add_argument("--subsample", type=float, default=1.0, help="Fraction of training data to use (for speed)")
    parser.add_argument("--n-folds", type=int, default=None, help="Override number of folds (e.g. 2 for speed)")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Override n_estimators (e.g. 500 for speed)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous ablation results")
    args = parser.parse_args()

    # Setup Logging
    logs_dir = ARTIFACTS_DIR / "logs"
    setup_logging(log_dir=str(logs_dir), experiment_name="lgbm_logo_ablation")
    logger = get_logger(__name__)

    logger.info("Initializing LOGO (Leave-One-Group-Out) Ablation Study for LightGBM...")
    
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

    # Group Features
    feature_to_group = {f: get_group_for_feature(f) for f in feature_cols}
    group_to_features = defaultdict(list)
    for f, g in feature_to_group.items():
        group_to_features[g].append(f)
        
    logger.info("Feature groupings:")
    for g, feats in group_to_features.items():
        logger.info(f"  - {g}: {len(feats)} features")
        
    # Load and override config
    lgb_cfg = load_yaml_config(CONFIG_DIR / "config_lgbm_tuned.yaml")
    if args.n_folds is not None:
        lgb_cfg["folds"]["n_splits"] = args.n_folds
    if args.n_estimators is not None:
        lgb_cfg["params"]["n_estimators"] = args.n_estimators
    
    logger.info(f"Using CV config: {lgb_cfg['folds']}")
    logger.info(f"Using estimator count: {lgb_cfg['params']['n_estimators']}")

    results_path = ARTIFACTS_DIR / "logo_ablation_results.csv"
    results = []
    
    if args.resume and results_path.exists():
        existing_results = pd.read_csv(results_path)
        results = existing_results.to_dict('records')
        logger.info(f"Resumed from {len(results)} previously completed runs.")
        completed_groups = set([r["ablated_group"] for r in results if r["ablated_group"] != "baseline"])
    else:
        completed_groups = set()
        # RUN BASELINE
        logger.info("--- RUNNING BASELINE ---")
        t0 = time.time()
        baseline_auc, fi_path = run_experiment(
            train_features_df, test_features_df, feature_cols, cat_features, "logo_baseline", lgb_cfg, logger
        )
        t_elapsed = time.time() - t0
        logger.info(f"Baseline AUC: {baseline_auc:.5f} (took {t_elapsed:.1f}s)")
        results.append({
            "ablated_group": "baseline",
            "features_removed": 0,
            "auc": baseline_auc,
            "auc_diff": 0.0,
            "time_seconds": t_elapsed
        })
        pd.DataFrame(results).to_csv(results_path, index=False)

    # Get Baseline AUC
    baseline_auc = next(r["auc"] for r in results if r["ablated_group"] == "baseline")

    groups_to_ablate = [g for g in group_to_features.keys() if g not in completed_groups]
    logger.info(f"Groups pending ablation: {len(groups_to_ablate)}")

    # RUN LOGO
    for i, group in enumerate(groups_to_ablate):
        logger.info(f"--- Ablating Group [{i+1}/{len(groups_to_ablate)}]: {group} ({len(group_to_features[group])} features) ---")
        
        group_feats = set(group_to_features[group])
        ablated_features = [f for f in feature_cols if f not in group_feats]
        ablated_cats = [f for f in cat_features if f not in group_feats]
        
        t0 = time.time()
        auc, _ = run_experiment(
            train_features_df, test_features_df, ablated_features, ablated_cats, f"logo_ablation_{group}", lgb_cfg, logger
        )
        t_elapsed = time.time() - t0
        
        auc_diff = auc - baseline_auc
        # A negative difference means AUC went DOWN when the group was removed (Group is GOOD)
        # A positive difference means AUC went UP when the group was removed (Group is NOISY / BAD)
        
        logger.info(f"Group '{group}' ablated -> AUC={auc:.5f} (Diff: {auc_diff:+.5f}, Time: {t_elapsed:.1f}s)")
        
        results.append({
            "ablated_group": group,
            "features_removed": len(group_feats),
            "auc": auc,
            "auc_diff": auc_diff,
            "time_seconds": t_elapsed
        })
        
        # Save incrementally
        results_df = pd.DataFrame(results)
        # Sort by auc_diff descending (highest diff = most helpful to remove)
        results_df.sort_values(by="auc_diff", ascending=False, inplace=True)
        results_df.to_csv(results_path, index=False)

    logger.info(f"LOGO Ablation sweep complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()
