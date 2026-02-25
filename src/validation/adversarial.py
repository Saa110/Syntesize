from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.features import build_features
from src.data.load_data import load_train_test
from src.validation.metrics import roc_auc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_adversarial_validation() -> Tuple[float, pd.Series]:
    """
    Train a quick model to distinguish train vs. test rows (IS_TEST)
    and return adversarial AUC and feature importances.
    """
    import lightgbm as lgb

    train_df, test_df = load_train_test(PROJECT_ROOT)

    X_train, feature_cols, _ = build_features(train_df)
    X_test, _, _ = build_features(test_df)

    X_train["IS_TEST"] = 0
    X_test["IS_TEST"] = 1

    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_adv = combined["IS_TEST"].values
    X_adv = combined[feature_cols].values

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=-1,
    )
    clf.fit(X_adv, y_adv)

    preds = clf.predict_proba(X_adv)[:, 1]
    auc = roc_auc(y_adv, preds)

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return auc, importances


if __name__ == "__main__":
    auc, imps = run_adversarial_validation()
    print(f"Adversarial AUC (train vs test): {auc:.4f}")
    print("Top 20 drift features:")
    print(imps.head(20))

