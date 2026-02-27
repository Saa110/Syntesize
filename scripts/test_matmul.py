import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from src.models.stacking import _prepare_meta_features
import warnings

def check_features():
    preds_dir = Path("artifacts/predictions")
    test_pred_paths = sorted(
        p for p in preds_dir.glob("pred_*_test.csv") if "mlp" not in p.name.lower()
    )
    
    merged = None
    for path in test_pred_paths:
        df = pd.read_csv(path)
        if merged is None:
            merged = df
        else:
            pred_cols = [c for c in df.columns if c != "SK_ID_CURR"]
            merged = merged.merge(df[["SK_ID_CURR"] + pred_cols], on="SK_ID_CURR", how="inner")
            
    stacking_model_path = Path("artifacts/models/stack_logreg_v1.pkl")
    bundle = load(stacking_model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    base_pred_cols = [c for c in merged.columns if c != "SK_ID_CURR"]
    X_df = _prepare_meta_features(merged, base_pred_cols)
    X_test = X_df[feature_cols].values
    
    warnings.simplefilter("always")
    print("Manual matmul:")
    res = X_test @ model.coef_.T + model.intercept_
    print("res shape:", res.shape, "max:", np.max(res), "min:", np.min(res))
    
    # Try predict_proba
    print("Calling predict_proba...")
    probs = model.predict_proba(X_test)
    print("probs max:", np.max(probs), "min:", np.min(probs))

if __name__ == "__main__":
    check_features()
