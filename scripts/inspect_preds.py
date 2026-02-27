import pandas as pd
from pathlib import Path
import numpy as np

preds_dir = Path("artifacts/predictions")
for path in sorted(preds_dir.glob("pred_*_test.csv")):
    if "mlp" in path.name.lower(): continue
    df = pd.read_csv(path)
    pred_col = [c for c in df.columns if c != "SK_ID_CURR"][0]
    print(f"{path.name} ({pred_col}):")
    print(df[pred_col].describe())
    print("NaNs:", df[pred_col].isna().sum())
    print("Infs:", np.isinf(df[pred_col]).sum())
    print()
