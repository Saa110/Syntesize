import pandas as pd
from pathlib import Path
import numpy as np

oof_dir = Path("artifacts/oof")
for path in sorted(oof_dir.glob("oof_*.csv")):
    if "mlp" in path.name.lower(): continue
    df = pd.read_csv(path)
    print(f"{path.name}: {df.columns.tolist()}")
