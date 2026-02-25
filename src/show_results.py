from __future__ import annotations

from pathlib import Path
import json

import yaml
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    metrics_dir = artifacts_dir / "metrics"
    config_dir = PROJECT_ROOT / "config"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in sorted(metrics_dir.glob("run_*.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        run_name = metrics_path.stem.replace("run_", "")

        # Try to infer related config
        cfg_path = None
        if "lgbm" in run_name:
            cfg_path = config_dir / "config_lgbm.yaml"
        elif "xgb" in run_name:
            cfg_path = config_dir / "config_xgb.yaml"
        elif "cat" in run_name:
            cfg_path = config_dir / "config_catboost.yaml"

        cfg = {}
        if cfg_path is not None and cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        rows.append(
            {
                "run": run_name,
                "mean_auc": metrics.get("mean_auc"),
                "std_auc": metrics.get("std_auc"),
                "min_auc": metrics.get("min_auc"),
                "max_auc": metrics.get("max_auc"),
                "config_file": str(cfg_path) if cfg_path else "",
                "params": cfg.get("params", {}),
            }
        )

    df = pd.DataFrame(rows)
    df.to_json(results_dir / "summary.json", orient="records", indent=2)
    if not df.empty:
        df_simple = df[["run", "mean_auc", "std_auc", "min_auc", "max_auc", "config_file"]]
        df_simple.to_csv(results_dir / "summary.csv", index=False)
        print(df_simple)
    else:
        print("No metrics found in artifacts/metrics.")


if __name__ == "__main__":
    main()

