from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.impute import SimpleImputer

from src.data.features import build_features
from src.data.load_data import load_train_test


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump the final engineered feature matrices used for training."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing train.csv and test.csv (default: .)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/features",
        help="Output directory (default: artifacts/features)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet). If parquet deps are missing, use csv.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print shape/columns and the first few rows instead of writing files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_test(data_dir)

    train_features_df, feature_cols, categorical_features = build_features(train_df)
    test_features_df, _, _ = build_features(test_df)

    # Match training pipeline: add TARGET, then median-impute numeric (excluding bin/cat cols)
    if "TARGET" in train_df.columns:
        train_features_df["TARGET"] = train_df["TARGET"].values

    imputer = SimpleImputer(strategy="median")
    base_numeric_cols = (
        train_features_df[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    )
    numeric_cols = [c for c in base_numeric_cols if c not in categorical_features]
    if numeric_cols:
        train_features_df[numeric_cols] = imputer.fit_transform(train_features_df[numeric_cols])
        test_features_df[numeric_cols] = imputer.transform(test_features_df[numeric_cols])

    if args.preview:
        print("TRAIN features shape:", train_features_df.shape)
        print("TEST  features shape:", test_features_df.shape)
        print("Number of feature columns:", len(feature_cols))
        print("First 25 columns:", feature_cols[:25])
        print("\nTrain head():")
        print(train_features_df.head())
        return

    if args.format == "parquet":
        train_out = out_dir / "train_features.parquet"
        test_out = out_dir / "test_features.parquet"
        _ensure_parent_dir(train_out)
        _ensure_parent_dir(test_out)
        try:
            train_features_df.to_parquet(train_out, index=False)
            test_features_df.to_parquet(test_out, index=False)
        except Exception as e:
            raise SystemExit(
                "Parquet export failed (likely missing 'pyarrow' or 'fastparquet'). "
                "Re-run with --format csv, or install pyarrow.\n"
                f"Original error: {e}"
            ) from e
    else:
        train_out = out_dir / "train_features.csv"
        test_out = out_dir / "test_features.csv"
        _ensure_parent_dir(train_out)
        _ensure_parent_dir(test_out)
        train_features_df.to_csv(train_out, index=False)
        test_features_df.to_csv(test_out, index=False)

    print(f"Wrote: {train_out}")
    print(f"Wrote: {test_out}")


if __name__ == "__main__":
    main()

