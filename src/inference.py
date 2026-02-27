import argparse
from pathlib import Path

from src.models.stacking import apply_stacking_to_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for Home Credit models")
    parser.add_argument(
        "--stacker",
        choices=["bayes", "logreg"],
        default="bayes",
        help="Stacking meta-model type: bayes (BayesianRidge) or logreg (LogisticRegression)",
    )
    args = parser.parse_args()

    preds_dir = ARTIFACTS_DIR / "predictions"
    
    # Set model path based on stacker choice
    model_version = "stack_ridge_v1" if args.stacker == "bayes" else "stack_logreg_v1"
    model_path = ARTIFACTS_DIR / "models" / f"{model_version}.pkl"

    # Exclude legacy MLP test predictions
    test_pred_paths = sorted(
        p for p in preds_dir.glob("pred_*_test.csv") if "mlp" not in p.name.lower()
    )

    # Final submission in project root, matching expected format
    output_path = PROJECT_ROOT / "data" / "submissions" / "submission.csv"

    apply_stacking_to_test(
        test_pred_paths=test_pred_paths,
        stacking_model_path=model_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

