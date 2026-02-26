from __future__ import annotations

from pathlib import Path

from src.models.stacking import apply_stacking_to_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main() -> None:
    preds_dir = ARTIFACTS_DIR / "predictions"
    model_path = ARTIFACTS_DIR / "models" / "stack_logreg_v1.pkl"

    test_pred_paths = sorted(
        p for p in preds_dir.glob("pred_*_test.csv") if "mlp" not in p.name.lower()
    )

    # Final submission in project root, matching expected format
    output_path = PROJECT_ROOT / "submission.csv"

    apply_stacking_to_test(
        test_pred_paths=test_pred_paths,
        stacking_model_path=model_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

