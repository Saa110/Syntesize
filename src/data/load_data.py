from pathlib import Path
from typing import Tuple

import pandas as pd


def load_train_test(
    data_dir: Path | str = Path("data/raw"),
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data with reasonable default dtypes.
    """
    data_path = Path(data_dir)

    train_path = data_path / train_filename
    test_path = data_path / test_filename

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test

