from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class FoldConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42


def make_stratified_folds(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    cfg: FoldConfig | None = None,
) -> pd.DataFrame:
    """
    Add a 'fold' column to the DataFrame using stratified K-Fold on the target.
    """
    if cfg is None:
        cfg = FoldConfig()

    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.random_state,
    )

    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df[target_col])):
        df.loc[val_idx, "fold"] = fold_idx

    return df


def get_fold_indices(df_with_folds: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    From a DataFrame with a 'fold' column, return train/valid indices per fold.
    """
    if "fold" not in df_with_folds.columns:
        raise ValueError("DataFrame must contain a 'fold' column.")

    indices: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        train_idx = np.where(df_with_folds["fold"] != fold)[0]
        valid_idx = np.where(df_with_folds["fold"] == fold)[0]
        indices.append((train_idx, valid_idx))
    return indices


def summarize_fold_scores(scores: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(scores), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

