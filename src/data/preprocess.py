from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


DAYS_EMPLOYED_ANOMALOUS_VALUE = 365243


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace anomalous DAYS_EMPLOYED value with NaN and add a flag.
    """
    if "DAYS_EMPLOYED" not in df.columns:
        return df

    col = "DAYS_EMPLOYED"
    anomaly_flag = f"{col}_ANOMALOUS"

    df[anomaly_flag] = (df[col] == DAYS_EMPLOYED_ANOMALOUS_VALUE).astype(np.int8)
    df.loc[df[col] == DAYS_EMPLOYED_ANOMALOUS_VALUE, col] = np.nan

    return df


def get_categorical_columns(df: pd.DataFrame) -> Iterable[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def fit_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    Fit a LabelEncoder per categorical column.
    """
    encoders: Dict[str, LabelEncoder] = {}
    for col in get_categorical_columns(df):
        le = LabelEncoder()
        # Fill NaNs with a dedicated token for encoding
        series = df[col].astype(str).fillna("__MISSING__")
        le.fit(series)
        encoders[col] = le
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Apply previously-fitted LabelEncoders to a DataFrame.
    Unknown categories are mapped to -1.
    """
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        series = df[col].astype(str).fillna("__MISSING__")
        # Map using existing classes; unknown -> -1
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        df[col] = series.map(mapping).fillna(-1).astype("int32")
    return df


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic, model-agnostic preprocessing steps.
    """
    df = df.copy()
    df = fix_days_employed(df)
    return df

