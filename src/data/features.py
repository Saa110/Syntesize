from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .preprocess import basic_preprocess


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace({0: np.nan})
    return result.astype("float32")


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core financial, temporal, and external-score-based features.
    """
    df = df.copy()

    # --- Core amount ratios ---
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["CREDIT_TERM"] = _safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["PAYMENT_TO_INCOME"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_TO_INCOME"] = _safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["GOODS_TO_CREDIT"] = _safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])

    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["ANNUITY_TO_CREDIT"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])

    if {"AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["GOODS_TO_INCOME"] = _safe_divide(df["AMT_GOODS_PRICE"], df["AMT_INCOME_TOTAL"])

    # --- Log-transformed amounts for heavy-tailed distributions ---
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df.columns:
            df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

    # --- External scores aggregations and transforms ---
    ext_sources = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_sources:
        df["EXT_SOURCE_MEAN"] = df[ext_sources].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_sources].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_sources].max(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_sources].std(axis=1)
        df["EXT_SOURCE_MEAN_SQ"] = df["EXT_SOURCE_MEAN"] ** 2

    # --- Temporal features based on DAYS_* ---
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["DAYS_EMPLOYED_PCT"] = _safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).astype("float32")
        df["AGE_YEARS_SQ"] = (df["AGE_YEARS"] ** 2).astype("float32")

    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).astype("float32")

    # --- Binned categorical features for CatBoost / trees ---
    try:
        if "AGE_YEARS" in df.columns:
            df["AGE_BIN"] = pd.cut(
                df["AGE_YEARS"],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=False,
                include_lowest=True,
            )

        if "LOG_AMT_INCOME_TOTAL" in df.columns:
            df["INCOME_BIN"] = pd.qcut(
                df["LOG_AMT_INCOME_TOTAL"],
                q=5,
                labels=False,
                duplicates="drop",
            )

        if "EXT_SOURCE_MEAN" in df.columns:
            df["EXT_SOURCE_MEAN_BIN"] = pd.qcut(
                df["EXT_SOURCE_MEAN"],
                q=5,
                labels=False,
                duplicates="drop",
            )
    except Exception:
        # If binning fails due to insufficient unique values, leave bins as NaN.
        pass

    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Full feature pipeline:
    - basic preprocessing
    - domain feature engineering

    Returns:
        features_df: DataFrame ready for modeling (TARGET excluded if present)
        feature_names: list of feature column names
        categorical_features: list of categorical feature names (tree models can use this)
    """
    df_proc = basic_preprocess(df)
    df_feat = add_domain_features(df_proc)

    # Do not include TARGET in the feature matrix
    target_cols = [c for c in ["TARGET"] if c in df_feat.columns]
    candidate_cols = [c for c in df_feat.columns if c not in target_cols]

    # Separate numeric and bin-like columns; encode bins as integer codes.
    bin_cols = [c for c in candidate_cols if c.endswith("_BIN")]

    numeric_cols = (
        df_feat[candidate_cols]
        .select_dtypes(include=["number", "bool"])
        .columns.tolist()
    )
    # Remove bin columns from numeric_cols to avoid duplicate feature names
    numeric_cols = [c for c in numeric_cols if c not in bin_cols]

    feature_cols: List[str] = numeric_cols + bin_cols

    df_model = df_feat[numeric_cols].copy()
    for col in bin_cols:
        codes = df_feat[col].astype("float32")
        df_model[col] = codes.fillna(-1).astype("int16")

    categorical_features: List[str] = bin_cols

    return df_model[feature_cols], feature_cols, categorical_features

