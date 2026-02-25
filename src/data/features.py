from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .preprocess import basic_preprocess


def fit_risk_groupanizer(df_train: pd.DataFrame, column_names: List[str], target_val: int = 1, upper_limit_ratio: float = 8.2, lower_limit_ratio: float = 8.2) -> Dict[str, Dict[str, List[Any]]]:
    """
    Fit risk groupanizer on training data.
    Finds categories in specified columns that correspond to high/medium/low risk of the target.
    """
    risk_mapping: Dict[str, Dict[str, List[Any]]] = {}
    if "TARGET" not in df_train.columns:
        return risk_mapping

    for col in column_names:
        temp_df = df_train.groupby([col, 'TARGET'])[['SK_ID_CURR']].count().reset_index()
        totals = temp_df.groupby(col)['SK_ID_CURR'].transform('sum')
        temp_df['ratio%'] = round(temp_df['SK_ID_CURR'] * 100 / totals, 1)

        col_groups_high_risk = temp_df[(temp_df['TARGET'] == target_val) & 
                                       (temp_df['ratio%'] >= upper_limit_ratio)][col].tolist()
        
        col_groups_low_risk = temp_df[(temp_df['TARGET'] == target_val) & 
                                      (temp_df['ratio%'] <= lower_limit_ratio)][col].tolist()
                                      
        col_groups_medium_risk = []
        if upper_limit_ratio != lower_limit_ratio:
            col_groups_medium_risk = temp_df[(temp_df['TARGET'] == target_val) & 
                                             (temp_df['ratio%'] < upper_limit_ratio) & 
                                             (temp_df['ratio%'] > lower_limit_ratio)][col].tolist()
                                             
        risk_mapping[col] = {
            '_high_risk': col_groups_high_risk,
            '_medium_risk': col_groups_medium_risk,
            '_low_risk': col_groups_low_risk
        }
    return risk_mapping


def transform_risk_groupanizer(df: pd.DataFrame, risk_mapping: Dict[str, Dict[str, List[Any]]]) -> pd.DataFrame:
    """
    Transform dataframe using fitted risk mapping.
    """
    df = df.copy()
    for col, mappings in risk_mapping.items():
        if col not in df.columns:
            continue
            
        high_risk = mappings.get('_high_risk', [])
        medium_risk = mappings.get('_medium_risk', [])
        low_risk = mappings.get('_low_risk', [])
        
        if medium_risk:
            for risk_label, risk_groups in zip(['_high_risk', '_medium_risk', '_low_risk'], 
                                               [high_risk, medium_risk, low_risk]):
                df[col + risk_label] = df[col].isin(risk_groups).astype(np.int8)
        else:
            for risk_label, risk_groups in zip(['_high_risk', '_low_risk'], 
                                               [high_risk, low_risk]):
                df[col + risk_label] = df[col].isin(risk_groups).astype(np.int8)
                
        if df[col].dtype == 'O' or (hasattr(df[col].dtype, 'name') and df[col].dtype.name == 'category'):
            df.drop(columns=[col], inplace=True)
            
    return df


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

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = _safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    if {"AMT_INCOME_TOTAL", "DAYS_EMPLOYED"}.issubset(df.columns):
        df["INCOME_TO_EMPLOYED"] = _safe_divide(df["AMT_INCOME_TOTAL"], df["DAYS_EMPLOYED"])

    if {"AMT_INCOME_TOTAL", "AMT_ANNUITY"}.issubset(df.columns):
        df["INCOME_MONTHLY_MINUS_ANNUITY"] = (df["AMT_INCOME_TOTAL"] / 12.0) - df["AMT_ANNUITY"]

    # --- Log-transformed amounts for heavy-tailed distributions ---
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df.columns:
            df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

    # --- Advanced EXT_SOURCE interactions ---
    ext_sources = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_sources:
        df["EXT_SOURCE_MEAN"] = df[ext_sources].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_sources].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_sources].max(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_sources].std(axis=1)
        df["EXT_SOURCE_MEAN_SQ"] = df["EXT_SOURCE_MEAN"] ** 2

        # Polynomials
        if {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(df.columns):
            df["EXT_1_2_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        if {"EXT_SOURCE_1", "EXT_SOURCE_3"}.issubset(df.columns):
            df["EXT_1_3_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
        if {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(df.columns):
            df["EXT_2_3_PROD"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        if {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(df.columns):
            df["EXT_1_2_3_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            df["EXT_SOURCES_WEIGHTED"] = df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 1 + df["EXT_SOURCE_3"] * 3
            
        # Ratios to Age/Employment
        if {"EXT_SOURCE_1", "DAYS_BIRTH"}.issubset(df.columns):
            df["EXT_1_TO_BIRTH"] = _safe_divide(df["EXT_SOURCE_1"], df["DAYS_BIRTH"])
        if {"EXT_SOURCE_2", "DAYS_BIRTH"}.issubset(df.columns):
            df["EXT_2_TO_BIRTH"] = _safe_divide(df["EXT_SOURCE_2"], df["DAYS_BIRTH"])
        if {"EXT_SOURCE_3", "DAYS_BIRTH"}.issubset(df.columns):
            df["EXT_3_TO_BIRTH"] = _safe_divide(df["EXT_SOURCE_3"], df["DAYS_BIRTH"])
            
        if {"EXT_SOURCE_1", "DAYS_EMPLOYED"}.issubset(df.columns):
            df["EXT_1_TO_EMPLOYED"] = _safe_divide(df["EXT_SOURCE_1"], df["DAYS_EMPLOYED"])
        if {"EXT_SOURCE_2", "DAYS_EMPLOYED"}.issubset(df.columns):
            df["EXT_2_TO_EMPLOYED"] = _safe_divide(df["EXT_SOURCE_2"], df["DAYS_EMPLOYED"])
        if {"EXT_SOURCE_3", "DAYS_EMPLOYED"}.issubset(df.columns):
            df["EXT_3_TO_EMPLOYED"] = _safe_divide(df["EXT_SOURCE_3"], df["DAYS_EMPLOYED"])

        # Advanced EXT_SOURCE specialized ratios
        if {"EXT_SOURCE_1", "CNT_FAM_MEMBERS"}.issubset(df.columns):
            df["EXT_1_TO_FAM_CNT_RATIO"] = _safe_divide(df["EXT_SOURCE_1"], df["CNT_FAM_MEMBERS"])
        if {"EXT_SOURCE_1", "AMT_GOODS_PRICE"}.issubset(df.columns):
            df["EXT_1_TO_GOODS_RATIO"] = _safe_divide(df["EXT_SOURCE_1"], df["AMT_GOODS_PRICE"])
        if {"EXT_SOURCE_1", "AMT_CREDIT"}.issubset(df.columns):
            df["EXT_1_TO_CREDIT_RATIO"] = _safe_divide(df["EXT_SOURCE_1"], df["AMT_CREDIT"])
        if {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(df.columns):
            df["EXT_1_TO_SCORE2_RATIO"] = _safe_divide(df["EXT_SOURCE_1"], df["EXT_SOURCE_2"])
        if {"EXT_SOURCE_1", "EXT_SOURCE_3"}.issubset(df.columns):
            df["EXT_1_TO_SCORE3_RATIO"] = _safe_divide(df["EXT_SOURCE_1"], df["EXT_SOURCE_3"])
            
        if {"EXT_SOURCE_2", "AMT_CREDIT"}.issubset(df.columns):
            df["EXT_2_TO_CREDIT_RATIO"] = _safe_divide(df["EXT_SOURCE_2"], df["AMT_CREDIT"])
        if {"EXT_SOURCE_2", "REGION_RATING_CLIENT"}.issubset(df.columns):
            df["EXT_2_TO_REGION_RATING_RATIO"] = _safe_divide(df["EXT_SOURCE_2"], df["REGION_RATING_CLIENT"])
        if {"EXT_SOURCE_2", "REGION_RATING_CLIENT_W_CITY"}.issubset(df.columns):
            df["EXT_2_TO_CITY_RATING_RATIO"] = _safe_divide(df["EXT_SOURCE_2"], df["REGION_RATING_CLIENT_W_CITY"])
        if {"EXT_SOURCE_2", "REGION_POPULATION_RELATIVE"}.issubset(df.columns):
            df["EXT_2_TO_POP_RATIO"] = _safe_divide(df["EXT_SOURCE_2"], df["REGION_POPULATION_RELATIVE"])
        if {"EXT_SOURCE_2", "DAYS_LAST_PHONE_CHANGE"}.issubset(df.columns):
            df["EXT_2_TO_PHONE_CHANGE_RATIO"] = _safe_divide(df["EXT_SOURCE_2"], df["DAYS_LAST_PHONE_CHANGE"])

    # --- Missingness Features ---
    # Many missing values are predictive
    df["TOTAL_MISSING"] = df.isnull().sum(axis=1)
    
    building_cols = [c for c in df.columns if c in [
        'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
        'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
        'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE'
    ]]
    if building_cols:
        df["BUILDING_MISSING_COUNT"] = df[building_cols].isnull().sum(axis=1)

    # --- Flag Aggregations ---
    doc_flags = [c for c in df.columns if "FLAG_DOCUMENT" in c]
    if doc_flags:
        df["DOCUMENT_COUNT"] = df[doc_flags].sum(axis=1)
        df["DOCUMENT_KURT"] = df[doc_flags].kurtosis(axis=1)

    contact_flags = [c for c in df.columns if c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]]
    if contact_flags:
        df["CONTACT_FLAGS_SUM"] = df[contact_flags].sum(axis=1)
        
    # --- Social Circle Features ---
    if {"DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"}.issubset(df.columns):
        df["SOCIAL_DEFAULT_RATIO_30"] = _safe_divide(df["DEF_30_CNT_SOCIAL_CIRCLE"], df["OBS_30_CNT_SOCIAL_CIRCLE"])
    if {"DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"}.issubset(df.columns):
        df["SOCIAL_DEFAULT_RATIO_60"] = _safe_divide(df["DEF_60_CNT_SOCIAL_CIRCLE"], df["OBS_60_CNT_SOCIAL_CIRCLE"])
        
    # --- Enquiries Summation ---
    enq_cols = [c for c in df.columns if "AMT_REQ_CREDIT_BUREAU" in c]
    if enq_cols:
        df["TOTAL_BUREAU_ENQ"] = df[enq_cols].sum(axis=1)
        
        recent_enq = [c for c in df.columns if c in ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON"]]
        older_enq = [c for c in df.columns if c in ["AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]]
        if recent_enq:
            df["RECENT_BUREAU_ENQ"] = df[recent_enq].sum(axis=1)
        if older_enq:
            df["OLDER_BUREAU_ENQ"] = df[older_enq].sum(axis=1)

    # --- Temporal features based on DAYS_* ---
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["DAYS_EMPLOYED_PCT"] = _safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
        df["DAYS_EMPLOYED_DAYS_BIRTH_DIFF"] = df["DAYS_EMPLOYED"] - df["DAYS_BIRTH"]

    if {"DAYS_ID_PUBLISH", "DAYS_BIRTH"}.issubset(df.columns):
        df["ID_TO_BIRTH_RATIO"] = _safe_divide(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
        
    if {"OWN_CAR_AGE", "DAYS_BIRTH"}.issubset(df.columns):
        df["CAR_TO_BIRTH_RATIO"] = _safe_divide(df["OWN_CAR_AGE"], df["DAYS_BIRTH"])
        
    if {"OWN_CAR_AGE", "DAYS_EMPLOYED"}.issubset(df.columns):
        df["CAR_TO_EMPLOYED_RATIO"] = _safe_divide(df["OWN_CAR_AGE"], df["DAYS_EMPLOYED"])
        
    if {"DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"}.issubset(df.columns):
        df["PHONE_TO_BIRTH_RATIO"] = _safe_divide(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])
        
    if {"DAYS_LAST_PHONE_CHANGE", "DAYS_EMPLOYED"}.issubset(df.columns):
        df["PHONE_TO_EMPLOYED_RATIO"] = _safe_divide(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_EMPLOYED"])

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
