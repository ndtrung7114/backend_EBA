"""
EBA Building Genome Web — Feature Engineering
================================================
IQR outlier removal, feature validation.
"""

import numpy as np
import pandas as pd
from app.config import ALL_FEATURES, IQR_K, MIN_ENERGY_KWH


def remove_outliers_iqr(df: pd.DataFrame, col: str = "daily_kwh", k: float = None) -> tuple:
    if k is None:
        k = IQR_K
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    lower = max(q25 - k * iqr, MIN_ENERGY_KWH)
    upper = q75 + k * iqr
    mask = (df[col] >= lower) & (df[col] <= upper)
    n_before = len(df)
    df_clean = df[mask].copy()
    stats = {
        "Q25": round(float(q25), 2),
        "Q75": round(float(q75), 2),
        "IQR": round(float(iqr), 2),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "total_before": n_before,
        "outliers_removed": n_before - len(df_clean),
        "total_after": len(df_clean),
        "pct_removed": round((n_before - len(df_clean)) / n_before * 100, 1) if n_before else 0,
    }
    return df_clean, stats


def get_available_features(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    """Return features that exist in the dataframe."""
    features = requested or ALL_FEATURES
    return [f for f in features if f in df.columns]
