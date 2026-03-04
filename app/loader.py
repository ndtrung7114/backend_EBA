"""
EBA Building Genome Web — Data Loader
=======================================
Load meter data from bundled CSV files.
"""

import pandas as pd
from app.config import DATA_DIR, TRAIN_DIR, TEST_DIR


def load_meter_summary() -> pd.DataFrame:
    """Load the meter summary CSV with site/building/location info."""
    path = DATA_DIR / "meter_summary.csv"
    return pd.read_csv(path)


def load_meter_train(meter_name: str) -> pd.DataFrame:
    """Load training data for a single meter."""
    path = TRAIN_DIR / f"{meter_name}_train.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.rename(columns={"usage_value": "daily_kwh"})
    return df


def load_meter_test(meter_name: str) -> pd.DataFrame:
    """Load testing data for a single meter."""
    path = TEST_DIR / f"{meter_name}_test.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.rename(columns={"usage_value": "daily_kwh"})
    return df


def load_meter_combined(meter_name: str) -> pd.DataFrame:
    """Load train + test data concatenated."""
    df_train = load_meter_train(meter_name)
    df_test = load_meter_test(meter_name)
    df = pd.concat([df_train, df_test]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df
