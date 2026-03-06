"""
EBA Building Genome Web — Data Loader
=======================================
Load meter data from bundled CSV files.
Supports three data groups: Building Genome, Saint-Gobain, ECD.
"""

import pandas as pd
from app.config import DATA_DIR, TRAIN_DIR, TEST_DIR, SG_DIR, ECD_DIR


def load_meter_summary() -> pd.DataFrame:
    """Load the meter summary CSV with site/building/location/group info."""
    path = DATA_DIR / "meter_summary.csv"
    return pd.read_csv(path)


def _get_meter_group(meter_name: str) -> str:
    """Determine group from meter name prefix."""
    if meter_name.startswith("SG_"):
        return "Saint-Gobain"
    elif meter_name.startswith("ECD_"):
        return "ECD"
    return "Building Genome"


def load_meter_train(meter_name: str) -> pd.DataFrame:
    """Load training data for a BG meter."""
    path = TRAIN_DIR / f"{meter_name}_train.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.rename(columns={"usage_value": "daily_kwh"})
    return df


def load_meter_test(meter_name: str) -> pd.DataFrame:
    """Load testing data for a BG meter."""
    path = TEST_DIR / f"{meter_name}_test.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.rename(columns={"usage_value": "daily_kwh"})
    return df


def load_meter_combined(meter_name: str) -> pd.DataFrame:
    """Load all data for any meter (BG, SG, or ECD)."""
    group = _get_meter_group(meter_name)

    if group == "Saint-Gobain":
        path = SG_DIR / f"{meter_name}.csv"
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    elif group == "ECD":
        path = ECD_DIR / f"{meter_name}.csv"
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    else:
        # Building Genome: combine train + test
        df_train = load_meter_train(meter_name)
        df_test = load_meter_test(meter_name)
        df = pd.concat([df_train, df_test]).sort_index()

    df = df[~df.index.duplicated(keep="first")]
    return df
