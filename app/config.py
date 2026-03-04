"""
EBA Building Genome Web — Configuration
=========================================
Subset of 15 representative meters, ElasticNet model only.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "training"
TEST_DIR = DATA_DIR / "testing"

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

WEATHER_FEATURES = [
    "maxtempC", "mintempC", "avgtempC",
    "humidity", "sunHour", "uvIndex",
    "windspeedKmph", "pressure", "winddirDegree",
    "visibility", "cloudcover",
    "HeatIndexC", "WindChillC", "WindGustKmph", "FeelsLikeC",
]

TIME_FEATURES = [
    "month", "month_day", "week_day",
    "season", "is_weekend", "is_holiday",
]

ALL_FEATURES = WEATHER_FEATURES + TIME_FEATURES

# ============================================================================
# MODEL CONFIG — ElasticNet only
# ============================================================================

IQR_K = 1.5
MIN_ENERGY_KWH = 0.1
RANDOM_STATE = 42
CV_FOLDS = 5
