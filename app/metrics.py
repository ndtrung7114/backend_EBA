"""
EBA Building Genome Web — ASHRAE Metrics & Savings
=====================================================
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return {}
    y_mean = float(np.mean(y_true))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
    cvrmse = rmse / y_mean * 100 if y_mean else float("inf")
    nmbe = float(np.sum(y_true - y_pred)) / (n * y_mean) * 100 if y_mean else float("inf")
    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE_pct": round(mape, 2),
        "CVRMSE_pct": round(cvrmse, 2),
        "NMBE_pct": round(nmbe, 2),
        "Mean_Actual": round(y_mean, 2),
        "Mean_Predicted": round(float(np.mean(y_pred)), 2),
        "n": n,
        "ASHRAE_CVRMSE": "PASS" if abs(cvrmse) < 25 else "FAIL",
        "ASHRAE_NMBE": "PASS" if abs(nmbe) < 10 else "FAIL",
    }


def compute_savings(y_actual: np.ndarray, y_predicted: np.ndarray) -> dict:
    savings = y_predicted - y_actual
    total_pred = float(np.sum(y_predicted))
    total_actual = float(np.sum(y_actual))
    total_save = float(np.sum(savings))
    pct = total_save / total_pred * 100 if total_pred else 0
    return {
        "total_predicted_kwh": round(total_pred, 0),
        "total_actual_kwh": round(total_actual, 0),
        "total_savings_kwh": round(total_save, 0),
        "savings_pct": round(pct, 2),
        "days_with_savings": int(np.sum(savings > 0)),
        "days_with_excess": int(np.sum(savings < 0)),
        "n_days": len(savings),
    }
