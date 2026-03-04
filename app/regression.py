"""
EBA Building Genome Web — ElasticNet Regression
=================================================
Fixed to ElasticNetCV — auto-tuned alpha and l1_ratio.
"""

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from app.config import RANDOM_STATE, CV_FOLDS


def build_pipeline() -> Pipeline:
    """Build ElasticNetCV pipeline with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-3, 3, 50),
            cv=CV_FOLDS,
            max_iter=50000,
            random_state=RANDOM_STATE,
        )),
    ])


def train_model(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline


def predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict(X)


def get_coefficients(pipeline: Pipeline, feature_names: list[str]) -> dict:
    reg = pipeline.named_steps["regressor"]
    coefs = reg.coef_
    intercept = float(reg.intercept_)
    return {
        "coefficients": dict(zip(feature_names, [round(float(c), 6) for c in coefs])),
        "intercept": round(intercept, 6),
    }


def get_model_info(pipeline: Pipeline) -> dict:
    reg = pipeline.named_steps["regressor"]
    info = {
        "name": "ElasticNet",
        "description": "L1+L2 hybrid — auto-tuned alpha and l1_ratio via ElasticNetCV.",
    }
    if hasattr(reg, "alpha_"):
        info["best_alpha"] = round(float(reg.alpha_), 6)
    if hasattr(reg, "l1_ratio_"):
        info["best_l1_ratio"] = round(float(reg.l1_ratio_), 4)
    if hasattr(reg, "coef_"):
        info["n_nonzero_coefs"] = int(np.sum(np.abs(reg.coef_) > 1e-8))
        info["n_total_coefs"] = len(reg.coef_)
    return info


def get_original_scale_formula(pipeline: Pipeline, feature_names: list[str], intercept: float, coefs: dict) -> dict:
    """Convert standardized coefficients to original-scale formula."""
    scaler = pipeline.named_steps["scaler"]
    scaler_means = scaler.mean_
    scaler_scales = scaler.scale_

    orig_intercept = intercept
    orig_coefs = {}
    for i, feat in enumerate(feature_names):
        std_coef = coefs.get(feat, 0.0)
        if scaler_scales[i] != 0:
            orig_coef = std_coef / scaler_scales[i]
            orig_intercept -= orig_coef * scaler_means[i]
        else:
            orig_coef = 0.0
        orig_coefs[feat] = round(orig_coef, 6)

    orig_intercept = round(orig_intercept, 6)
    return {
        "intercept": orig_intercept,
        "coefficients": orig_coefs,
    }
