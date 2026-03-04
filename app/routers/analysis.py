"""
EBA Building Genome Web — Analysis Router
============================================
Main API endpoints for the energy baseline analysis.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import ALL_FEATURES, WEATHER_FEATURES, TIME_FEATURES
from app.loader import load_meter_summary, load_meter_combined
from app.features import remove_outliers_iqr, get_available_features
from app.regression import (
    build_pipeline, train_model, predict,
    get_coefficients, get_model_info, get_original_scale_formula,
)
from app.metrics import compute_metrics
from app.schemas import (
    MeterListResponse, MeterInfo, MeterDataResponse, MeterDataPoint,
    AnalysisRequest, AnalysisResponse,
    TrainingResult, ReportingResult, BaselineResult,
    TimeSeriesPoint, FormulaResult,
    DriverRow, DriverResult, MonthlyContribution,
    YoYResult, YoYMonth, MonthlySavingsRow,
)

router = APIRouter(prefix="/api", tags=["analysis"])


# ── Meter list ──────────────────────────────────────────────────────────────

@router.get("/meters", response_model=MeterListResponse)
def list_meters():
    """Return all available meters with metadata."""
    summary = load_meter_summary()
    meters = []
    for _, row in summary.iterrows():
        meters.append(MeterInfo(
            meter=row["meter"],
            site=row["site"],
            building_type=row["building_type"],
            lat=row.get("lat"),
            lng=row.get("lng"),
            timezone=row.get("timezone"),
            total_days=int(row.get("total_days", 0)),
            train_days=int(row.get("train_days", 0)),
            test_days=int(row.get("test_days", 0)),
        ))
    return MeterListResponse(meters=meters, total=len(meters))


# ── Meter data ──────────────────────────────────────────────────────────────

@router.get("/meters/{meter_name}", response_model=MeterDataResponse)
def get_meter_data(meter_name: str):
    """Return full time series data for a meter."""
    summary = load_meter_summary()
    if meter_name not in summary["meter"].values:
        raise HTTPException(404, f"Meter '{meter_name}' not found")

    info = summary[summary["meter"] == meter_name].iloc[0]
    df = load_meter_combined(meter_name)
    features = get_available_features(df)

    data = []
    for date_idx, row in df.iterrows():
        data.append(MeterDataPoint(
            date=date_idx.strftime("%Y-%m-%d"),
            daily_kwh=round(float(row["daily_kwh"]), 2),
        ))

    return MeterDataResponse(
        meter=meter_name,
        site=str(info["site"]),
        building_type=str(info["building_type"]),
        min_date=df.index.min().strftime("%Y-%m-%d"),
        max_date=df.index.max().strftime("%Y-%m-%d"),
        total_days=len(df),
        data=data,
        features=features,
    )


# ── Features list ──────────────────────────────────────────────────────────

@router.get("/features")
def list_features():
    """Return all available features grouped by type."""
    return {
        "weather": WEATHER_FEATURES,
        "time": TIME_FEATURES,
        "all": ALL_FEATURES,
    }


# ── Run analysis ────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse)
def run_analysis(req: AnalysisRequest):
    """Run ElasticNet analysis and return complete results."""
    # Load data
    summary = load_meter_summary()
    if req.meter not in summary["meter"].values:
        raise HTTPException(404, f"Meter '{req.meter}' not found")

    meter_row = summary[summary["meter"] == req.meter].iloc[0]
    df = load_meter_combined(req.meter)

    min_date = df.index.min()
    max_date = df.index.max()
    rp_start = pd.to_datetime(req.rp_start)
    rp_end = pd.to_datetime(req.rp_end)

    # Validate reporting period first
    if rp_start >= rp_end:
        raise HTTPException(400, "Reporting start must be before end")

    # Resolve training period
    if req.training_mode == "custom" and req.tr_start and req.tr_end:
        tr_start = pd.to_datetime(req.tr_start)
        tr_end = pd.to_datetime(req.tr_end)
    elif req.training_mode == "sync_baseline" and req.bl_start and req.bl_end:
        tr_start = pd.to_datetime(req.bl_start)
        tr_end = pd.to_datetime(req.bl_end)
    else:  # "all"
        tr_start = min_date
        tr_end = min(max_date, rp_start - pd.Timedelta(days=1))

    # Validate
    if tr_start >= tr_end:
        raise HTTPException(400, "Training start must be before end")

    # Resolve features
    feat_cols = get_available_features(df, req.features)
    if not feat_cols:
        raise HTTPException(400, "No valid features selected")

    # ── Training data ──
    train_mask = (df.index >= tr_start) & (df.index <= tr_end)
    df_train_raw = df[train_mask].copy()

    if req.use_iqr and len(df_train_raw) > 0:
        df_train, outlier_stats = remove_outliers_iqr(df_train_raw, "daily_kwh", req.iqr_k)
    else:
        df_train = df_train_raw.copy()
        outlier_stats = {"outliers_removed": 0, "total_before": len(df_train_raw),
                         "total_after": len(df_train_raw), "pct_removed": 0}

    if len(df_train) < 5:
        raise HTTPException(400, f"Not enough training data: {len(df_train)} days (need >= 5)")

    # ── Reporting data ──
    report_mask = (df.index >= rp_start) & (df.index <= rp_end)
    df_report = df[report_mask].copy()
    if len(df_report) < 1:
        raise HTTPException(400, "No reporting data in selected period")

    # ── Baseline data (mandatory) ──
    bl_start_dt = pd.to_datetime(req.bl_start)
    bl_end_dt = pd.to_datetime(req.bl_end)
    if bl_start_dt >= bl_end_dt:
        raise HTTPException(400, "Baseline start must be before end")
    bl_mask = (df.index >= bl_start_dt) & (df.index <= bl_end_dt)
    df_baseline = df[bl_mask].copy()
    if len(df_baseline) < 1:
        raise HTTPException(400, "No baseline data in selected period")

    # ── Train model ──
    X_train = df_train[feat_cols].values
    y_train = df_train["daily_kwh"].values
    X_report = df_report[feat_cols].values
    y_report = df_report["daily_kwh"].values

    pipeline = build_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)

    y_pred_train = predict(pipeline, X_train)
    y_pred_report = predict(pipeline, X_report)

    # ── Metrics (model quality) ──
    metrics_train = compute_metrics(y_train, y_pred_train)
    metrics_report = compute_metrics(y_report, y_pred_report)
    coef_info = get_coefficients(pipeline, feat_cols)
    m_info = get_model_info(pipeline)

    # ── Savings: Baseline Actual vs Reporting Actual ──
    bl_total_actual = float(df_baseline["daily_kwh"].sum())
    rp_total_actual = float(np.sum(y_report))
    total_savings = bl_total_actual - rp_total_actual
    savings_pct = total_savings / bl_total_actual * 100 if bl_total_actual else 0
    savings = {
        "baseline_total_kwh": round(bl_total_actual, 0),
        "reporting_total_kwh": round(rp_total_actual, 0),
        "total_savings_kwh": round(total_savings, 0),
        "savings_pct": round(savings_pct, 2),
    }

    # ── Training result ──
    train_points = [
        TimeSeriesPoint(
            date=d.strftime("%Y-%m-%d"),
            actual=round(float(a), 2),
            predicted=round(float(p), 2),
        )
        for d, a, p in zip(df_train.index, y_train, y_pred_train)
    ]
    training_result = TrainingResult(
        metrics=metrics_train,
        days=len(df_train),
        outlier_stats=outlier_stats,
        data=train_points,
    )

    # ── Reporting result ──
    savings_daily = (y_pred_report - y_report).tolist()
    cumulative = np.cumsum(y_pred_report - y_report).tolist()
    report_points = [
        TimeSeriesPoint(
            date=d.strftime("%Y-%m-%d"),
            actual=round(float(a), 2),
            predicted=round(float(p), 2),
        )
        for d, a, p in zip(df_report.index, y_report, y_pred_report)
    ]
    reporting_result = ReportingResult(
        metrics=metrics_report,
        days=len(df_report),
        data=report_points,
        savings_daily=[round(float(s), 2) for s in savings_daily],
        cumulative_savings=[round(float(c), 2) for c in cumulative],
    )

    # ── Baseline result ──
    X_bl = df_baseline[feat_cols].values
    y_bl = df_baseline["daily_kwh"].values
    y_pred_bl = predict(pipeline, X_bl)
    bl_points = [
        TimeSeriesPoint(
            date=d.strftime("%Y-%m-%d"),
            actual=round(float(a), 2),
            predicted=round(float(p), 2),
        )
        for d, a, p in zip(df_baseline.index, y_bl, y_pred_bl)
    ]
    baseline_result = BaselineResult(days=len(df_baseline), data=bl_points)

    # ── Formula ──
    orig_formula = get_original_scale_formula(
        pipeline, feat_cols, coef_info["intercept"], coef_info["coefficients"]
    )

    # Build formula strings
    std_parts = []
    for feat, coef in coef_info["coefficients"].items():
        if abs(coef) > 1e-8:
            sign = "+" if coef >= 0 else "-"
            std_parts.append(f" {sign} {abs(coef):.4f}*z({feat})")
    std_formula = f"daily_kwh = {coef_info['intercept']:.4f}" + "".join(std_parts)

    orig_parts = []
    for feat, coef in orig_formula["coefficients"].items():
        if abs(coef) > 1e-8:
            sign = "+" if coef >= 0 else "-"
            orig_parts.append(f"  {sign} {abs(coef):.6f} * {feat}")
    orig_formula_str = f"daily_kwh = {orig_formula['intercept']:.4f}\n" + "\n".join(orig_parts)

    excel_parts = [f"{orig_formula['intercept']:.6f}"]
    for feat, coef in orig_formula["coefficients"].items():
        if abs(coef) > 1e-8:
            sign = "+" if coef >= 0 else ""
            excel_parts.append(f"{sign}{coef:.6f}*[{feat}]")
    excel_formula = "= " + "".join(excel_parts)

    formula_result = FormulaResult(
        standardized=std_formula,
        original_scale=orig_formula_str,
        excel=excel_formula,
        coefficients=coef_info["coefficients"],
        original_coefficients=orig_formula["coefficients"],
        intercept=coef_info["intercept"],
        original_intercept=orig_formula["intercept"],
    )

    # ── Driver Analysis ──
    nonzero_feats = [f for f in feat_cols if abs(orig_formula["coefficients"].get(f, 0)) > 1e-8]
    driver_rows = []
    for feat in nonzero_feats:
        tr_mean = float(df_train[feat].mean())
        rp_mean = float(df_report[feat].mean())
        change = rp_mean - tr_mean
        coef = orig_formula["coefficients"][feat]
        impact = change * coef
        direction = "increase" if impact > 0 else "decrease"
        driver_rows.append(DriverRow(
            feature=feat,
            training_avg=round(tr_mean, 2),
            reporting_avg=round(rp_mean, 2),
            change=round(change, 2),
            coefficient=round(coef, 6),
            energy_impact=round(impact, 2),
            direction=direction,
        ))

    # Monthly contributions
    contrib_df = pd.DataFrame(index=df_report.index)
    for feat in feat_cols:
        coef = orig_formula["coefficients"].get(feat, 0)
        contrib_df[feat] = df_report[feat].values * coef
    contrib_df["intercept"] = orig_formula["intercept"]
    contrib_df["total_predicted"] = y_pred_report
    contrib_df["_period"] = contrib_df.index.to_period("M").astype(str)
    monthly_contrib = contrib_df.groupby("_period", sort=True).sum(numeric_only=True)

    monthly_contribs = []
    for month, row in monthly_contrib.iterrows():
        contribs = {f: round(float(row.get(f, 0)), 2) for f in nonzero_feats}
        contribs["intercept"] = round(float(row.get("intercept", 0)), 2)
        monthly_contribs.append(MonthlyContribution(
            month=str(month),
            contributions=contribs,
            total_predicted=round(float(row.get("total_predicted", 0)), 2),
        ))

    driver_result = DriverResult(
        drivers=sorted(driver_rows, key=lambda x: abs(x.energy_impact), reverse=True),
        monthly_contributions=monthly_contribs,
    )

    # ── Year-over-Year: Baseline Actual vs Reporting Actual ──
    bl_monthly = df_baseline.resample("ME")["daily_kwh"].sum().reset_index()
    bl_monthly["month_num"] = bl_monthly["date"].dt.month

    rp_monthly = df_report.resample("ME")["daily_kwh"].sum().reset_index()
    rp_monthly["month_num"] = rp_monthly["date"].dt.month

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    yoy_months = []
    for m in range(1, 13):
        bl_val = bl_monthly[bl_monthly["month_num"] == m]["daily_kwh"].sum()
        rp_val = rp_monthly[rp_monthly["month_num"] == m]["daily_kwh"].sum()

        bl_v = round(float(bl_val), 0) if bl_val > 0 else None
        rp_v = round(float(rp_val), 0) if rp_val > 0 else None

        sav = round(bl_v - rp_v, 0) if (bl_v and rp_v) else None
        sav_pct = round(sav / bl_v * 100, 1) if (sav is not None and bl_v) else None

        yoy_months.append(YoYMonth(
            month=month_names[m - 1],
            month_num=m,
            baseline_actual=bl_v,
            reporting_actual=rp_v,
            savings_kwh=sav,
            savings_pct=sav_pct,
        ))

    # Totals
    yoy_totals = {
        "baseline_actual": round(bl_total_actual, 0),
        "reporting_actual": round(rp_total_actual, 0),
        "savings_kwh": round(total_savings, 0),
        "savings_pct": round(savings_pct, 1),
    }

    yoy_result = YoYResult(months=yoy_months, totals=yoy_totals)

    # ── Monthly Savings: Baseline Actual vs Reporting Actual by month ──
    bl_month_agg = df_baseline.copy()
    bl_month_agg["month_num"] = bl_month_agg.index.month
    bl_month_agg = bl_month_agg.groupby("month_num")["daily_kwh"].sum()

    rp_month_agg = df_report.copy()
    rp_month_agg["month_num"] = rp_month_agg.index.month
    rp_month_agg = rp_month_agg.groupby("month_num")["daily_kwh"].sum()

    month_names_short = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_savings_rows = []
    for m in range(1, 13):
        bl_kwh = float(bl_month_agg.get(m, 0))
        rp_kwh = float(rp_month_agg.get(m, 0))
        if bl_kwh == 0 and rp_kwh == 0:
            continue
        sav = bl_kwh - rp_kwh
        sav_pct = sav / bl_kwh * 100 if bl_kwh else 0
        monthly_savings_rows.append(MonthlySavingsRow(
            month=month_names_short[m - 1],
            actual=round(rp_kwh, 0),
            baseline=round(bl_kwh, 0),
            savings=round(sav, 0),
            savings_pct=round(sav_pct, 1),
        ))

    return AnalysisResponse(
        meter=req.meter,
        site=str(meter_row["site"]),
        building_type=str(meter_row["building_type"]),
        model_info=m_info,
        training=training_result,
        reporting=reporting_result,
        baseline=baseline_result,
        savings=savings,
        formula=formula_result,
        drivers=driver_result,
        yoy=yoy_result,
        monthly_savings=monthly_savings_rows,
        features_used=feat_cols,
    )
