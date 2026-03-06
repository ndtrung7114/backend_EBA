"""
EBA Building Genome Web — Pydantic Schemas
=============================================
Request / response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class MeterInfo(BaseModel):
    meter: str
    group: str
    site: str
    building_type: str
    total_days: int
    min_date: str = ""
    max_date: str = ""
    avg_daily_kwh: float = 0


class MeterListResponse(BaseModel):
    meters: list[MeterInfo]
    total: int
    groups: list[str]


class MeterDataPoint(BaseModel):
    date: str
    daily_kwh: float


class MeterDataResponse(BaseModel):
    meter: str
    group: str
    site: str
    building_type: str
    min_date: str
    max_date: str
    total_days: int
    data: list[MeterDataPoint]
    features: list[str]


class AnalysisRequest(BaseModel):
    meter: str
    rp_start: str  # YYYY-MM-DD
    rp_end: str
    baseline_enabled: bool = False
    bl_start: Optional[str] = None
    bl_end: Optional[str] = None
    tr_start: Optional[str] = None
    tr_end: Optional[str] = None
    training_mode: str = "all"  # "all" | "custom" | "sync_baseline"
    features: Optional[list[str]] = None
    use_iqr: bool = True
    iqr_k: float = Field(default=1.5, ge=1.0, le=3.0)


class TimeSeriesPoint(BaseModel):
    date: str
    actual: float
    predicted: float


class TrainingResult(BaseModel):
    metrics: dict
    days: int
    outlier_stats: dict
    data: list[TimeSeriesPoint]


class ReportingResult(BaseModel):
    metrics: dict
    days: int
    data: list[TimeSeriesPoint]
    savings_daily: list[float]
    cumulative_savings: list[float]


class BaselineResult(BaseModel):
    days: int
    data: list[TimeSeriesPoint]


class FormulaResult(BaseModel):
    standardized: str
    original_scale: str
    excel: str
    coefficients: dict[str, float]
    original_coefficients: dict[str, float]
    intercept: float
    original_intercept: float


class DriverRow(BaseModel):
    feature: str
    training_avg: float
    reporting_avg: float
    change: float
    coefficient: float
    energy_impact: float
    direction: str


class MonthlyContribution(BaseModel):
    month: str
    contributions: dict[str, float]
    total_predicted: float


class DriverResult(BaseModel):
    drivers: list[DriverRow]
    monthly_contributions: list[MonthlyContribution]


# YoY: Baseline Actual vs Reporting Actual (direct DB comparison)
class YoYMonth(BaseModel):
    month: str
    month_num: int
    baseline_actual: Optional[float] = None
    reporting_actual: Optional[float] = None
    savings_kwh: Optional[float] = None
    savings_pct: Optional[float] = None


class YoYResult(BaseModel):
    months: list[YoYMonth]
    totals: dict


# Monthly: Reporting Actual vs Reporting Predicted (normalized comparison)
class MonthlySavingsRow(BaseModel):
    month: str
    actual: float      # actual usage from DB in reporting period
    predicted: float   # model-predicted (normalized) usage
    savings: float     # actual - predicted
    savings_pct: float


class AnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    meter: str
    group: str
    site: str
    building_type: str
    model_info: dict
    training: TrainingResult
    reporting: ReportingResult
    baseline: Optional[BaselineResult] = None
    savings: dict
    formula: FormulaResult
    drivers: DriverResult
    yoy: Optional[YoYResult] = None
    monthly_savings: list[MonthlySavingsRow]
    features_used: list[str]
