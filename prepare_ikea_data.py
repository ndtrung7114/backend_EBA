"""
Prepare IKEA monitoring point data for EBA Building Genome Web.

Three monitoring points (Asia/Shanghai timezone):
  IKEA_Wuhan           — monitoring_point_id=72126
  IKEA_Hangzhou        — monitoring_point_id=72114
  IKEA_Chengdu_Chenghua — monitoring_point_id=72109

Steps:
  1. Query ClickHouse for daily usage data.
  2. Fetch weather from WorldWeatherOnline API for each city.
  3. Merge + add temporal features -> CSV in data/ikea/.
  4. Append rows to meter_summary.csv.

Usage:
  Set environment variables or edit the CLICKHOUSE_* constants below, then:
    python prepare_ikea_data.py
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import holidays
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# ─── ClickHouse connection ────────────────────────────────────────────────────
CLICKHOUSE_HOST     = os.getenv("CLICKHOUSE_HOST", "ops.akila3d.com")
CLICKHOUSE_PORT     = int(os.getenv("CLICKHOUSE_PORT", "23123"))
CLICKHOUSE_USER     = os.getenv("CLICKHOUSE_USER", "akila3d")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "pwd4clickhouse")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "ecd_uat__ck")

# ─── Weather API ──────────────────────────────────────────────────────────────
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "66bd052572f14cdfb41100047262502")
WEATHER_URL     = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"
CHUNK_DAYS      = 29          # API limit per request
MAX_RETRIES     = 3

# ─── Monitoring points ────────────────────────────────────────────────────────
MONITORING_POINTS = [
    {
        "meter":      "IKEA_Wuhan",
        "mp_id":      "72126",
        "location":   "Wuhan,China",
        "site":       "Wuhan",
        "building_type": "Retail",
    },
    {
        "meter":      "IKEA_Hangzhou",
        "mp_id":      "72114",
        "location":   "Hangzhou,China",
        "site":       "Hangzhou",
        "building_type": "Retail",
    },
    {
        "meter":      "IKEA_Chengdu_Chenghua",
        "mp_id":      "72109",
        "location":   "Chengdu,China",
        "site":       "Chengdu",
        "building_type": "Retail",
    },
]

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
IKEA_DIR = DATA_DIR / "ikea"
IKEA_DIR.mkdir(exist_ok=True)

WEATHER_COLS = [
    "maxtempC", "mintempC", "avgtempC", "humidity", "sunHour", "uvIndex",
    "windspeedKmph", "pressure", "winddirDegree", "visibility", "cloudcover",
    "HeatIndexC", "WindChillC", "WindGustKmph", "FeelsLikeC",
]

# ─── ClickHouse helpers ───────────────────────────────────────────────────────

def _get_ch_client():
    """Return a clickhouse_connect Client (HTTP interface). Requires: pip install clickhouse-connect"""
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )


def fetch_usage(mp_id: str) -> pd.DataFrame:
    """Query daily usage for one monitoring point from ClickHouse.

    Returns a DataFrame with columns: date, daily_kwh
    Index is NOT set — caller handles it.
    """
    print(f"  Querying ClickHouse for monitoring_point_id={mp_id} …")
    client = _get_ch_client()

    df = client.query_df(
        """
        SELECT
            data_date                AS date,
            usage_value,
            interpolated_usage_value AS interpolated_value
        FROM ecd_uat__ck.monitoring_point_usage_day
        WHERE monitoring_point_id = {mp_id:UInt64}
        ORDER BY date
        """,
        parameters={"mp_id": int(mp_id)},
    )

    df["date"] = pd.to_datetime(df["date"])

    # Use usage_value; fall back to interpolated_value when null/zero
    df["daily_kwh"] = df["usage_value"].where(
        df["usage_value"].notna() & (df["usage_value"] > 0),
        other=df["interpolated_value"],
    )

    print(f"    -> {len(df)} days  ({df['date'].min().date()} -> {df['date'].max().date()})")
    return df[["date", "daily_kwh", "interpolated_value"]]


# ─── Weather helpers ─────────────────────────────────────────────────────────

def _process_weather_response(data: Dict[str, Any]) -> Dict[str, dict]:
    result = {}
    for day in data["data"]["weather"]:
        hourly = pd.DataFrame(day["hourly"])
        result[day["date"]] = {
            "maxtempC":     float(day["maxtempC"]),
            "mintempC":     float(day["mintempC"]),
            "avgtempC":     float(day["avgtempC"]),
            "sunHour":      float(day["sunHour"]),
            "uvIndex":      float(day["uvIndex"]),
            "windspeedKmph":  hourly["windspeedKmph"].astype(float).mean(),
            "winddirDegree":  hourly["winddirDegree"].astype(float).mean(),
            "humidity":       hourly["humidity"].astype(float).mean(),
            "visibility":     hourly["visibility"].astype(float).mean(),
            "pressure":       hourly["pressure"].astype(float).mean(),
            "cloudcover":     hourly["cloudcover"].astype(float).mean(),
            "HeatIndexC":     hourly["HeatIndexC"].astype(float).mean(),
            "WindChillC":     hourly["WindChillC"].astype(float).mean(),
            "WindGustKmph":   hourly["WindGustKmph"].astype(float).mean(),
            "FeelsLikeC":     hourly["FeelsLikeC"].astype(float).mean(),
        }
    return result


def _fetch_chunk(location: str, start: str, end: str) -> Dict[str, dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                WEATHER_URL,
                params={"key": WEATHER_API_KEY, "q": location,
                        "date": start, "enddate": end, "format": "json"},
                timeout=15,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"    Rate-limited — waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data.get("data", {}):
                raise ValueError(data["data"]["error"][0]["msg"])
            return _process_weather_response(data)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    Attempt {attempt} failed: {exc}. Retrying …")
            time.sleep(2 ** attempt)
    return {}


def fetch_weather(location: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather for location over [start_date, end_date] (inclusive)."""
    print(f"  Fetching weather for {location}: {start_date} -> {end_date}")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    all_data: Dict[str, dict] = {}
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        chunk = _fetch_chunk(
            location,
            cur.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        all_data.update(chunk)
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.3)   # polite pause between chunks

    weather_df = pd.DataFrame.from_dict(all_data, orient="index")
    weather_df.index = pd.to_datetime(weather_df.index)
    weather_df = weather_df.sort_index()
    weather_df.index.name = "date"
    print(f"    -> {len(weather_df)} weather days")
    return weather_df


# ─── Temporal features ────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df["month"]      = idx.month
    df["month_day"]  = idx.day
    df["week_day"]   = idx.dayofweek
    df["season"]     = idx.month.map(
        lambda m: {12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4}[m]
    )
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    years = range(idx.year.min(), idx.year.max() + 1)
    try:
        cn_hols = holidays.China(years=years)
        df["is_holiday"] = idx.map(lambda d: 1 if d in cn_hols else 0)
    except Exception:
        df["is_holiday"] = 0
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_one(mp: dict) -> dict:
    meter = mp["meter"]
    print(f"\n{'='*60}")
    print(f"  {meter}  (ID={mp['mp_id']}, location={mp['location']})")
    print(f"{'='*60}")

    # 1. Usage from ClickHouse
    usage_df = fetch_usage(mp["mp_id"])
    usage_df = usage_df.set_index("date").sort_index()
    usage_df = usage_df[~usage_df.index.duplicated(keep="first")]

    start_str = usage_df.index.min().strftime("%Y-%m-%d")
    end_str   = usage_df.index.max().strftime("%Y-%m-%d")

    # 2. Weather
    weather_df = fetch_weather(mp["location"], start_str, end_str)

    # 3. Merge
    merged = usage_df.join(weather_df, how="inner")
    missing_weather = len(usage_df) - len(merged)
    if missing_weather:
        print(f"  ! Dropped {missing_weather} days without weather data")

    # Drop rows where daily_kwh is null
    before = len(merged)
    merged = merged.dropna(subset=["daily_kwh"])
    if len(merged) < before:
        print(f"  ! Dropped {before - len(merged)} rows with null daily_kwh")

    # 4. Temporal features
    merged = add_temporal_features(merged)

    # Ensure column order matches ECD format, with interpolated_value appended
    col_order = ["daily_kwh", "interpolated_value"] + WEATHER_COLS + [
        "month", "month_day", "week_day", "season", "is_weekend", "is_holiday"
    ]
    merged = merged[col_order]

    # 5. Save CSV
    out_path = IKEA_DIR / f"{meter}.csv"
    merged.to_csv(out_path)
    print(f"  OK Saved {out_path}  ({len(merged)} rows)")

    return {
        "meter":         meter,
        "group":         "IKEA",
        "site":          mp["site"],
        "building_type": mp["building_type"],
        "total_days":    len(merged),
        "min_date":      str(merged.index.min().date()),
        "max_date":      str(merged.index.max().date()),
        "avg_daily_kwh": round(float(merged["daily_kwh"].mean()), 2),
    }


def update_meter_summary(new_rows: list[dict]):
    summary_path = DATA_DIR / "meter_summary.csv"
    summary = pd.read_csv(summary_path)

    # Remove any stale IKEA rows then append fresh ones
    summary = summary[summary["group"] != "IKEA"]
    summary = pd.concat([summary, pd.DataFrame(new_rows)], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nOK meter_summary.csv updated — {len(new_rows)} IKEA rows added/refreshed")


def main():
    new_rows = []
    for mp in MONITORING_POINTS:
        row = process_one(mp)
        new_rows.append(row)

    update_meter_summary(new_rows)

    print("\n=== Done ===")
    for r in new_rows:
        print(f"  {r['meter']}: {r['total_days']} days, avg={r['avg_daily_kwh']} kWh/day")


if __name__ == "__main__":
    main()
