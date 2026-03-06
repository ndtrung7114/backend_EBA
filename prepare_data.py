"""
Prepare SG and ECD data for the web app.
Creates per-equipment CSVs with weather + temporal features in data/sg/ and data/ecd/.
Updates meter_summary.csv with group column.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import holidays

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ═════════════════════════════════════════════════════════════════════════
# HELPER: add temporal features
# ═════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame, country: str = "CN") -> pd.DataFrame:
    """Add temporal features matching the BG format."""
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df["month"] = idx.month
    df["month_day"] = idx.day
    df["week_day"] = idx.dayofweek
    df["season"] = idx.month.map(lambda m: {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[m])
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    
    try:
        cn_holidays = holidays.China(years=range(idx.year.min(), idx.year.max() + 1))
        df["is_holiday"] = idx.map(lambda d: 1 if d in cn_holidays else 0)
    except Exception:
        df["is_holiday"] = 0
    return df


# ═════════════════════════════════════════════════════════════════════════
# SAINT-GOBAIN: merge energy + weather, pick top 10
# ═════════════════════════════════════════════════════════════════════════

def prepare_sg():
    print("═══ Preparing Saint-Gobain data ═══")
    sg_energy = pd.read_csv(r"I:\EBA_Saint_Goban_Comparison\data\energy_all_equipment.csv")
    sg_weather = pd.read_csv(r"I:\EBA_Saint_Goban_Comparison\data\weather_shanghai.csv")
    
    sg_energy["date"] = pd.to_datetime(sg_energy["date"])
    sg_weather["date"] = pd.to_datetime(sg_weather["date"])
    
    # Pick top 10 equipment by total kWh, with good data quality
    equip_stats = sg_energy.groupby("equipment_code").agg(
        n_days=("daily_kwh", "count"),
        total_kwh=("daily_kwh", "sum"),
        avg_daily=("daily_kwh", "mean"),
        zero_days=("daily_kwh", lambda x: (x == 0).sum()),
    ).sort_values("total_kwh", ascending=False)
    
    good = equip_stats[
        (equip_stats.n_days >= 500) & 
        (equip_stats.avg_daily > 50) & 
        (equip_stats.zero_days / equip_stats.n_days < 0.15)
    ]
    top10 = good.head(10).index.tolist()
    print(f"  Selected {len(top10)} equipment")
    
    sg_dir = DATA / "sg"
    sg_dir.mkdir(exist_ok=True)
    
    weather_cols = [
        "date", "maxtempC", "mintempC", "avgtempC", "humidity", "sunHour",
        "uvIndex", "windspeedKmph", "pressure", "winddirDegree", "visibility",
        "cloudcover", "HeatIndexC", "WindChillC", "WindGustKmph", "FeelsLikeC",
    ]
    w = sg_weather[weather_cols].copy()
    
    summary_rows = []
    for i, code in enumerate(top10, 1):
        name = f"SG_equip_{i:02d}"
        equip_data = sg_energy[sg_energy["equipment_code"] == code][["date", "daily_kwh"]].copy()
        equip_data = equip_data.sort_values("date").drop_duplicates("date")
        
        # Merge with weather
        merged = equip_data.merge(w, on="date", how="inner")
        merged = merged.set_index("date").sort_index()
        
        # Add temporal features
        merged = add_temporal_features(merged, "CN")
        
        # Save
        out_path = sg_dir / f"{name}.csv"
        merged.to_csv(out_path)
        
        summary_rows.append({
            "meter": name,
            "group": "Saint-Gobain",
            "site": "Shanghai",
            "building_type": "industrial",
            "total_days": len(merged),
            "min_date": str(merged.index.min().date()),
            "max_date": str(merged.index.max().date()),
            "avg_daily_kwh": round(merged["daily_kwh"].mean(), 1),
        })
        print(f"  {name}: {len(merged)} days, avg={merged['daily_kwh'].mean():.1f} kWh/day")
    
    return summary_rows


# ═════════════════════════════════════════════════════════════════════════
# ECD: copy top 10 equipment from site_ec8469 dataset
# ═════════════════════════════════════════════════════════════════════════

def prepare_ecd():
    print("\n═══ Preparing ECD data ═══")
    ecd_src = Path(r"I:\EBA_data\ecd_uat_output\site_ec8469_dataset\dataset")
    
    # Get stats for all equipment
    stats = []
    for f in ecd_src.glob("*.csv"):
        code = f.stem
        df = pd.read_csv(f)
        stats.append({
            "code": code,
            "file": f,
            "days": len(df),
            "total": df["usage_value"].sum(),
            "avg": df["usage_value"].mean(),
            "zeros": (df["usage_value"] == 0).sum(),
        })
    
    stats_df = pd.DataFrame(stats).sort_values("total", ascending=False)
    good = stats_df[(stats_df["avg"] > 100) & (stats_df["zeros"] / stats_df["days"] < 0.15)]
    top10 = good.head(10)
    print(f"  Selected {len(top10)} equipment")
    
    ecd_dir = DATA / "ecd"
    ecd_dir.mkdir(exist_ok=True)
    
    summary_rows = []
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        name = f"ECD_equip_{i:02d}"
        df = pd.read_csv(row["file"], parse_dates=["date"], index_col="date")
        df = df.rename(columns={"usage_value": "daily_kwh"})
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        
        out_path = ecd_dir / f"{name}.csv"
        df.to_csv(out_path)
        
        summary_rows.append({
            "meter": name,
            "group": "ECD",
            "site": "ec8469",
            "building_type": "commercial",
            "total_days": len(df),
            "min_date": str(df.index.min().date()),
            "max_date": str(df.index.max().date()),
            "avg_daily_kwh": round(df["daily_kwh"].mean(), 1),
        })
        print(f"  {name}: {len(df)} days, avg={df['daily_kwh'].mean():.1f} kWh/day")
    
    return summary_rows


# ═════════════════════════════════════════════════════════════════════════
# BUILD COMBINED meter_summary.csv
# ═════════════════════════════════════════════════════════════════════════

def build_summary():
    print("\n═══ Building combined meter_summary.csv ═══")
    
    # Existing BG data
    old_summary = pd.read_csv(DATA / "meter_summary.csv")
    bg_rows = []
    for _, row in old_summary.iterrows():
        bg_rows.append({
            "meter": row["meter"],
            "group": "Building Genome",
            "site": row["site"],
            "building_type": row["building_type"],
            "total_days": int(row["total_days"]),
            "min_date": "",  # will be filled from data
            "max_date": "",
            "avg_daily_kwh": 0,
        })
    
    # Fill BG date ranges from actual data
    for r in bg_rows:
        try:
            train = pd.read_csv(DATA / "training" / f"{r['meter']}_train.csv")
            test = pd.read_csv(DATA / "testing" / f"{r['meter']}_test.csv")
            all_dates = list(train["date"]) + list(test["date"])
            all_dates = sorted(all_dates)
            r["min_date"] = all_dates[0]
            r["max_date"] = all_dates[-1]
            all_vals = list(train["usage_value"]) + list(test["usage_value"])
            r["avg_daily_kwh"] = round(sum(all_vals) / len(all_vals), 1)
        except Exception:
            pass
    
    return bg_rows


def main():
    bg_rows = build_summary()
    sg_rows = prepare_sg()
    ecd_rows = prepare_ecd()
    
    all_rows = bg_rows + sg_rows + ecd_rows
    summary = pd.DataFrame(all_rows)
    
    # Save new summary
    summary.to_csv(DATA / "meter_summary.csv", index=False)
    print(f"\n✓ Saved meter_summary.csv with {len(summary)} entries")
    print(f"  Building Genome: {len(bg_rows)}")
    print(f"  Saint-Gobain: {len(sg_rows)}")
    print(f"  ECD: {len(ecd_rows)}")


if __name__ == "__main__":
    main()
