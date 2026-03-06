"""
Prepare Selected Equipment Data for EBA Web App
================================================
Selects 10 SG + 10 ECD equipment based on ElasticNetCV R² (test)
ranked by the actual web app pipeline across 3 period combinations:
  - Top 4 best R² test
  - Middle 3 (median-centered) R² test
  - Bottom 3 worst R² test

Full scan: 62 SG + 87 ECD equipment tested (scan_all_equipment.py).

Saves outputs to:
  backend/data/sg/SG_equip_01..10.csv
  backend/data/ecd/ECD_equip_01..10.csv
  backend/data/meter_summary.csv  (updates SG + ECD entries)
"""

import sys
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BACKEND_DIR   = Path(__file__).resolve().parent
DATA_DIR      = BACKEND_DIR / 'data'
SG_OUT_DIR    = DATA_DIR / 'sg'
ECD_OUT_DIR   = DATA_DIR / 'ecd'
SUMMARY_CSV   = DATA_DIR / 'meter_summary.csv'

SG_ENERGY_CSV  = Path(r'I:\EBA_Saint_Goban_Comparison\data\energy_all_equipment.csv')
SG_WEATHER_CSV = Path(r'I:\EBA_Saint_Goban_Comparison\data\weather_shanghai.csv')
ECD_DATASET    = Path(r'I:\EBA_data\ecd_uat_output\site_ec8469_dataset\dataset')

# ============================================================================
# SELECTED EQUIPMENT CODES
# Format: (meter_name, equipment_code, label)
# ============================================================================

# Selection based on web app ElasticNetCV R² (scan_all_equipment.py — 62 equip × 3 periods)
SG_SELECTED = [
    # Top 4 highest R² test (rank 1-4 of 61 valid)
    ('SG_equip_01', 'QPouxaIvOkoiiUfrjgyq6+xMkGmnoM9RKw==',        'Top1 R²=0.59  Annual Y2→Y3'),
    ('SG_equip_02', 'Mr8FmI8eCkoi70wFjm/s6gPmvEVLjYV9VA==',        'Top2 R²=0.45  AllHist→2024'),
    ('SG_equip_03', 'Mr906J5nel/L4jvvzXgkeUq8NJZAtws=',             'Top3 R²=0.33  Semi H2→H1 2024'),
    ('SG_equip_04', 'S/cm3ackL0oiYPqkl4j8LG0SOoWdeWbOTg==',        'Top4 R²=0.33  AllHist→2024'),
    # Middle 3 (ranks 30-32 of 61)
    ('SG_equip_05', 'UNUFhI8ae1Xpdk6aN9iUM3ZoFBn60FDr',             'Mid1 R²=-0.0015  Annual Y2→Y3'),
    ('SG_equip_06', 'QsJ0mON79zG0L/z65T0o7l/nGv33Wg==',             'Mid2 R²=-0.0019  Annual Y2→Y3'),
    ('SG_equip_07', 'Mr916JoIZVNL7QMQDQ+vsRR1JKE3ApwU',             'Mid3 R²=-0.0029  Annual Y2→Y3'),
    # Bottom 3 worst R² test (ranks 59-61 of 61)
    ('SG_equip_08', 'UNUFhI8ae1YCdBvGO0LtScYolP2xgOps',             'Bot1 R²=-0.27  Annual Y2→Y3'),
    ('SG_equip_09', 'UNUFhI8GeUNaKPwMG/KRZ3NyoXmceg4=',             'Bot2 R²=-0.29  AllHist→2024'),
    ('SG_equip_10', 'YPouxaIvOkohD+UZ0NeAfKVmlzaitET4OQ==',          'Bot3 R²=-0.50  AllHist→2024'),
]

# Selection based on web app ElasticNetCV R² (scan_all_equipment.py — 87 equip × 3 periods)
ECD_SELECTED = [
    # Top 4 highest R² test (rank 1-4 of 82 valid)
    ('ECD_equip_01', 'a5f3e9e815ee1756279bdccabde64b038652dfbd470da49fa0',  'Top1 R²=0.63  Apr-Sep→Oct-Feb'),
    ('ECD_equip_02', 'b3c1f0af7dfe91a916e3cd04a068accb47592d3589d15a6846',  'Top2 R²=0.59  8mo→5mo'),
    ('ECD_equip_03', 'fe7251e333d36e468a93d28d6af637304301c27d4ed8a09c5e',  'Top3 R²=0.58  Apr-Sep→Oct-Feb'),
    ('ECD_equip_04', '0e695d012b29168cd7fdfe28aedf2776369d445773524ef966',  'Top4 R²=0.56  Apr-Sep→Oct-Feb'),
    # Middle 3 (ranks 40-42 of 82)
    ('ECD_equip_05', 'fab4a83b7fddff86136820b1c06ba6430374c096f6f6bd2aad',  'Mid1 R²=-0.0030  H1→H2 2025'),
    ('ECD_equip_06', 'd2d7289925559aacfe8c8ad732cc9347071bb11eeabfcdc13d',  'Mid2 R²=-0.0035  Apr-Sep→Oct-Feb'),
    ('ECD_equip_07', '42aa979c14d145eecfe58f94b9879357bb1ee208bd04c0156e',  'Mid3 R²=-0.0036  H1→H2 2025'),
    # Bottom 3 worst R² test (ranks 80-82 of 82)
    ('ECD_equip_08', 'd74c6a1587806b5cf1333ac3c211c71d382a2fb0ddc97a0eff',  'Bot1 R²=-2.79  H1→H2 2025'),
    ('ECD_equip_09', 'e09cc78dee560fb8ed19d1fba22f38a7482ef46ac5cce2e694',  'Bot2 R²=-3.30  Apr-Sep→Oct-Feb'),
    ('ECD_equip_10', '98f7b6b5f30423715ad6218123db57a76cf11debfef04185da',  'Bot3 R²=-4.32  8mo→5mo'),
]

# ============================================================================
# HELPERS
# ============================================================================

def get_season(month: int) -> int:
    if month in [12, 1, 2]: return 1
    if month in [3, 4, 5]:  return 2
    if month in [6, 7, 8]:  return 3
    return 4


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, month_day, week_day, season, is_weekend, is_holiday columns."""
    df['month']      = df.index.month
    df['month_day']  = df.index.day
    df['week_day']   = df.index.dayofweek
    df['season']     = df['month'].apply(get_season)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    years = df.index.year.unique()
    cn_holidays = holidays.China(years=years.tolist())
    df['is_holiday'] = df.index.to_series().apply(
        lambda d: 1 if d in cn_holidays else 0
    )
    return df


# ============================================================================
# PROCESS SG EQUIPMENT
# ============================================================================
print("=== Processing SG Equipment ===")
print("Loading SG energy data (long format)...")
df_energy = pd.read_csv(SG_ENERGY_CSV, parse_dates=['date'])
df_energy['date'] = pd.to_datetime(df_energy['date'])

print("Loading SG weather data...")
df_weather = pd.read_csv(SG_WEATHER_CSV, parse_dates=['date'], index_col='date')

SG_OUT_DIR.mkdir(parents=True, exist_ok=True)
sg_rows = []

for meter_name, equip_code, label in SG_SELECTED:
    print(f"  {meter_name}: {equip_code[:30]}... ({label})")

    # Filter this equipment's rows
    df_eq = df_energy[df_energy['equipment_code'] == equip_code].copy()
    if df_eq.empty:
        print(f"    WARNING: equipment_code not found in energy file! Skipping.")
        continue

    df_eq = df_eq.set_index('date')[['daily_kwh']].sort_index()
    df_eq = df_eq[~df_eq.index.duplicated(keep='first')]

    # Join with weather
    df_out = df_eq.join(df_weather, how='inner')
    df_out = df_out.dropna(subset=['daily_kwh'])

    # Add temporal features
    df_out = add_temporal(df_out)

    # Save
    out_path = SG_OUT_DIR / f'{meter_name}.csv'
    df_out.to_csv(out_path)
    print(f"    Saved {len(df_out)} rows → {out_path.name}")

    # Collect summary row
    sg_rows.append({
        'meter': meter_name,
        'group': 'Saint-Gobain',
        'site': 'Shanghai',
        'building_type': 'Factory',
        'total_days': len(df_out),
        'min_date': str(df_out.index.min().date()),
        'max_date': str(df_out.index.max().date()),
        'avg_daily_kwh': round(float(df_out['daily_kwh'].mean()), 2),
    })

# ============================================================================
# PROCESS ECD EQUIPMENT
# ============================================================================
print("\n=== Processing ECD Equipment ===")
ECD_OUT_DIR.mkdir(parents=True, exist_ok=True)
ecd_rows = []

for meter_name, equip_code, label in ECD_SELECTED:
    print(f"  {meter_name}: {equip_code[:30]}... ({label})")

    src_path = ECD_DATASET / f'{equip_code}.csv'
    if not src_path.exists():
        print(f"    WARNING: source file not found! Skipping.")
        continue

    df_eq = pd.read_csv(src_path, parse_dates=['date'], index_col='date')
    df_eq = df_eq.rename(columns={'usage_value': 'daily_kwh'})
    df_eq = df_eq[~df_eq.index.duplicated(keep='first')].sort_index()

    # Save (already has weather + temporal features)
    out_path = ECD_OUT_DIR / f'{meter_name}.csv'
    df_eq.to_csv(out_path)
    print(f"    Saved {len(df_eq)} rows → {out_path.name}")

    ecd_rows.append({
        'meter': meter_name,
        'group': 'ECD',
        'site': 'EC8469',
        'building_type': 'Commercial',
        'total_days': len(df_eq),
        'min_date': str(df_eq.index.min().date()),
        'max_date': str(df_eq.index.max().date()),
        'avg_daily_kwh': round(float(df_eq['daily_kwh'].mean()), 2),
    })

# ============================================================================
# UPDATE meter_summary.csv
# ============================================================================
print("\n=== Updating meter_summary.csv ===")
existing = pd.read_csv(SUMMARY_CSV)

# Keep only Building Genome rows (drop old SG_ and ECD_ rows)
bg_rows = existing[existing['group'] == 'Building Genome'].copy()

new_sg  = pd.DataFrame(sg_rows)
new_ecd = pd.DataFrame(ecd_rows)

combined = pd.concat([bg_rows, new_sg, new_ecd], ignore_index=True)
combined.to_csv(SUMMARY_CSV, index=False)
print(f"  Written {len(combined)} rows to {SUMMARY_CSV}")
print(f"  BG: {len(bg_rows)}, SG: {len(new_sg)}, ECD: {len(new_ecd)}")

print("\nDONE.")
