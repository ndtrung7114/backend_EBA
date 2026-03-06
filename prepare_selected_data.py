"""
Prepare Selected Equipment Data for EBA Web App
================================================
Selects 10 SG + 10 ECD equipment based on ElasticNet R² ranking
from 10-model sliding-window checkpoint results:
  - Top 4 best avg ElasticNet R² test
  - Middle 3 avg ElasticNet R² test
  - Bottom 3 worst avg ElasticNet R² test

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

# Selection based on ElasticNet model R² ranking from 10-model checkpoints
SG_SELECTED = [
    # Top 4 highest ElasticNet avg R² (rank 1-4 of 62)
    ('SG_equip_01', 'Mr906J5nel/L4jvvzXgkeUq8NJZAtws=',             'Top1 AvgR²=-0.12  BestR²=0.62'),
    ('SG_equip_02', 'Mr916JoIZV9j5FZgP2ZKW9V3JzAaQHYK',            'Top2 AvgR²=-0.26  BestR²=0.20'),
    ('SG_equip_03', 'UNUFhI8ae1QOd4JRzFFDGuZXlEXD4Dlp',            'Top3 AvgR²=-0.27  BestR²=0.32'),
    ('SG_equip_04', 'Mr8FmI8GAEoiFFOpdvkud6ZinEZ1uiYaOg==',        'Top4 AvgR²=-0.28  BestR²=0.33'),
    # Middle 3 (ranks 31-33 of 62)
    ('SG_equip_05', 'QP0q2bwvOxR80pyUXnWjskMffSDPgBmXZKKP4Q==',    'Mid1 AvgR²=-1.23'),
    ('SG_equip_06', 'Mr916JoIZVYjbrQwtOceO5Bc1WM+CvRgfA==',        'Mid2 AvgR²=-1.30'),
    ('SG_equip_07', 'Mr8FmIsGAEoi272OKl9uiacSeIfTsDD5yg==',         'Mid3 AvgR²=-1.31'),
    # Bottom 3 worst ElasticNet R² (ranks 60-62 of 62)
    ('SG_equip_08', 'Mr926JoIZV9tnSzQf+auEIXNoL7TRAaA',            'Bot1 AvgR²=-1106'),
    ('SG_equip_09', 'Mr926JoIZVCuliz8QRUygPSPI8u1VlYZ',             'Bot2 AvgR²=-1337'),
    ('SG_equip_10', 'Mr8FmI8GAEoj9jCoujLV/nFLDHn15h8qUw==',        'Bot3 AvgR²=-4530'),
]

# Selection based on ElasticNet model R² ranking from 10-model checkpoints
ECD_SELECTED = [
    # Top 4 highest ElasticNet avg R² (rank 1-4 of 81) — positive R²!
    ('ECD_equip_01', 'f0b3224c448327d7cae09bfd641aa0f3ab76319e6da268cb3a',  'Top1 AvgR²=+0.092 BestR²=0.41'),
    ('ECD_equip_02', 'c74ab8448e44008872515b1590d23af05f79befdf31642e9bb',  'Top2 AvgR²=+0.053 BestR²=0.69'),
    ('ECD_equip_03', '06df53341048dd2bad65dc046edf71cb920e33f009b8d78e89',  'Top3 AvgR²=+0.052 BestR²=0.68'),
    ('ECD_equip_04', 'a01058ebafc16c9a798b01d41ecbe32ee2bbb7171ab22db6f2',  'Top4 AvgR²=+0.033 BestR²=0.39'),
    # Middle 3 (ranks 40-42 of 81)
    ('ECD_equip_05', 'c2d5342f494662fc25926fa75aa51d66cd2340e4d3334dac2b',  'Mid1 AvgR²=-0.75'),
    ('ECD_equip_06', 'dd4569416ee60170254e09b87475584eb8bb8c59b350f49e50',  'Mid2 AvgR²=-0.77'),
    ('ECD_equip_07', 'ac77142ab83b60470fc1cd494471a6caf0a29dda18431137f6',  'Mid3 AvgR²=-0.85'),
    # Bottom 3 worst ElasticNet R² (ranks 79-81 of 81)
    ('ECD_equip_08', '37e7971693e8f03d0a7379aa7b884b6091a985e480f40d9ae4',  'Bot1 AvgR²=-28.8'),
    ('ECD_equip_09', '3bdec6d7197a91900ff81511c2b8e13be79497f618909dd99f',  'Bot2 AvgR²=-458'),
    ('ECD_equip_10', '48457b799fca6fde796392b134d47ad0c4232c69461276477d',  'Bot3 AvgR²=-492'),
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
