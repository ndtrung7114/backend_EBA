"""
Prepare Selected Equipment Data for EBA Web App
================================================
Selects 10 SG + 10 ECD equipment based on R² ranking:
  - Top 4 best R² test
  - Middle 3 R² test
  - Bottom 3 worst R² test

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

SG_SELECTED = [
    # Top 4 highest avg_R2 (least negative)
    ('SG_equip_01', 'UNUFhI8GeUNaKPwMG/KRZ3NyoXmceg4=',       'Top1  R²=-1.70'),
    ('SG_equip_02', 'QsJ2mP/0h/ALCIZcSvC2/xXmkZCa',           'Top2  R²=-2.13'),
    ('SG_equip_03', 'Mr8FmI8eCkohC+kH2zPgcHqcDAXLac4t7g==',   'Top3  R²=-2.13'),
    ('SG_equip_04', 'QsJ1mP1neewDrfK9GuvQD1TGtsLbQxw=',       'Top4  R²=-2.32'),
    # Middle 3 (ranks 31–33 of 64)
    ('SG_equip_05', 'Mr8FmI8aAEognVogG/mqpvI37ocxcGIaQg==',   'Mid1  R²=-6.33e3'),
    ('SG_equip_06', 'QPoyyqYvJkoiLVTsCtU5t+TkEpxG8JWrQQ==',  'Mid2  R²=-6.42e3'),
    ('SG_equip_07', 'Mr926JoIZVKilbVrtgac09TwI3PHNoUc',       'Mid3  R²=-9.79e3'),
    # Bottom 3 worst R²
    ('SG_equip_08', 'S/cm3ackL0oiYPqkl4j8LG0SOoWdeWbOTg==',  'Bot1  R²≈-2.9e72'),
    ('SG_equip_09', 'SMYG+f/tGwSDIHuNOAV1vx7GxyJq',          'Bot2  R²≈-5.4e203'),
    ('SG_equip_10', 'YPouxaIvOkohD+UZ0NeAfKVmlzaitET4OQ==',   'Bot3  R²≈-3.9e267'),
]

ECD_SELECTED = [
    # Top 4 highest avg_R2 (valid, non-zero-consumption)
    ('ECD_equip_01', 'a01058ebafc16c9a798b01d41ecbe32ee2bbb7171ab22db6f2',  'Top1  R²=-0.028'),
    ('ECD_equip_02', '0e19f704ebdc23040972dc144eed1c0f80c6d22ebef98fc87d',  'Top2  R²=-0.153'),
    ('ECD_equip_03', 'ed8c600d70262f56a9e5118cf3b745a0df27f5d6f5ca6aa241',  'Top3  R²=-0.224'),
    ('ECD_equip_04', 'fab4a83b7fddff86136820b1c06ba6430374c096f6f6bd2aad',  'Top4  R²=-0.245'),
    # Middle 3 (ranks ~40–42 of 83 valid entries)
    ('ECD_equip_05', '40f4de4301f41d686c5d49305c2d68e62bab9bc5d05e5ccba4', 'Mid1  R²=-0.872'),
    ('ECD_equip_06', '60ee639d12235d49655ffb17a54686b3f42ec3a17e72c65bcf', 'Mid2  R²=-0.964'),
    ('ECD_equip_07', 'eebd3bf70f9a9ae88023093a305bbbd48866113ad8311107f3', 'Mid3  R²=-0.966'),
    # Bottom 3 worst R²
    ('ECD_equip_08', '3bdec6d7197a91900ff81511c2b8e13be79497f618909dd99f', 'Bot1  R²≈-3.2e5'),
    ('ECD_equip_09', '48457b799fca6fde796392b134d47ad0c4232c69461276477d', 'Bot2  R²≈-3.6e5'),
    ('ECD_equip_10', '9792671c468bf50985778c6ecdb02409f20be1028c2f7cea49', 'Bot3  R²≈-3.8e6'),
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
