"""
Test actual ElasticNetCV R² for each selected equipment using the web app pipeline.
Tries multiple period combinations to find what gives good/bad results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from app.loader import load_meter_combined, load_meter_summary
from app.features import get_available_features, remove_outliers_iqr
from app.regression import build_pipeline, train_model, predict, get_model_info
from app.metrics import compute_metrics

def test_equipment(meter_name, tr_start, tr_end, rp_start, rp_end):
    """Run the exact same pipeline as the web app analysis endpoint."""
    df = load_meter_combined(meter_name)
    feat_cols = get_available_features(df)
    
    # Training
    tr_s, tr_e = pd.to_datetime(tr_start), pd.to_datetime(tr_end)
    rp_s, rp_e = pd.to_datetime(rp_start), pd.to_datetime(rp_end)
    
    train_mask = (df.index >= tr_s) & (df.index <= tr_e)
    df_train = df[train_mask].copy()
    
    # IQR outlier removal (same as web app default)
    if len(df_train) > 0:
        df_train, _ = remove_outliers_iqr(df_train, "daily_kwh", 1.5)
    
    if len(df_train) < 5:
        return None, f"Only {len(df_train)} train days"
    
    # Reporting
    rp_mask = (df.index >= rp_s) & (df.index <= rp_e)
    df_report = df[rp_mask].copy()
    if len(df_report) < 1:
        return None, "No reporting data"
    
    # Fit
    X_train = df_train[feat_cols].values
    y_train = df_train["daily_kwh"].values
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    info = get_model_info(pipeline)
    
    y_pred_train = predict(pipeline, X_train)
    train_metrics = compute_metrics(y_train, y_pred_train)
    
    X_report = df_report[feat_cols].values
    y_report = df_report["daily_kwh"].values
    y_pred_report = predict(pipeline, X_report)
    report_metrics = compute_metrics(y_report, y_pred_report)
    
    return {
        'n_train': len(df_train),
        'n_report': len(df_report),
        'R2_train': round(train_metrics['R2'], 4),
        'R2_test': round(report_metrics['R2'], 4),
        'CVRMSE': round(report_metrics['CVRMSE_pct'], 1),
        'NMBE': round(report_metrics['NMBE_pct'], 1),
        'alpha': info.get('best_alpha', '?'),
        'l1_ratio': info.get('best_l1_ratio', '?'),
        'n_features': info.get('n_nonzero_coefs', '?'),
    }, None


# ============================================================================
# Define test periods for each dataset
# ============================================================================

# SG data: 2020-07-06 to 2024-10-17 (~1569 days)
SG_PERIODS = [
    # Annual: train year 1, test year 2
    ("2020-07-06", "2021-07-05", "2021-07-06", "2022-07-05", "Annual Y1→Y2"),
    ("2021-07-06", "2022-07-05", "2022-07-06", "2023-07-05", "Annual Y2→Y3"),
    ("2022-07-06", "2023-07-05", "2023-07-06", "2024-07-05", "Annual Y3→Y4"),
    # All history train, last year test
    ("2020-07-06", "2023-07-05", "2023-07-06", "2024-07-05", "AllHist→LastYr"),
    # Semi-annual: 6-month train, 6-month test
    ("2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", "Semi H1→H2 2023"),
    ("2023-07-01", "2023-12-31", "2024-01-01", "2024-06-30", "Semi H2→H1 2024"),
    # Seasonal match: same season, different year
    ("2021-01-01", "2021-12-31", "2023-01-01", "2023-12-31", "FullYr 21→23"),
    # Short: quarterly
    ("2023-04-01", "2023-06-30", "2023-07-01", "2023-09-30", "Q2→Q3 2023"),
    ("2023-07-01", "2023-09-30", "2023-10-01", "2023-12-31", "Q3→Q4 2023"),
    # User-friendly: 6-month blocks
    ("2022-01-01", "2022-12-31", "2023-01-01", "2023-06-30", "Yr22→H1-23"),
    ("2020-07-06", "2024-01-01", "2024-01-01", "2024-10-17", "AllHist→2024"),
]

# ECD data: 2025-01-01 to 2026-02-04 (~400 days)
ECD_PERIODS = [
    # Simple splits for ~13 months of data
    ("2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31", "H1→H2 2025"),
    ("2025-01-01", "2025-09-30", "2025-10-01", "2026-02-04", "9mo→rest"),
    ("2025-01-01", "2025-08-31", "2025-09-01", "2026-02-04", "8mo→5mo"),
    ("2025-03-01", "2025-08-31", "2025-09-01", "2026-02-04", "Mar-Aug→Sep-Feb"),
    ("2025-04-01", "2025-09-30", "2025-10-01", "2026-02-04", "Apr-Sep→Oct-Feb"),
    ("2025-01-01", "2025-04-30", "2025-05-01", "2025-08-31", "Q1Q2→Q2Q3"),
    ("2025-05-01", "2025-08-31", "2025-09-01", "2025-12-31", "May-Aug→Sep-Dec"),
    ("2025-02-01", "2025-07-31", "2025-08-01", "2026-01-31", "Feb-Jul→Aug-Jan"),
    ("2025-01-01", "2025-12-31", "2025-06-01", "2025-12-31", "FullYr→H2(overlap)"),
]

SG_METERS = [f"SG_equip_{i:02d}" for i in range(1, 11)]
ECD_METERS = [f"ECD_equip_{i:02d}" for i in range(1, 11)]

SG_CATS = ['Top-1','Top-2','Top-3','Top-4','Mid-1','Mid-2','Mid-3','Bot-1','Bot-2','Bot-3']
ECD_CATS = SG_CATS.copy()

# ============================================================================
print("=" * 120)
print("  SAINT-GOBAIN Equipment — ElasticNetCV Pipeline Test")
print("=" * 120)

sg_best = {}
for meter, cat in zip(SG_METERS, SG_CATS):
    print(f"\n{'─'*120}")
    print(f"  {meter} ({cat})")
    print(f"  {'Period':<25} {'Train':>6} {'Test':>5} {'R²Train':>8} {'R²Test':>8} {'CVRMSE':>7} {'NMBE':>7} {'Alpha':>8} {'L1':>5} {'Feats':>5}")
    
    best_r2 = -1e30
    worst_r2 = 1e30
    best_info = None
    worst_info = None
    
    for tr_s, tr_e, rp_s, rp_e, label in SG_PERIODS:
        result, err = test_equipment(meter, tr_s, tr_e, rp_s, rp_e)
        if err:
            print(f"  {label:<25} ERROR: {err}")
            continue
        
        r = result
        print(f"  {label:<25} {r['n_train']:>6} {r['n_report']:>5} {r['R2_train']:>8.4f} {r['R2_test']:>8.4f} {r['CVRMSE']:>6.1f}% {r['NMBE']:>6.1f}% {r['alpha']:>8.4f} {r['l1_ratio']:>5.2f} {r['n_features']:>5}")
        
        if r['R2_test'] > best_r2:
            best_r2 = r['R2_test']
            best_info = (label, tr_s, tr_e, rp_s, rp_e, r)
        if r['R2_test'] < worst_r2:
            worst_r2 = r['R2_test']
            worst_info = (label, tr_s, tr_e, rp_s, rp_e, r)
    
    sg_best[meter] = (cat, best_info, worst_info)
    if best_info:
        b = best_info
        print(f"  ★ BEST:  {b[0]} → R²={b[5]['R2_test']:.4f} | BL/Train: {b[1]}→{b[2]} | Report: {b[3]}→{b[4]}")
    if worst_info:
        w = worst_info
        print(f"  ✗ WORST: {w[0]} → R²={w[5]['R2_test']:.4f} | BL/Train: {w[1]}→{w[2]} | Report: {w[3]}→{w[4]}")


print("\n\n" + "=" * 120)
print("  ECD / EC8469 Equipment — ElasticNetCV Pipeline Test")
print("=" * 120)

ecd_best = {}
for meter, cat in zip(ECD_METERS, ECD_CATS):
    print(f"\n{'─'*120}")
    print(f"  {meter} ({cat})")
    print(f"  {'Period':<25} {'Train':>6} {'Test':>5} {'R²Train':>8} {'R²Test':>8} {'CVRMSE':>7} {'NMBE':>7} {'Alpha':>8} {'L1':>5} {'Feats':>5}")
    
    best_r2 = -1e30
    worst_r2 = 1e30
    best_info = None
    worst_info = None
    
    for tr_s, tr_e, rp_s, rp_e, label in ECD_PERIODS:
        result, err = test_equipment(meter, tr_s, tr_e, rp_s, rp_e)
        if err:
            print(f"  {label:<25} ERROR: {err}")
            continue
        
        r = result
        print(f"  {label:<25} {r['n_train']:>6} {r['n_report']:>5} {r['R2_train']:>8.4f} {r['R2_test']:>8.4f} {r['CVRMSE']:>6.1f}% {r['NMBE']:>6.1f}% {r['alpha']:>8.4f} {r['l1_ratio']:>5.2f} {r['n_features']:>5}")
        
        if r['R2_test'] > best_r2:
            best_r2 = r['R2_test']
            best_info = (label, tr_s, tr_e, rp_s, rp_e, r)
        if r['R2_test'] < worst_r2:
            worst_r2 = r['R2_test']
            worst_info = (label, tr_s, tr_e, rp_s, rp_e, r)
    
    ecd_best[meter] = (cat, best_info, worst_info)
    if best_info:
        b = best_info
        print(f"  ★ BEST:  {b[0]} → R²={b[5]['R2_test']:.4f} | BL/Train: {b[1]}→{b[2]} | Report: {b[3]}→{b[4]}")
    if worst_info:
        w = worst_info
        print(f"  ✗ WORST: {w[0]} → R²={w[5]['R2_test']:.4f} | BL/Train: {w[1]}→{w[2]} | Report: {w[3]}→{w[4]}")


# ============================================================================
# SUMMARY: Recommended settings for each equipment
# ============================================================================
print("\n\n" + "=" * 120)
print("  RECOMMENDED SETTINGS FOR WEB APP USERS")
print("=" * 120)

print("\n--- SAINT-GOBAIN (SG) ---")
print(f"{'Meter':<15} {'Cat':<7} {'R²Best':>8} {'BL/Train Start':>14} {'BL/Train End':>12} {'Report Start':>13} {'Report End':>12} {'Period Label'}")
for meter in SG_METERS:
    cat, best, worst = sg_best[meter]
    if best:
        b = best
        print(f"{meter:<15} {cat:<7} {b[5]['R2_test']:>8.4f} {b[1]:>14} {b[2]:>12} {b[3]:>13} {b[4]:>12} {b[0]}")

print("\n--- ECD / EC8469 ---")
print(f"{'Meter':<15} {'Cat':<7} {'R²Best':>8} {'BL/Train Start':>14} {'BL/Train End':>12} {'Report Start':>13} {'Report End':>12} {'Period Label'}")
for meter in ECD_METERS:
    cat, best, worst = ecd_best[meter]
    if best:
        b = best
        print(f"{meter:<15} {cat:<7} {b[5]['R2_test']:>8.4f} {b[1]:>14} {b[2]:>12} {b[3]:>13} {b[4]:>12} {b[0]}")
