"""
Analyze ElasticNet R² across all equipment checkpoints
=======================================================
For each dataset (SG and ECD):
  1. Load all checkpoint JSONs
  2. Filter ElasticNet results only
  3. Compute average R² test per equipment
  4. Rank and select: Top 4, Middle 3, Bottom 3
  5. For each selected equipment, find the split/period that produced
     the BEST and WORST R² test
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================================
SG_CHECKPOINTS  = Path(r'I:\EBA_SG_10_Models\output_all\checkpoints')
ECD_CHECKPOINTS = Path(r'I:\EBA_EC8469_10_Models\output_all\checkpoints')
MODEL_FILTER    = 'ElasticNet'


def load_all_elasticnet(checkpoint_dir: Path) -> pd.DataFrame:
    """Load all checkpoint JSONs, extract ElasticNet results into a DataFrame."""
    rows = []
    for jf in sorted(checkpoint_dir.glob('*.json')):
        data = json.load(open(jf, encoding='utf-8'))
        for r in data['results']:
            if r['model'] != MODEL_FILTER:
                continue
            rows.append({
                'equipment_code': r['equipment_code'],
                'strategy':       r['strategy'],
                'split_id':       r['split_id'],
                'train_start':    r['train_start'],
                'train_end':      r['train_end'],
                'test_start':     r['test_start'],
                'test_end':       r['test_end'],
                'n_train':        r['n_train'],
                'n_test':         r['n_test'],
                'R2':             r['metrics_test']['R2'],
                'CVRMSE_pct':     r['metrics_test']['CVRMSE_pct'],
                'NMBE_pct':       r['metrics_test']['NMBE_pct'],
                'Mean_Actual':    r['metrics_test']['Mean_Actual'],
            })
    return pd.DataFrame(rows)


def select_equipment(df: pd.DataFrame, label: str):
    """Rank equipment by avg R² and select Top4 + Mid3 + Bot3."""
    # Average R2 per equipment
    avg = df.groupby('equipment_code')['R2'].mean().reset_index()
    avg.columns = ['equipment_code', 'avg_R2']
    avg = avg.sort_values('avg_R2', ascending=False).reset_index(drop=True)
    avg['rank'] = range(1, len(avg) + 1)

    n = len(avg)
    mid_center = n // 2
    top_idx  = list(range(0, 4))
    mid_idx  = list(range(mid_center - 1, mid_center + 2))
    bot_idx  = list(range(n - 3, n))

    selected_idx = top_idx + mid_idx + bot_idx
    selected = avg.iloc[selected_idx].copy()
    cats = ['Top-1','Top-2','Top-3','Top-4','Mid-1','Mid-2','Mid-3','Bot-1','Bot-2','Bot-3']
    selected['category'] = cats

    print(f"\n{'='*100}")
    print(f"  {label} — ElasticNet R² Ranking  ({n} total equipment)")
    print(f"{'='*100}")
    print(f"{'Cat':<7} {'Rank':>4}  {'Avg R²':>12}  Equipment Code")
    print(f"{'-'*7} {'-'*4}  {'-'*12}  {'-'*50}")
    for _, row in selected.iterrows():
        print(f"{row['category']:<7} {row['rank']:>4}  {row['avg_R2']:>12.6f}  {row['equipment_code']}")

    # For each selected equipment, find BEST and WORST individual split
    print(f"\n{'='*100}")
    print(f"  {label} — Best & Worst Splits per Selected Equipment")
    print(f"{'='*100}")

    results_detail = []
    for _, eq_row in selected.iterrows():
        code = eq_row['equipment_code']
        cat  = eq_row['category']
        rank = eq_row['rank']

        eq_df = df[df['equipment_code'] == code].copy()

        best_idx = eq_df['R2'].idxmax()
        worst_idx = eq_df['R2'].idxmin()
        best = eq_df.loc[best_idx]
        worst = eq_df.loc[worst_idx]

        print(f"\n  [{cat}] Rank {rank} | Avg R²={eq_row['avg_R2']:.6f} | code={code[:40]}...")
        print(f"    BEST  R²={best['R2']:>10.4f} | Strategy={best['strategy']:<12} "
              f"| Train: {best['train_start']} → {best['train_end']} "
              f"| Test:  {best['test_start']} → {best['test_end']} "
              f"| CVRMSE={best['CVRMSE_pct']:.1f}% NMBE={best['NMBE_pct']:.1f}%")
        print(f"    WORST R²={worst['R2']:>10.4f} | Strategy={worst['strategy']:<12} "
              f"| Train: {worst['train_start']} → {worst['train_end']} "
              f"| Test:  {worst['test_start']} → {worst['test_end']} "
              f"| CVRMSE={worst['CVRMSE_pct']:.1f}% NMBE={worst['NMBE_pct']:.1f}%")

        results_detail.append({
            'category': cat, 'rank': rank,
            'equipment_code': code,
            'avg_R2': eq_row['avg_R2'],
            'avg_daily_kwh': eq_df['Mean_Actual'].mean(),
            'n_splits': len(eq_df),
            'best_R2': best['R2'],
            'best_strategy': best['strategy'],
            'best_train': f"{best['train_start']} → {best['train_end']}",
            'best_test':  f"{best['test_start']} → {best['test_end']}",
            'best_CVRMSE': best['CVRMSE_pct'],
            'best_NMBE': best['NMBE_pct'],
            'worst_R2': worst['R2'],
            'worst_strategy': worst['strategy'],
            'worst_train': f"{worst['train_start']} → {worst['train_end']}",
            'worst_test':  f"{worst['test_start']} → {worst['test_end']}",
            'worst_CVRMSE': worst['CVRMSE_pct'],
            'worst_NMBE': worst['NMBE_pct'],
        })

    # Strategy breakdown for selected equipment
    print(f"\n{'='*100}")
    print(f"  {label} — R² by Strategy (selected equipment)")
    print(f"{'='*100}")
    print(f"{'Cat':<7} {'Equipment':<42} {'Annual':>10} {'Semi-ann':>10} {'Quarterly':>10}")
    print(f"{'-'*7} {'-'*42} {'-'*10} {'-'*10} {'-'*10}")
    for _, eq_row in selected.iterrows():
        code = eq_row['equipment_code']
        cat  = eq_row['category']
        eq_df = df[df['equipment_code'] == code]
        strat_avg = eq_df.groupby('strategy')['R2'].mean()
        ann = strat_avg.get('Annual', float('nan'))
        semi = strat_avg.get('Semi-annual', float('nan'))
        qtr = strat_avg.get('Quarterly', float('nan'))
        print(f"{cat:<7} {code[:42]:<42} {ann:>10.4f} {semi:>10.4f} {qtr:>10.4f}")

    return selected, pd.DataFrame(results_detail)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("Loading SG checkpoints...")
    sg_df = load_all_elasticnet(SG_CHECKPOINTS)
    print(f"  Loaded {len(sg_df)} ElasticNet results from {sg_df['equipment_code'].nunique()} equipment")
    sg_sel, sg_detail = select_equipment(sg_df, "SAINT-GOBAIN (SG)")

    print("\n\n" + "#"*100)
    print("#"*100 + "\n")

    print("Loading ECD checkpoints...")
    ecd_df = load_all_elasticnet(ECD_CHECKPOINTS)
    print(f"  Loaded {len(ecd_df)} ElasticNet results from {ecd_df['equipment_code'].nunique()} equipment")
    ecd_sel, ecd_detail = select_equipment(ecd_df, "ECD (EC8469)")

    # Print summary table for easy copy
    print("\n\n" + "="*100)
    print("  FINAL SELECTION SUMMARY")
    print("="*100)
    for tag, detail in [("SG", sg_detail), ("ECD", ecd_detail)]:
        print(f"\n--- {tag} ---")
        print(f"{'#':<5} {'Cat':<7} {'Avg R²':>12} {'Best R²':>10} {'Worst R²':>12} {'Avg kWh/d':>10}  Equipment")
        for i, (_, r) in enumerate(detail.iterrows(), 1):
            print(f"{tag}_equip_{i:02d}  {r['category']:<7} {r['avg_R2']:>12.4f} {r['best_R2']:>10.4f} {r['worst_R2']:>12.4f} {r['avg_daily_kwh']:>10.1f}  {r['equipment_code'][:50]}")
