"""
Scan ALL equipment from SG (62) and ECD (99) datasets using the web app's
ElasticNetCV pipeline with 3 representative period combinations each.
Rank by best R² (test) across periods → select Top4 / Mid3 / Bot3.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from app.features import get_available_features, remove_outliers_iqr
from app.regression import build_pipeline, train_model, predict, get_model_info
from app.metrics import compute_metrics
from app.config import ALL_FEATURES

# ============================================================================
# Data paths
# ============================================================================
SG_ENERGY = Path(r"I:\EBA_Saint_Goban_Comparison\data\energy_all_equipment.csv")
SG_WEATHER = Path(r"I:\EBA_Saint_Goban_Comparison\data\weather_shanghai.csv")
ECD_DATASET = Path(r"I:\EBA_data\ecd_uat_output\site_ec8469_dataset\dataset")
ECD_SUMMARY = ECD_DATASET.parent / "dataset_summary.csv"

# ============================================================================
# Period definitions (3 each)
# ============================================================================
SG_PERIODS = [
    # Annual Y2→Y3: covers middle of dataset
    ("2021-07-06", "2022-07-05", "2022-07-06", "2023-07-05", "Annual Y2→Y3"),
    # Semi H2→H1 2024: most recent data
    ("2023-07-01", "2023-12-31", "2024-01-01", "2024-06-30", "Semi H2→H1 2024"),
    # AllHist→2024: maximum training, newest test
    ("2020-07-06", "2024-01-01", "2024-01-01", "2024-10-17", "AllHist→2024"),
]

ECD_PERIODS = [
    # H1→H2 2025
    ("2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31", "H1→H2 2025"),
    # 8mo train → 5mo test
    ("2025-01-01", "2025-08-31", "2025-09-01", "2026-02-04", "8mo→5mo"),
    # Apr-Sep train → Oct-Feb test
    ("2025-04-01", "2025-09-30", "2025-10-01", "2026-02-04", "Apr-Sep→Oct-Feb"),
]

# ============================================================================
# Temporal features (same logic as prepare_selected_data.py)
# ============================================================================
CN_HOLIDAYS = [
    # 2020-2024 (SG)
    "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-04", "2020-10-05",
    "2020-10-06", "2020-10-07", "2020-10-08",
    "2021-01-01", "2021-01-02", "2021-01-03",
    "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15",
    "2021-02-16", "2021-02-17",
    "2021-04-03", "2021-04-04", "2021-04-05",
    "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", "2021-05-05",
    "2021-06-12", "2021-06-13", "2021-06-14",
    "2021-09-19", "2021-09-20", "2021-09-21",
    "2021-10-01", "2021-10-02", "2021-10-03", "2021-10-04", "2021-10-05",
    "2021-10-06", "2021-10-07",
    "2022-01-01", "2022-01-02", "2022-01-03",
    "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04",
    "2022-02-05", "2022-02-06",
    "2022-04-03", "2022-04-04", "2022-04-05",
    "2022-04-30", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04",
    "2022-06-03", "2022-06-04", "2022-06-05",
    "2022-09-10", "2022-09-11", "2022-09-12",
    "2022-10-01", "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05",
    "2022-10-06", "2022-10-07",
    "2023-01-01", "2023-01-02",
    "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25",
    "2023-01-26", "2023-01-27",
    "2023-04-05",
    "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
    "2023-06-22", "2023-06-23", "2023-06-24",
    "2023-09-29", "2023-09-30",
    "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05",
    "2023-10-06",
    "2024-01-01",
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-15", "2024-02-16", "2024-02-17",
    "2024-04-04", "2024-04-05", "2024-04-06",
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
    "2024-06-08", "2024-06-09", "2024-06-10",
    "2024-09-15", "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05",
    "2024-10-06", "2024-10-07",
    # 2025-2026 (ECD)
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01",
    "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2025-05-31", "2025-06-01", "2025-06-02",
    "2025-09-07", "2025-09-08", "2025-09-09",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05",
    "2025-10-06", "2025-10-07",
    "2026-01-01",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21",
    "2026-02-22", "2026-02-23",
]
CN_HOLIDAYS_DT = set(pd.to_datetime(list(CN_HOLIDAYS)))


def add_temporal_features(df):
    """Add temporal features matching web app expectations."""
    idx = df.index
    df["month"] = idx.month
    df["month_day"] = idx.day
    df["week_day"] = idx.dayofweek
    df["season"] = idx.month.map(lambda m: {12:0,1:0,2:0,3:1,4:1,5:1,
                                              6:2,7:2,8:2,9:3,10:3,11:3}.get(m,0))
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = idx.isin(CN_HOLIDAYS_DT).astype(int)
    return df


# ============================================================================
# Core test function
# ============================================================================
def test_one(df, tr_start, tr_end, rp_start, rp_end):
    """Run ElasticNetCV pipeline on one equipment + one period. Returns dict or None."""
    feat_cols = get_available_features(df)
    tr_s, tr_e = pd.to_datetime(tr_start), pd.to_datetime(tr_end)
    rp_s, rp_e = pd.to_datetime(rp_start), pd.to_datetime(rp_end)

    df_train = df[(df.index >= tr_s) & (df.index <= tr_e)].copy()
    if len(df_train) > 0:
        df_train, _ = remove_outliers_iqr(df_train, "daily_kwh", 1.5)
    if len(df_train) < 10:
        return None

    df_report = df[(df.index >= rp_s) & (df.index <= rp_e)].copy()
    if len(df_report) < 5:
        return None

    X_train = df_train[feat_cols].values
    y_train = df_train["daily_kwh"].values
    X_report = df_report[feat_cols].values
    y_report = df_report["daily_kwh"].values

    try:
        pipeline = build_pipeline()
        train_model(pipeline, X_train, y_train)
        info = get_model_info(pipeline)

        y_hat_tr = predict(pipeline, X_train)
        y_hat_rp = predict(pipeline, X_report)
        m_tr = compute_metrics(y_train, y_hat_tr)
        m_rp = compute_metrics(y_report, y_hat_rp)

        return {
            "n_train": len(df_train),
            "n_test": len(df_report),
            "R2_train": round(m_tr["R2"], 4),
            "R2_test": round(m_rp["R2"], 4),
            "CVRMSE": round(m_rp["CVRMSE_pct"], 1),
            "NMBE": round(m_rp["NMBE_pct"], 1),
            "alpha": info.get("best_alpha", "?"),
            "l1_ratio": info.get("best_l1_ratio", "?"),
            "n_features": info.get("n_nonzero_coefs", 0),
        }
    except Exception as e:
        return None


# ============================================================================
# Load SG data
# ============================================================================
def load_sg_all():
    print("Loading SG energy data...")
    df_energy = pd.read_csv(SG_ENERGY, parse_dates=["date"])
    print(f"  Energy: {len(df_energy)} rows, {df_energy['equipment_code'].nunique()} equipment")

    print("Loading SG weather data...")
    df_weather = pd.read_csv(SG_WEATHER, parse_dates=["date"])
    print(f"  Weather: {len(df_weather)} rows")

    return df_energy, df_weather


def prepare_sg_equipment(df_energy, df_weather, equip_code):
    """Prepare single SG equipment DataFrame (same as web app format)."""
    df_eq = df_energy[df_energy["equipment_code"] == equip_code][["date", "daily_kwh"]].copy()
    df_eq = df_eq.set_index("date").sort_index()
    df_eq = df_eq[~df_eq.index.duplicated(keep="first")]

    # Join weather
    df_w = df_weather.set_index("date").sort_index()
    df_w = df_w[~df_w.index.duplicated(keep="first")]
    df = df_eq.join(df_w, how="inner")

    # Add temporal features
    df = add_temporal_features(df)
    return df


# ============================================================================
# Load ECD data
# ============================================================================
def load_ecd_list():
    """List all ECD equipment CSV files."""
    csv_files = sorted(ECD_DATASET.glob("*.csv"))
    print(f"Found {len(csv_files)} ECD equipment files")
    return csv_files


def prepare_ecd_equipment(csv_path):
    """Load single ECD equipment (already has weather + temporal features)."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = df.rename(columns={"usage_value": "daily_kwh"})
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    # Ensure temporal features exist (ECD files may already have them)
    if "month" not in df.columns:
        df = add_temporal_features(df)
    return df


# ============================================================================
# Main scan
# ============================================================================
def scan_dataset(dataset_name, equipment_items, periods):
    """
    Scan all equipment in a dataset.
    equipment_items: list of (equip_id, df) tuples
    periods: list of (tr_start, tr_end, rp_start, rp_end, label) tuples
    Returns: DataFrame with columns [equip_id, best_R2, best_period, all_R2s...]
    """
    results = []
    total = len(equipment_items)

    for i, (equip_id, df) in enumerate(equipment_items):
        row = {"equip_id": equip_id, "n_rows": len(df)}

        best_r2 = -999999
        best_period = "NONE"
        all_r2 = {}

        for (tr_s, tr_e, rp_s, rp_e, label) in periods:
            res = test_one(df, tr_s, tr_e, rp_s, rp_e)
            if res is not None:
                r2 = res["R2_test"]
                all_r2[label] = r2
                if r2 > best_r2:
                    best_r2 = r2
                    best_period = label
                    row["best_n_train"] = res["n_train"]
                    row["best_n_test"] = res["n_test"]
                    row["best_R2_train"] = res["R2_train"]
                    row["best_CVRMSE"] = res["CVRMSE"]
                    row["best_NMBE"] = res["NMBE"]
                    row["best_alpha"] = res["alpha"]
                    row["best_l1_ratio"] = res["l1_ratio"]
                    row["best_n_features"] = res["n_features"]
            else:
                all_r2[label] = None

        row["best_R2"] = best_r2 if best_r2 > -999998 else None
        row["best_period"] = best_period

        # Store individual period R2s
        for label in [p[4] for p in periods]:
            row[f"R2_{label}"] = all_r2.get(label)

        results.append(row)

        status = f"R²={best_r2:.4f}" if best_r2 > -999998 else "NO VALID"
        print(f"  [{i+1:3d}/{total}] {equip_id[:30]:30s} → {status} ({best_period})")

    return pd.DataFrame(results)


def select_top_mid_bot(df_results, n_top=4, n_mid=3, n_bot=3):
    """Select Top-N, Mid-N, Bot-N by best_R2."""
    # Filter out equipment with no valid results
    valid = df_results.dropna(subset=["best_R2"]).copy()
    valid = valid.sort_values("best_R2", ascending=False).reset_index(drop=True)

    n = len(valid)
    if n < n_top + n_mid + n_bot:
        print(f"  Warning: only {n} valid equipment, selecting what we can")

    top = valid.head(n_top)

    # Middle: centered around median
    mid_center = n // 2
    mid_start = max(n_top, mid_center - n_mid // 2)
    mid = valid.iloc[mid_start:mid_start + n_mid]

    # Bottom: worst R2
    bot = valid.tail(n_bot)

    return top, mid, bot


if __name__ == "__main__":
    t0 = time.time()
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    # ========================================================================
    # SAINT-GOBAIN
    # ========================================================================
    log("=" * 80)
    log("SAINT-GOBAIN (SG) — Scanning ALL equipment")
    log("=" * 80)
    log(f"Periods: {[p[4] for p in SG_PERIODS]}")
    log()

    df_energy, df_weather = load_sg_all()
    sg_codes = sorted(df_energy["equipment_code"].unique())
    log(f"Total SG equipment: {len(sg_codes)}")

    sg_items = []
    for code in sg_codes:
        df = prepare_sg_equipment(df_energy, df_weather, code)
        sg_items.append((code, df))
    log(f"Prepared all {len(sg_items)} SG equipment DataFrames")
    log()

    sg_results = scan_dataset("SG", sg_items, SG_PERIODS)

    log()
    log("-" * 80)
    log("SG RANKING (by best R² test across 3 periods):")
    log("-" * 80)
    sg_ranked = sg_results.dropna(subset=["best_R2"]).sort_values("best_R2", ascending=False)
    for i, (_, r) in enumerate(sg_ranked.iterrows()):
        log(f"  #{i+1:2d}  R²={r['best_R2']:9.4f}  period={r['best_period']:20s}  "
            f"α={str(r.get('best_alpha','?')):>10s}  feats={str(r.get('best_n_features','?')):>3s}  "
            f"code={r['equip_id'][:40]}")

    sg_top, sg_mid, sg_bot = select_top_mid_bot(sg_results)

    log()
    log("=" * 60)
    log("SG SELECTED EQUIPMENT:")
    log("=" * 60)
    for cat, df_sel in [("Top", sg_top), ("Mid", sg_mid), ("Bot", sg_bot)]:
        for j, (_, r) in enumerate(df_sel.iterrows()):
            label = f"{cat}-{j+1}"
            log(f"  {label:6s}  R²={r['best_R2']:9.4f}  period={r['best_period']:20s}  "
                f"CVRMSE={r.get('best_CVRMSE','?'):>6}%  NMBE={r.get('best_NMBE','?'):>6}%  "
                f"code={r['equip_id']}")

    # ========================================================================
    # ECD
    # ========================================================================
    log()
    log("=" * 80)
    log("ECD / EC8469 — Scanning ALL equipment")
    log("=" * 80)
    log(f"Periods: {[p[4] for p in ECD_PERIODS]}")
    log()

    ecd_files = load_ecd_list()
    ecd_items = []
    skipped = 0
    for f in ecd_files:
        code = f.stem
        try:
            df = prepare_ecd_equipment(f)
            if df["daily_kwh"].mean() < 0.01:
                skipped += 1
                continue
            ecd_items.append((code, df))
        except Exception as e:
            skipped += 1
            print(f"  Skipped {code}: {e}")

    log(f"Total ECD equipment: {len(ecd_items)} (skipped {skipped} with zero/bad data)")
    log()

    ecd_results = scan_dataset("ECD", ecd_items, ECD_PERIODS)

    log()
    log("-" * 80)
    log("ECD RANKING (by best R² test across 3 periods):")
    log("-" * 80)
    ecd_ranked = ecd_results.dropna(subset=["best_R2"]).sort_values("best_R2", ascending=False)
    for i, (_, r) in enumerate(ecd_ranked.iterrows()):
        log(f"  #{i+1:2d}  R²={r['best_R2']:9.4f}  period={r['best_period']:20s}  "
            f"α={str(r.get('best_alpha','?')):>10s}  feats={str(r.get('best_n_features','?')):>3s}  "
            f"code={r['equip_id'][:40]}")

    ecd_top, ecd_mid, ecd_bot = select_top_mid_bot(ecd_results)

    log()
    log("=" * 60)
    log("ECD SELECTED EQUIPMENT:")
    log("=" * 60)
    for cat, df_sel in [("Top", ecd_top), ("Mid", ecd_mid), ("Bot", ecd_bot)]:
        for j, (_, r) in enumerate(df_sel.iterrows()):
            label = f"{cat}-{j+1}"
            log(f"  {label:6s}  R²={r['best_R2']:9.4f}  period={r['best_period']:20s}  "
                f"CVRMSE={r.get('best_CVRMSE','?'):>6}%  NMBE={r.get('best_NMBE','?'):>6}%  "
                f"code={r['equip_id']}")

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed = time.time() - t0
    log()
    log("=" * 80)
    log(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("=" * 80)

    # Save full results to CSV for reference
    sg_results.to_csv("sg_all_scan.csv", index=False)
    ecd_results.to_csv("ecd_all_scan.csv", index=False)
    log("Saved: sg_all_scan.csv, ecd_all_scan.csv")

    # Save output log
    with open("scan_all_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log("Saved: scan_all_output.txt")
