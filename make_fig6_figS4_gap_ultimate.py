#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate reproducible script (GAP-based):
- Fig6: SHAP summary plot for ΔY (Yield_gap) using LAI_gap + Biomass_gap
- FigS4: LOYO RMSE by held-out year + baseline reference line
- Also saves LOYO predictions and SHAP mean(|value|) ranking table.

Designed to reproduce your "last-night" logic:
- Model: sklearn GradientBoostingRegressor (fixed)
- Features: LAI_gap, Biomass_gap (fixed)
- Target: Yield_gap (preferred), fallback to Yield_Diff variants
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# SHAP is optional but strongly recommended for Fig6
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# =========================
# Config (只改这里就行)
# =========================
CSV_PATH = Path("/Users/mengchang/Desktop/重要文稿/meta_model_input_44.csv")
RANDOM_STATE = 42

# =========================
# Helpers
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace; keep original case
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates) -> str:
    """
    Find a column in df given candidate names (case-insensitive).
    """
    # direct hit
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive map
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        key = str(c).lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(f"Cannot find any of {candidates}.\nAvailable columns:\n{df.columns.tolist()}")

def coerce_numeric(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().any():
        # allow a few NaNs; we will fill later (for features). For y/Year we will error.
        pass
    return out

def ensure_year_int(df: pd.DataFrame, year_col: str) -> pd.Series:
    y = pd.to_numeric(df[year_col], errors="raise")
    return y.astype(int)

def make_out_paths(csv_path: Path):
    out_dir = csv_path.parent
    fig6 = out_dir / "Fig6_gap.png"
    figs4 = out_dir / "FigS4_gap_LOYO.png"
    pred_csv = out_dir / "loyo_predictions_gap.csv"
    shap_csv = out_dir / "shap_importance_gap.csv"
    return fig6, figs4, pred_csv, shap_csv


# =========================
# Main
# =========================
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = clean_columns(df)

    # --- detect required columns ---
    year_col = find_col(df, ["Year", "year", "YEAR", "yr", "YR"])

    # Gap-only features (路线B/昨晚最优性能版本)
    lai_gap_col = find_col(df, ["LAI_gap", "lai_gap", "LAI_GAP"])
    bio_gap_col = find_col(df, ["Biomass_gap", "biomass_gap", "BIOMASS_GAP"])

    # Target (prefer Yield_gap)
    target_col = None
    for cand in ["Yield_gap", "yield_gap", "YIELD_GAP", "Yield_Diff", "YieldDiff", "Yield_Difference"]:
        if cand in df.columns or cand.lower() in {c.lower() for c in df.columns}:
            target_col = find_col(df, [cand])
            break
    if target_col is None:
        raise KeyError(f"Cannot find target column (Yield_gap / Yield_Diff ...).\nAvailable:\n{df.columns.tolist()}")

    # --- prepare X / y ---
    df[year_col] = ensure_year_int(df, year_col)

    X = df[[lai_gap_col, bio_gap_col]].copy()
    X = X.apply(lambda s: coerce_numeric(s, s.name))
    # mean impute
    X = X.fillna(X.mean(numeric_only=True))

    y = coerce_numeric(df[target_col], target_col)
    if y.isna().any():
        bad = df.loc[y.isna(), [year_col, target_col]]
        raise ValueError(f"Target column '{target_col}' has NaNs after coercion.\n"
                         f"Check these rows:\n{bad.head(10)}")

    # rename for plotting aesthetics
    X.columns = ["LAI_gap", "Biomass_gap"]

    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print("Detected:")
    print(f"  Year   = {year_col}")
    print(f"  LAI    = {lai_gap_col}")
    print(f"  Biomass= {bio_gap_col}")
    print(f"  Target = {target_col}")
    print("Using GAP-ONLY features:", X.columns.tolist())

    fig6_path, figs4_path, pred_csv_path, shap_csv_path = make_out_paths(CSV_PATH)

    # =========================
    # Part A: LOYO (FigS4)
    # =========================
    years = sorted(df[year_col].unique().tolist())
    rows = []
    fold_rmses = {}

    for yr in years:
        train_mask = df[year_col] != yr
        test_mask = df[year_col] == yr

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        # baseline = mean of training y
        baseline = float(y_train.mean())
        baseline_pred = np.full(shape=len(y_test), fill_value=baseline, dtype=float)

        # fixed model: GradientBoostingRegressor (昨晚版本)
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_rmses[int(yr)] = rmse(y_test, y_pred)

        test_idx = list(X_test.index)
        for j, idx in enumerate(test_idx):
            rows.append({
                "Year": int(df.loc[idx, year_col]),
                "y_true": float(y.loc[idx]),
                "y_pred": float(y_pred[j]),
                "y_baseline": baseline,
            })

    pred_df = pd.DataFrame(rows).sort_values(["Year"])
    overall_rmse = rmse(pred_df["y_true"], pred_df["y_pred"])
    baseline_rmse = rmse(pred_df["y_true"], pred_df["y_baseline"])
    skill = 1.0 - (overall_rmse**2) / (baseline_rmse**2)

    print(f"\nOverall RMSE (model):   {overall_rmse:.6f}")
    print(f"Overall RMSE (baseline):{baseline_rmse:.6f}")
    print(f"Skill:                 {skill:.6f}")

    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved: {pred_csv_path}")

    # Plot FigS4
    xs = list(fold_rmses.keys())
    ys = [fold_rmses[k] for k in xs]

    plt.figure(figsize=(12, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.axhline(baseline_rmse, linestyle="--")
    # annotate baseline
    plt.text(xs[0], baseline_rmse + 0.03, f"baseline RMSE = {baseline_rmse:.3f} t/ha")
    plt.xlabel("Held-out year")
    plt.ylabel("LOYO RMSE (t/ha)")
    plt.title(f"Meta-model LOYO performance (Gap-based; GBR); overall RMSE = {overall_rmse:.3f} t/ha")
    plt.tight_layout()
    plt.savefig(figs4_path, dpi=300)
    print(f"Saved: {figs4_path}")

    # =========================
    # Part B: SHAP (Fig6)
    # =========================
    if not _HAS_SHAP:
        print("\n[WARN] shap is not installed/importable. Skip Fig6 generation.")
        print("Install via: pip install shap")
        return

    # train on full data (standard for SHAP summary)
    final_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    final_model.fit(X, y)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    # Save SHAP ranking evidence (mean absolute SHAP)
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    shap_rank.to_csv(shap_csv_path, index=False)
    print(f"Saved: {shap_csv_path}")

    # Plot Fig6
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Drivers of Yield Difference (ΔY = WOFOST - APSIM), Gap-based features")
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=300)
    print(f"Saved: {fig6_path}")

    print("\n✅ Done. Outputs:")
    print(f"  Fig6 : {fig6_path}")
    print(f"  FigS4: {figs4_path}")
    print(f"  LOYO : {pred_csv_path}")
    print(f"  SHAP : {shap_csv_path}")


if __name__ == "__main__":
    main()