import pandas as pd
import numpy as np
from pathlib import Path

# ====== 可选：用于训练 meta-model（如果你有特征表）======
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# =========================
# 0) 路径：只需要改这里
# =========================
SCENARIO_CSV = Path("/Users/mengchang/Desktop/scenario_table_44_A0.csv")

# 如果你有 meta-model 的特征表（含 year + sowing + N_label + 一堆特征列），填上它的路径；
# 没有就先保持 None，脚本会只算 ΔY 和 baseline
#FEATURES_CSV = None
FEATURES_CSV = Path("/Users/mengchang/Desktop/重要文稿/meta_model_input_44.csv")

# 输出：LOYO 预测保存位置
OUT_PRED_CSV = Path("/Users/mengchang/Desktop/predictions_loyo.csv")

# =========================
# 1) 读取 scenario 表并计算 ΔY
# =========================
df = pd.read_csv(SCENARIO_CSV)
print("Scenario columns:", df.columns.tolist())
print("Scenario rows:", len(df))

# 统一列名（你的文件已经是这5列）
COL_YEAR = "year"
COL_SOW  = "sowing"
COL_N    = "N_label"
COL_APSIM = "yield_APSIM"
COL_WOF  = "yield_WOFOST"

df["deltaY"] = df[COL_WOF] - df[COL_APSIM]

# =========================
# 2) ΔY summary（含 median + IQR）
# =========================
deltaY = df["deltaY"].to_numpy(dtype=float)

mean_dy = float(np.mean(deltaY))
std_dy  = float(np.std(deltaY, ddof=1))
min_dy  = float(np.min(deltaY))
max_dy  = float(np.max(deltaY))

median_dy = float(np.median(deltaY))
q25 = float(np.percentile(deltaY, 25))
q75 = float(np.percentile(deltaY, 75))

print("\n=== ΔY summary (t/ha) ===")
print(f"mean   = {mean_dy:.4f}")
print(f"std    = {std_dy:.4f}")
print(f"min    = {min_dy:.4f}")
print(f"max    = {max_dy:.4f}")

print("\n=== ΔY robust summary (t/ha) ===")
print(f"median = {median_dy:.4f}")
print(f"IQR    = [{q25:.4f}, {q75:.4f}]")

# =========================
# 3) LOYO baseline overall RMSE（拼接所有 test 点）
# =========================
years = sorted(df[COL_YEAR].unique().tolist())

all_pred_base = []
all_true = []

for y in years:
    train = df[df[COL_YEAR] != y]
    test  = df[df[COL_YEAR] == y]

    train_mean = float(train["deltaY"].mean())
    y_test = test["deltaY"].to_numpy(dtype=float)

    all_pred_base.extend([train_mean] * len(test))
    all_true.extend(y_test.tolist())

all_pred_base = np.array(all_pred_base, dtype=float)
all_true = np.array(all_true, dtype=float)

baseline_rmse = float(np.sqrt(mean_squared_error(all_true, all_pred_base)))

print("\n=== Overall LOYO baseline RMSE ===")
print(f"{baseline_rmse:.4f} t/ha")

# =========================
# 4) meta-model LOYO RMSE + skill（如果提供 FEATURES_CSV）
# =========================
if FEATURES_CSV is None:
    print("\n[STOP] FEATURES_CSV is None, so I only computed ΔY + baseline.")
    print("NEXT: Provide a features CSV (44 rows) to compute rmse_model + skill.")
else:
    feat = pd.read_csv(FEATURES_CSV)
    print("\nFeatures columns:", feat.columns.tolist())
    print("Features rows:", len(feat))

    # ========= 关键：把 features 的列名对齐到 scenario 的 key =========
    rename_map = {}
    if "Year" in feat.columns and "year" not in feat.columns:
        rename_map["Year"] = "year"
    if "fertilizer" in feat.columns and "N_label" not in feat.columns:
        rename_map["fertilizer"] = "N_label"
    feat = feat.rename(columns=rename_map)

    # ========= 清洗 key 字段，避免大小写/空格导致 merge 失败 =========
    for c in ["year", "sowing", "N_label"]:
        if c not in feat.columns:
            raise KeyError(f"[features] missing key column: {c}")

    feat["year"] = pd.to_numeric(feat["year"], errors="coerce").astype(int)

    feat["sowing"] = feat["sowing"].astype(str).str.strip().str.lower()
    feat["N_label"] = feat["N_label"].astype(str).str.strip().str.lower()

    # scenario 这边也清洗一下（更稳）
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["sowing"] = df["sowing"].astype(str).str.strip().str.lower()
    df["N_label"] = df["N_label"].astype(str).str.strip().str.lower()

    # ========= merge =========
    key = [COL_YEAR, COL_SOW, COL_N]  # ['year','sowing','N_label']
    dfm = df[key + ["deltaY"]].merge(feat, on=key, how="inner")

    if len(dfm) != 44:
        # 方便你定位是哪些没对上
        left_keys = set(tuple(x) for x in df[key].values.tolist())
        right_keys = set(tuple(x) for x in feat[key].values.tolist())
        missing_in_feat = sorted(left_keys - right_keys)[:10]
        missing_in_sce  = sorted(right_keys - left_keys)[:10]
        raise ValueError(
            f"After merge, rows={len(dfm)} not equal to 44.\n"
            f"Example keys missing in features: {missing_in_feat}\n"
            f"Example keys missing in scenario: {missing_in_sce}\n"
            f"Check sowing/N_label spelling."
        )

    # ========= 自动识别特征列，但要严格避免 yield 泄漏 =========
    # 任何包含 'yield' 的列都剔除（APSIM_Yield, WOFOST_Yield, Yield_gap 等）
    feature_cols = []
    for c in dfm.columns:
        if c in key or c == "deltaY":
            continue
        if "yield" in c.lower():
            continue
        feature_cols.append(c)

    # 只保留数值特征
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(dfm[c])]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found after filtering (and removing yield leakage).")

    print("\nUsing feature columns (n=%d):" % len(feature_cols))
    print(feature_cols)

    # ========= LOYO：拼接所有 test 预测，算 overall rmse_model =========
    all_pred_model = []
    all_true_model = []
    pred_rows = []

    for y in years:
        train = dfm[dfm[COL_YEAR] != y].copy()
        test  = dfm[dfm[COL_YEAR] == y].copy()

        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train["deltaY"].to_numpy(dtype=float)

        X_test  = test[feature_cols].to_numpy(dtype=float)
        y_test  = test["deltaY"].to_numpy(dtype=float)

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_pred_model.extend(y_pred.tolist())
        all_true_model.extend(y_test.tolist())

        for i in range(len(test)):
            pred_rows.append({
                "year": int(test.iloc[i][COL_YEAR]),
                "sowing": str(test.iloc[i][COL_SOW]),
                "N_label": str(test.iloc[i][COL_N]),
                "y_true": float(y_test[i]),
                "y_pred": float(y_pred[i]),
            })

    all_pred_model = np.array(all_pred_model, dtype=float)
    all_true_model = np.array(all_true_model, dtype=float)

    rmse_model = float(np.sqrt(mean_squared_error(all_true_model, all_pred_model)))
    skill = 1 - (rmse_model**2 / baseline_rmse**2)

    print("\n=== Overall LOYO model RMSE ===")
    print(f"{rmse_model:.4f} t/ha")

    print("\n=== Skill score ===")
    print(f"baseline_rmse = {baseline_rmse:.4f} t/ha")
    print(f"rmse_model    = {rmse_model:.4f} t/ha")
    print(f"skill         = {skill:.4f}")

    pd.DataFrame(pred_rows).to_csv(OUT_PRED_CSV, index=False)
    print(f"\nSaved predictions: {OUT_PRED_CSV}")
    import matplotlib.pyplot as plt

pred = pd.read_csv(OUT_PRED_CSV)

# 每折 RMSE
rmse_by_year = (
    pred.groupby("year")
        .apply(lambda g: np.sqrt(np.mean((g["y_true"] - g["y_pred"])**2)))
        .reset_index(name="rmse")
        .sort_values("year")
)

overall_rmse = float(np.sqrt(np.mean((pred["y_true"] - pred["y_pred"])**2)))

fig = plt.figure(figsize=(10, 4.8))
ax = plt.gca()

ax.plot(rmse_by_year["year"], rmse_by_year["rmse"], marker="o", linewidth=2)
ax.axhline(baseline_rmse, linestyle="--", linewidth=1.5)
ax.text(rmse_by_year["year"].min(), baseline_rmse,
        f" baseline RMSE = {baseline_rmse:.3f} t/ha",
        va="bottom")

ax.set_xlabel("Held-out year")
ax.set_ylabel("LOYO RMSE (t/ha)")
ax.set_title(f"Meta-model LOYO performance (GradientBoosting); overall RMSE = {overall_rmse:.3f} t/ha")
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("/Users/mengchang/Desktop/重要文稿/FigS4_meta_model_performance.png", dpi=300)
print("Saved figure: /Users/mengchang/Desktop/重要文稿/FigS4_meta_model_performance.png")