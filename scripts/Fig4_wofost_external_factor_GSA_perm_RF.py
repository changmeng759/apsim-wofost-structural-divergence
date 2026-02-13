from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# ======================
# 路径配置（仓库相对路径）
# ======================
ROOT = Path(__file__).resolve().parents[1]

WOFOST_CSV = ROOT / "data" / "WOFOST_daily_sorted.csv"
OUT_DIR    = ROOT / "outputs" / "fig4"
os.makedirs(OUT_DIR, exist_ok=True)


# ======================
# 1. 构建 “Year-Sowing-N” 场景级数据
# ======================
print(f"📂 读取 WOFOST 日度数据: {WOFOST_CSV}")
df = pd.read_csv(WOFOST_CSV)

# 日期与年份
df["day"] = pd.to_datetime(df["day"], errors="coerce")
if "Year" not in df.columns:
    df["Year"] = df["day"].dt.year

# 只保留有作物的记录（TAGP / Yield 非空）
df = df.dropna(subset=["TAGP", "Yield", "Year", "Sowing", "N"])

# 按 Year-Sowing-N 聚合，得到每个情景的 TAGP_max / Yield_max
group_cols = ["Year", "Sowing", "N"]
agg_df = (
    df.groupby(group_cols, as_index=False)
      .agg(
          TAGP_max=("TAGP", "max"),
          Yield_max=("Yield", "max")
      )
)

print("\n✅ 场景级数据预览（前 10 行）：")
print(agg_df.head(10))

# ======================
# 2. 准备特征 X 和目标 y
# ======================

# 将 Sowing 字段编码成数字（作为类别编码）
sowing_cat = agg_df["Sowing"].astype("category")
agg_df["Sowing_code"] = sowing_cat.cat.codes

feature_names = ["Year", "Sowing_code", "N"]

X = agg_df[feature_names].values
y = agg_df["TAGP_max"].values      # 这里选择 TAGP_max 为目标
# 如果以后想换成 Yield，只需要改成：y = agg_df["Yield_max"].values

# ======================
# 3. 拟合 Random Forest
# ======================

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf.fit(X, y)

# 简单评估一下拟合效果（只是 sanity check，不写进论文）
y_pred = rf.predict(X)
r2 = r2_score(y, y_pred)
print(f"\n📊 Random Forest 拟合 R²（训练集）：{r2:.3f}")
print(f"📊 OOB score（如果可用）：{getattr(rf, 'oob_score_', np.nan):.3f}")

# ======================
# 4. Permutation Importance
# ======================
print("\n🔍 计算 Permutation Importance ...")
perm_result = permutation_importance(
    rf,
    X,
    y,
    n_repeats=50,
    random_state=42,
    n_jobs=-1
)

perm_means = perm_result.importances_mean
perm_stds  = perm_result.importances_std

# ======================
# 5. RF 自带 Feature Importance
# ======================
rf_importances = rf.feature_importances_

# ======================
# 6. 绘图：Fig4 左右两幅
# ======================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# 统一 y 轴标签名字
y_labels = ["Year", "Sowing date", "N rate"]

# 排序：按照 Permutation importance 从大到小排序（两幅图一致）
order = np.argsort(-perm_means)
y_pos  = np.arange(len(feature_names))

# ---- 左图：Permutation-based GSA ----
ax = axes[0]
ax.barh(
    y_pos,
    perm_means[order],
    xerr=perm_stds[order],
    align="center",
    alpha=0.8
)
ax.set_yticks(y_pos)
ax.set_yticklabels([y_labels[i] for i in order])
ax.invert_yaxis()
ax.set_xlabel("Permutation importance")
ax.set_title("Global GSA — TAGP$_{max}$\n(Permutation-based)")

# ---- 右图：Random Forest GSA ----
ax = axes[1]
ax.barh(
    y_pos,
    rf_importances[order],
    align="center",
    alpha=0.8
)
ax.set_yticks(y_pos)
ax.set_yticklabels([y_labels[i] for i in order])
ax.invert_yaxis()
ax.set_xlabel("RF feature importance")
ax.set_title("WOFOST GSA\n(Random Forest-based)")

# 网格与布局
for ax in axes:
    ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()

print(f"\n🎯 Fig4（外部因子 GSA）已生成：{FIG_PATH}")