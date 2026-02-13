
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# ========== 仓库根目录 ==========
ROOT = Path(__file__).resolve().parents[1]

# ========== 数据与输出路径 ==========
CSV_PATH = ROOT / "data" / "scenario_maxima_44.csv"
OUT_DIR  = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ========== 全局绘图风格（期刊风） ==========
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
})

# ========== 1. 读取年度汇总表 ==========
df = pd.read_csv(CSV_PATH)
df = df.sort_values("Year").reset_index(drop=True)

# 基本检查
needed_cols = [
    "Year",
    "Yield_APSIM", "Yield_WOFOST",
    "LAImax_APSIM", "LAImax_WOFOST",
    "TAGPmax_APSIM", "TAGPmax_WOFOST"
]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"年度汇总文件缺少列: {missing}")

print("✅ 年度汇总数据预览：")
print(df)

# 取向量
y_a = df["Yield_APSIM"].values
y_w = df["Yield_WOFOST"].values

# 计算统计量
R2   = r2_score(y_w, y_a)
RMSE = mean_squared_error(y_w, y_a, squared=False)
Bias = (y_a - y_w).mean()


# ========== 2. Fig2a：年度产量散点对比 ==========
fig, ax = plt.subplots(figsize=(6, 6))

# 散点
ax.scatter(y_w, y_a, s=60, edgecolor="black", linewidth=0.6)

# 1:1 线
max_val = max(y_w.max(), y_a.max()) * 1.05
ax.plot([0, max_val], [0, max_val], "--", color="grey", linewidth=1)

# 标题（含 R2 / RMSE / Bias）
title_line1 = "APSIM vs WOFOST yield by year"
title_line2 = r"$R^2={:.3f}$, RMSE={:.1f}, Bias={:.1f}$".format(R2, RMSE, Bias)
ax.set_title(title_line1 + "\n" + title_line2)

# 坐标轴
ax.set_xlabel("WOFOST yield (kg/ha)")
ax.set_ylabel("APSIM yield (kg/ha)")

ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

# 年份标签：做少量错位，避免叠加
for i, row in df.iterrows():
    x = row["Yield_WOFOST"]
    y = row["Yield_APSIM"]
    year = int(row["Year"])

    # 根据索引和年份做轻微偏移
    offset_x = 120 if (i % 2 == 0) else -120
    offset_y = 80 if (year % 2 == 0) else -80

    ax.annotate(
        str(year),
        xy=(x, y),
        xytext=(x + offset_x, y + offset_y),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.8),
        arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.6)
    )

ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)

fig.tight_layout()
fig2a_path = os.path.join(OUT_DIR, "Fig2a_year_yield_scatter.png")
fig.savefig(fig2a_path, dpi=600)
plt.close(fig)
print(f"✅ 已保存: {fig2a_path}")


# ========== 3. Fig2b：年度最大产量箱线图 ==========
fig, ax = plt.subplots(figsize=(5, 6))

data_box = [df["Yield_APSIM"].values, df["Yield_WOFOST"].values]

bp = ax.boxplot(
    data_box,
    patch_artist=True,
    labels=["APSIM", "WOFOST"],
    widths=0.6,
    showfliers=True
)

# 统一颜色（淡色填充）
colors = ["#4C72B0", "#DD8452"]  # 蓝 / 橙
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax.set_ylabel("Yield (kg/ha)")
ax.set_title("Distribution of annual maximum yield\n(APSIM vs WOFOST)")
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

fig.tight_layout()
fig2b_path = os.path.join(OUT_DIR, "Fig2b_year_yield_boxplot.png")
fig.savefig(fig2b_path, dpi=600)
plt.close(fig)
print(f"✅ 已保存: {fig2b_path}")


# ========== 4. Fig2c：LAImax & TAGPmax 对比（双子图） ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ---- 左：LAImax ----
ax = axes[0]
x_lai = df["LAImax_WOFOST"].values
y_lai = df["LAImax_APSIM"].values

ax.scatter(x_lai, y_lai, s=50, edgecolor="black", linewidth=0.6)

max_lai = max(x_lai.max(), y_lai.max()) * 1.05
ax.plot([0, max_lai], [0, max_lai], "--", color="grey", linewidth=1)

ax.set_xlabel("WOFOST LAI$_{max}$ (m$^2$/m$^2$)")
ax.set_ylabel("APSIM LAI$_{max}$ (m$^2$/m$^2$)")
ax.set_title("LAI$_{max}$ comparison")
ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

# 保持图整洁，这一面不加年份标签


# ---- 右：TAGPmax ----
ax2 = axes[1]
x_t = df["TAGPmax_WOFOST"].values
y_t = df["TAGPmax_APSIM"].values

ax2.scatter(x_t, y_t, s=50, edgecolor="black", linewidth=0.6)

max_tagp = max(x_t.max(), y_t.max()) * 1.05
ax2.plot([0, max_tagp], [0, max_tagp], "--", color="grey", linewidth=1)

ax2.set_xlabel("WOFOST TAGP$_{max}$ (kg/ha)")
ax2.set_ylabel("APSIM TAGP$_{max}$ (kg/ha)")
ax2.set_title("TAGP$_{max}$ comparison")
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)

# 在 TAGP 图上加年份标签（点比较分散，便于阅读）
for i, row in df.iterrows():
    year = int(row["Year"])
    x = row["TAGPmax_WOFOST"]
    y = row["TAGPmax_APSIM"]

    offset_x = 250 if (i % 2 == 0) else -250
    offset_y = 150 if (year % 2 == 0) else -150

    ax2.annotate(
        str(year),
        xy=(x, y),
        xytext=(x + offset_x, y + offset_y),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.8),
        arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.6)
    )

fig.tight_layout()
fig2c_path = os.path.join(OUT_DIR, "Fig2c_LAI_TAGP_scatter.png")
fig.savefig(fig2c_path, dpi=600)
plt.close(fig)
print(f"✅ 已保存: {fig2c_path}")

print("\n🎯 Fig2 最终版全部完成！可直接用于论文排版。")