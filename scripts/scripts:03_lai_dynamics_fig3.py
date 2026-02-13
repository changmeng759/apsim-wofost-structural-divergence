import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
	
ROOT = Path(__file__).resolve().parents[1]
LAI_CSV = ROOT / "data" / "L2" / "lai_timeseries_for_fig.csv"
OUT_PNG = ROOT / "outputs" / "fig3" / "Fig3_LAI_dynamics.png"

# 方案A核心：先按 (model, year, doy) 聚合 -> 再按 (model, doy) 做分位数
DOY_MIN, DOY_MAX = 130, 260

# 可选：是否只用某个氮水平（只对 WOFOST 生效；APSIM nitrogen=NaN 会被“池化”）
FILTER_N = None  # 例如 100；要完全不筛选就 None

df = pd.read_csv(LAI_CSV)
df["model"] = df["model"].astype(str).str.strip().str.upper()
df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
df["lai"] = pd.to_numeric(df["lai"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["model", "doy", "lai", "year"])
df["doy"] = df["doy"].astype(int)
df["year"] = df["year"].astype(int)

# DOY 裁剪（关键：去掉 WOFOST 后期异常长尾）
df = df[(df["doy"] >= DOY_MIN) & (df["doy"] <= DOY_MAX)].copy()

# nitrogen 筛选（只筛 WOFOST；APSIM 仍池化）
if FILTER_N is not None and "nitrogen" in df.columns:
    is_wo = df["model"].str.contains("WOFOST")
    df_wo = df[is_wo].copy()
    df_ap = df[~is_wo].copy()
    df_wo = df_wo[df_wo["nitrogen"] == FILTER_N].copy()
    df = pd.concat([df_ap, df_wo], ignore_index=True)

# ----------------------------
# 方案A：按年等权
# 1) 先把同一年同一DOY的多条记录压成一个值（避免某年某DOY重复太多）
# ----------------------------
yearly = (
    df.groupby(["model", "year", "doy"], as_index=False)
      .agg(lai=("lai", "mean"))   # 同年同DOY多条 -> 先平均
)

# 2) 再按 model × doy 计算 mean 与 IQR（此时每一年贡献权重相同）
summ = (
    yearly.groupby(["model", "doy"], as_index=False)
          .agg(
              mean=("lai", "mean"),
              q25=("lai", lambda x: x.quantile(0.25)),
              q75=("lai", lambda x: x.quantile(0.75)),
              n_years=("year", "nunique"),
              n_points=("lai", "size"),
          )
)

aps = summ[summ["model"].str.contains("APSIM")].sort_values("doy")
wof = summ[summ["model"].str.contains("WOFOST")].sort_values("doy")

# ----------------------------
# 画图
# ----------------------------
plt.figure(figsize=(10, 5.8))
ax = plt.gca()

ax.plot(wof["doy"], wof["mean"], linewidth=2.2, label="WOFOST")
ax.fill_between(wof["doy"], wof["q25"], wof["q75"], alpha=0.25)

ax.plot(aps["doy"], aps["mean"], linestyle="--", linewidth=2.2, label="APSIM")
ax.fill_between(aps["doy"], aps["q25"], aps["q75"], alpha=0.25)

ax.set_xlabel("DOY")
ax.set_ylabel("LAI")
title = f"LAI Seasonal Dynamics (Mean ± IQR, DOY {DOY_MIN}–{DOY_MAX})"
if FILTER_N is not None:
    title += f"  [WOFOST N={FILTER_N}; APSIM pooled]"
ax.set_title(title)

ax.grid(True, linestyle=":", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"✅ Saved: {OUT_PNG}")
print("Raw rows used (after DOY cut):")
print(f"  APSIM : {df[df['model'].str.contains('APSIM')].shape[0]}")
print(f"  WOFOST: {df[df['model'].str.contains('WOFOST')].shape[0]}")
print("Yearly-aggregated points used:")
print(f"  APSIM : {yearly[yearly['model'].str.contains('APSIM')].shape[0]}")
print(f"  WOFOST: {yearly[yearly['model'].str.contains('WOFOST')].shape[0]}")
print("Effective years contributing (approx.):")
print(summ.groupby("model")["n_years"].median())