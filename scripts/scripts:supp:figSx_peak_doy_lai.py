from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) Path
# =========================
path = ROOT / "data" / "lai_timeseries_for_fig.csv"

# =========================
# 1) Load
# =========================
df = pd.read_csv(path)
print("Raw Columns:", df.columns.tolist())
print(df.head())

# =========================
# 2) Normalize columns
# =========================
df.columns = [c.strip().lower() for c in df.columns]

required = {"year", "doy", "model", "lai"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 统一 model 字段（避免 APSIM / apsim / APSIM 这种）
df["model"] = df["model"].astype(str).str.strip().str.upper()

# =========================
# 3) Convert types
# =========================
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["doy"]  = pd.to_numeric(df["doy"],  errors="coerce")
df["lai"]  = pd.to_numeric(df["lai"],  errors="coerce")

df = df.dropna(subset=["year", "doy", "lai", "model"])

# （可选）避免 lai 为负或异常值
# df = df[df["lai"] >= 0]

# =========================
# 4) CORE: peak per model × year
# =========================
idx = df.groupby(["model", "year"])["lai"].idxmax()

peak = (
    df.loc[idx, ["model", "year", "doy", "lai"]]
      .rename(columns={"doy": "peak_doy", "lai": "peak_lai"})
      .sort_values(["model", "year"])
      .reset_index(drop=True)
)

# 输出 peak 表
out_csv = ROOT / "data" / "peak_DOY_from_lai_timeseries.csv"
peak.to_csv(out_csv, index=False)

print("\n✅ Saved peak DOY table to:", out_csv)
print(peak.head(10))
print("\nCounts by model:")
print(peak["model"].value_counts())

# =========================
# 5) Plot: peak DOY distribution (hist + rug)
# =========================
fig, ax = plt.subplots(figsize=(7.2, 4.0))

models = sorted(peak["model"].unique())

for model in models:
    vals = peak.loc[peak["model"] == model, "peak_doy"].dropna().astype(int).values
    ax.hist(vals, bins=10, alpha=0.5, label=model, edgecolor="black")
    ax.plot(vals, np.zeros_like(vals), "|", markersize=12)

ax.set_xlabel("Day of Year (DOY)")
ax.set_ylabel("Count")
ax.set_title("Distribution of peak DOY for LAI")
ax.legend(frameon=False)

plt.tight_layout()

OUT_DIR = ROOT / "outputs" / "supp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_png = OUT_DIR / "FigSx_peak_DOY_distribution.png"
plt.savefig(out_png, dpi=300)
plt.show()

print("✅ Figure saved to:", out_png)