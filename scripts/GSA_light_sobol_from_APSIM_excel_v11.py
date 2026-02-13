
import pandas as pd
import numpy as np
import itertools
import re
import matplotlib.pyplot as plt
from pathlib import Path

# =============== 0) 基本设置 ===============
excel_path = ROOT / "data" / "sobol_input.xlsx"
out_dir = ROOT / "outputs" / "sobol"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"✅ 正在读取 Excel：{excel_path}")
df = pd.read_excel(excel_path)
print(f"✅ 读取成功，维度: {df.shape}")

# =============== 1) 列名标准化 ===============
def norm_col(c: str) -> str:
    c = str(c).strip()
    c = c.replace("[", "").replace("]", "")
    c = re.sub(r"\s+", " ", c)
    aliases = {
        "Clock.Today": "Clock.Today",
        "Today": "Clock.Today",
        "Date": "Clock.Today",
        "Soybean.Leaf.LAI": "Soybean.Leaf.LAI",
        "LAI": "Soybean.LAI",
        "Plant.LAI": "Soybean.LAI",
        "Soybean.LAI": "Soybean.LAI",
        "Soybean.AboveGround.Wt": "Soybean.AboveGround.Wt",
        "AboveGround.Wt": "Soybean.AboveGround.Wt",
        "Plant.AboveGround.Wt": "Soybean.AboveGround.Wt",
        "Yield": "Yield",
        "GrainYield": "Yield",
        "Soybean.Grain.Total.Wt": "Soybean.Grain.Total.Wt",
        "Soybean.Grain.Wt": "Soybean.Grain.Wt",
        "Experiment": "Experiment",
        "Factor": "Factor",
        "SimulationID": "SimulationID",
        "SimulationName": "SimulationName",
        "Zone": "Zone",
    }
    return aliases.get(c, c)

df.columns = [norm_col(c) for c in df.columns]

if "Yield" not in df.columns:
    if "Soybean.Grain.Total.Wt" in df.columns:
        df["Yield"] = pd.to_numeric(df["Soybean.Grain.Total.Wt"], errors="coerce") * 10.0

# =============== 2) 生成/修复 SimulationName ===============
def build_simulation_names_auto(df_in: pd.DataFrame, approx_rows_per_sim: int = 4000) -> pd.Series:
    if "SimulationName" in df_in.columns:
        return df_in["SimulationName"].astype(str)
    if "SimulationID" in df_in.columns:
        return df_in["SimulationID"].astype(str).map(lambda x: f"Sim_{x}")
    if "Experiment" in df_in.columns and "Factor" in df_in.columns:
        return (df_in["Experiment"].astype(str).fillna("Exp")
                + "_F" + df_in["Factor"].astype(str).fillna("NA"))

    if "Clock.Today" in df_in.columns:
        dates = pd.to_datetime(df_in["Clock.Today"], errors="coerce")
        change = dates.diff().dt.days.fillna(0)
        new_sim = (change < 0) | (change > 40)
        sim_idx = new_sim.cumsum()
        return sim_idx.map(lambda i: f"Sim_{int(i)+1}")

    n = len(df_in)
    num_sims = max(1, int(round(n / approx_rows_per_sim)))
    edges = np.linspace(0, n, num_sims + 1, dtype=int)
    sim_ids = np.zeros(n, dtype=int)
    for i in range(num_sims):
        sim_ids[edges[i]:edges[i+1]] = i
    return pd.Series(sim_ids, index=df_in.index).map(lambda i: f"Sim_{int(i)+1}")

df["SimulationName"] = build_simulation_names_auto(df, approx_rows_per_sim=4000)
num_sims = df["SimulationName"].nunique()
rows_per_sim = int(len(df) / max(1, num_sims))
print(f"✅ 已确定 SimulationName，共 {num_sims} 个模拟，约 {rows_per_sim} 行/模拟")
print("ℹ 每模拟行数 分布：")
print(df["SimulationName"].value_counts().sort_index().to_string())

# =============== 3) 提取最大 LAI + 最后一天 ===============
lai_col = None
for cand in ["Soybean.Leaf.LAI", "Soybean.LAI"]:
    if cand in df.columns:
        lai_col = cand
        break

lai_max_df = pd.DataFrame()
if lai_col is not None and df[lai_col].notna().any():
    lai_max_idx = (
        df.dropna(subset=[lai_col])
          .groupby("SimulationName")[lai_col]
          .idxmax()
    )
    if len(lai_max_idx) > 0:
        lai_max_df = df.loc[lai_max_idx].copy()
        lai_max_df["Stage"] = "Max_LAI"
        print(f"✅ 成功提取每个模拟的最大 LAI，共 {len(lai_max_idx)} 行")
    else:
        print("⚠️ 未能定位最大 LAI 行，跳过")
else:
    print("⚠️ 未检测到 LAI 列或全为空，跳过最大 LAI 提取")

last_day_df = df.groupby("SimulationName").tail(1).copy()
last_day_df["Stage"] = "Last_Day"

merged = pd.concat([lai_max_df, last_day_df], ignore_index=True)
print(f"✅ 聚合后行数: {len(merged)} (包含 Max_LAI 与 Last_Day)")

# =============== 4) Factor 列清洗与模式识别 ===============
if "Factor" not in merged.columns:
    merged["Factor"] = np.nan

factor_raw = merged["Factor"].astype(str).str.strip()
factor_raw = factor_raw.replace({"None": np.nan, "none": np.nan, "NaN": np.nan, "nan": np.nan, "": np.nan})
merged["Factor_raw"] = factor_raw
merged["Factor_int"] = factor_raw.str.extract(r"^(\d+)$")[0]
print("✅ Factor 唯一值（样例）:", merged["Factor_raw"].dropna().astype(str).unique()[:10])

unique_raw_clean = set(merged["Factor_raw"].dropna().astype(str))
only_rue_three = (len(unique_raw_clean) > 0) and unique_raw_clean.issubset({"1.2", "1.4", "1.6"})
param_names = ["RUE","TSUM1","TSUM2","NRate","SowingTiming"]

if only_rue_three:
    print("🔎 检测到仅 RUE 三水平模式（1.2/1.4/1.6） → 模式B（RUE-only）")
    DEFAULTS = {"TSUM1": 800, "TSUM2": 600, "NRate": 0, "SowingTiming": "normal"}
    merged["RUE"] = merged["Factor_raw"].map({"1.2": 1.2, "1.4": 1.4, "1.6": 1.6}).astype(float)
    for k,v in DEFAULTS.items():
        merged[k] = v
    merged["SowingTiming_code"] = pd.Categorical(
        merged["SowingTiming"], categories=["early","normal","late"], ordered=True
    ).codes
    X_cols = ["RUE","TSUM1","TSUM2","NRate","SowingTiming_code"]
else:
    print("🔎 检测到 Factor 为整数/混合编号 → 模式A（全因子/多因子映射）")
    levels = {
        "RUE": [1.2,1.5,1.8],
        "TSUM1": [800,900,1000],
        "TSUM2": [600,700,800],
        "NRate": [0,50,100],
        "SowingTiming": ["early","normal","late"]
    }
    combos = list(itertools.product(*levels.values()))
    param_table = pd.DataFrame(combos, columns=param_names)
    param_table["Factor_int"] = [str(i+1) for i in range(len(param_table))]
    merged = merged.merge(param_table, on="Factor_int", how="left", validate="many_to_one")
    merged["SowingTiming_code"] = pd.Categorical(
        merged["SowingTiming"].astype(str), categories=["early","normal","late"], ordered=True
    ).codes
    X_cols = ["RUE","TSUM1","TSUM2","NRate","SowingTiming_code"]

# =============== 5) 轻量“组间均值方差贡献”分析 ===============
def simple_sobol_effect(X: np.ndarray, Y: np.ndarray, names: list[str]) -> pd.DataFrame:
    df_tmp = pd.DataFrame(X, columns=names)
    df_tmp["Y"] = Y
    total_var = np.var(Y, ddof=1)
    if total_var == 0 or np.isnan(total_var):
        return pd.DataFrame({"Parameter": names,
                             "Effect": [np.nan]*len(names),
                             "%Variance_Explained": [np.nan]*len(names)})
    rows = []
    for col in names:
        gmean = df_tmp.groupby(col)["Y"].mean()
        var_explained = np.var(gmean.values, ddof=1) if len(gmean)>1 else 0.0
        rows.append((col, var_explained/total_var))
    out = pd.DataFrame(rows, columns=["Parameter","Effect"]).sort_values("Effect", ascending=False)
    out["%Variance_Explained"] = (out["Effect"] * 100).round(2)
    return out

candidates = ["Yield","Soybean.Leaf.LAI","Soybean.LAI","Soybean.AboveGround.Wt"]
output_cols = [c for c in candidates if c in merged.columns]
print(f"📊 检测到输出列: {output_cols}")

for y_col in output_cols:
    print(f"\n=== 🔍 分析输出变量: {y_col} ===")
    data = merged.dropna(subset=[y_col])[X_cols + [y_col]].copy()
    for c in X_cols + [y_col]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    X = data[X_cols].values
    Y = data[y_col].values

    if len(Y) < 8 or np.nanstd(Y)==0:
        print(f"⚠️ 样本数不足或方差为0（n={len(Y)}），跳过 {y_col}")
        continue

    Si_df = simple_sobol_effect(X, Y, ["RUE","TSUM1","TSUM2","NRate","SowingTiming"])
    csv_path = out_dir / f"SobolLite_Sensitivity_{y_col}.csv"
    Si_df.to_csv(csv_path, index=False)
    print(f"💾 已保存结果: {csv_path}")
    print(Si_df)

    plt.figure(figsize=(8,5))
    plt.bar(Si_df["Parameter"], Si_df["Effect"])
    plt.title(f"Simplified Sensitivity for {y_col}")
    plt.ylabel("Effect (Variance Contribution)")
    plt.tight_layout()
    plt.savefig(out_dir / f"SobolLite_{y_col}_BarChart.png", dpi=300)
    plt.close()

print("\n🎯 完成！结果目录：", out_dir)