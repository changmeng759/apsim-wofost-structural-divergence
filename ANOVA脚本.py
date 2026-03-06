import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

IN_CSV  = Path("/Users/mengchang/Desktop/scenario_yield_44x2.csv")
OUT_MAIN = Path("/Users/mengchang/Desktop/TableS1_ANOVA_main_effects.csv")
OUT_INT  = Path("/Users/mengchang/Desktop/TableS1_ANOVA_with_interactions.csv")

df = pd.read_csv(IN_CSV)

# --- 强制类型 ---
df["Year"] = df["Year"].astype(str)
df["NitrogenLabel"] = df["NitrogenLabel"].astype(str)
df["Sowing"] = df["Sowing"].astype(str)
df["model"] = df["model"].astype(str)

def anova_table(sub, formula, rename_map):
    fit = smf.ols(formula, data=sub).fit()
    aov = sm.stats.anova_lm(fit, typ=2)

    aov = aov.rename(index=rename_map)

    out = aov[["df", "sum_sq"]].copy()
    total_ss = out["sum_sq"].sum()
    out["variance_fraction_%"] = out["sum_sq"] / total_ss * 100.0

    out = out.reset_index().rename(columns={
        "index": "Source",
        "df": "df",
        "sum_sq": "SS"
    })
    return out

def run_by_model(formula, rename_map, out_path):
    rows = []
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m].copy()
        tab = anova_table(sub, formula, rename_map)
        tab.insert(0, "Model", m)
        rows.append(tab)
    final = pd.concat(rows, ignore_index=True)
    final.to_csv(out_path, index=False)
    return final

# -------------------------
# A) 主效应版本（你现在的）
# -------------------------
formula_main = "Yield ~ C(Year) + C(NitrogenLabel) + C(Sowing)"
rename_main = {
    "C(Year)": "Year",
    "C(NitrogenLabel)": "NitrogenLabel",
    "C(Sowing)": "Sowing",
    "Residual": "Residuals"
}

main_tab = run_by_model(formula_main, rename_main, OUT_MAIN)
print("\n✅ 主效应版 Table S1 已生成：", OUT_MAIN)
print(main_tab)

# -------------------------
# B) 含交互版本（建议你比较一下）
# -------------------------
formula_int = (
    "Yield ~ C(Year) + C(NitrogenLabel) + C(Sowing)"
    " + C(Year):C(NitrogenLabel)"
    " + C(Year):C(Sowing)"
    " + C(NitrogenLabel):C(Sowing)"
)

rename_int = {
    "C(Year)": "Year",
    "C(NitrogenLabel)": "NitrogenLabel",
    "C(Sowing)": "Sowing",
    "C(Year):C(NitrogenLabel)": "Year×NitrogenLabel",
    "C(Year):C(Sowing)": "Year×Sowing",
    "C(NitrogenLabel):C(Sowing)": "NitrogenLabel×Sowing",
    "Residual": "Residuals"
}

int_tab = run_by_model(formula_int, rename_int, OUT_INT)
print("\n✅ 含交互版 Table S1 已生成：", OUT_INT)
print(int_tab)

print("\n提示：看哪一版 Residuals 显著下降、且交互项有清晰解释意义，就选哪一版放论文。")