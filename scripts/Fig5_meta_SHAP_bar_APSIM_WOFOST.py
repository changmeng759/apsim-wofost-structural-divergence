from pathlib import Path
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ========= 路径（仓库相对路径）=========
ROOT = Path(__file__).resolve().parents[1]

CSV = ROOT / "data" / "APSIM_WOFOST_year_summary_FINAL.csv"
OUT_DIR = ROOT / "outputs" / "fig5"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 读取数据 =========
df = pd.read_csv(CSV)

# ========= 通用 SHAP 函数 =========
def run_shap(X, y, feature_names, title, out_path):
    """随机森林 + SHAP bar 图（论文级）"""
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 已输出：{out_path}")

# ========= Fig5A — APSIM SHAP (仅 APSIM 自身的 TAGP / LAI) =========
features_APSIM = ["TAGPmax_APSIM", "LAImax_APSIM"]
X_APSIM = df[features_APSIM]
y_APSIM = df["Yield_APSIM"]

run_shap(
    X_APSIM,
    y_APSIM,
    features_APSIM,
    "Fig5A — SHAP for APSIM Yield",
    os.path.join(OUT_DIR, "Fig5A_APSIM_SHAP.png")
)

# ========= Fig5B — WOFOST SHAP (仅 WOFOST 自身的 TAGP / LAI) =========
features_WOFOST = ["TAGPmax_WOFOST", "LAImax_WOFOST"]
X_WOFOST = df[features_WOFOST]
y_WOFOST = df["Yield_WOFOST"]

run_shap(
    X_WOFOST,
    y_WOFOST,
    features_WOFOST,
    "Fig5B — SHAP for WOFOST Yield",
    os.path.join(OUT_DIR, "Fig5B_WOFOST_SHAP.png")
)

print("\n🎉 Fig5（SHAP 解释图，TAGP/LAI 版本）已全部生成！")