import os
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap


# =========================
# 0) 路径（修法A：绝对路径）
# =========================
CSV_PATH = "/Users/mengchang/Desktop/重要文稿/meta_model_input_44.csv"

OUT_TOP1 = "/Users/mengchang/Desktop/shap_bootstrap_top1_frequency.csv"
OUT_TOP3 = "/Users/mengchang/Desktop/shap_bootstrap_top3_frequency.csv"


# =========================
# 1) 读数据
# =========================
df = pd.read_csv(CSV_PATH)

# 目标列（你文件里是 Yield_gap）
TARGET_COL = "Yield_gap"
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Columns = {df.columns.tolist()}")

y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy()


# =========================
# 2) 选特征（避免目标泄露）
# =========================
drop_cols = ["Yield_gap", "APSIM_Yield", "WOFOST_Yield", "Year"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# 类别特征 one-hot（如果不存在也不会报错）
cat_cols = [c for c in ["sowing", "fertilizer"] if c in X.columns]
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

# 强制数值化
X = X.apply(pd.to_numeric, errors="coerce")

# inf -> NaN
X = X.replace([np.inf, -np.inf], np.nan)

# 填补 NaN：列中位数（仅数值列）
X = X.fillna(X.median(numeric_only=True))

# 最终保证 float64
X = X.astype("float64")

# 同时把 y 的 NaN 行删掉（避免 sklearn 报错）
mask = np.isfinite(y)
X = X.loc[mask].copy()
y = y[mask].astype("float64")

feature_names = X.columns.to_list()
X_values = X.to_numpy(dtype="float64")

n = len(y)
p = X_values.shape[1]
print(f"[INFO] Loaded: n={n}, p={p}")
print(f"[INFO] Output files:\n  {OUT_TOP1}\n  {OUT_TOP3}")


# =========================
# 3) 模型（与你论文一致）
# =========================
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor(random_state=42))
])


# =========================
# 4) Bootstrap SHAP ranking stability
#    - 每次 bootstrap 重新拟合模型
#    - 用 TreeExplainer 计算全体样本的 shap（一致的比较基准）
# =========================
B = 500   # 你想跑1000就改成1000
rng = np.random.default_rng(42)

top1_counter = Counter()
top3_counter = Counter()

for b in range(1, B + 1):
    # 4.1 bootstrap 采样
    idx = rng.integers(0, n, size=n)
    Xb = X_values[idx]
    yb = y[idx]

    # 4.2 拟合
    model.fit(Xb, yb)

    # 4.3 取出 pipeline 内部对象
    scaler = model.named_steps["scaler"]
    gbr = model.named_steps["gbr"]

    # 4.4 用训练好的 scaler 变换全数据
    X_all_s = scaler.transform(X_values)

    # 4.5 TreeExplainer（快且稳定）
    explainer = shap.TreeExplainer(gbr)
    shap_vals = explainer.shap_values(X_all_s)

    # shap_vals shape: (n, p)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top1 = feature_names[order[0]]
    top3 = [feature_names[i] for i in order[:3]]

    top1_counter[top1] += 1
    for f in top3:
        top3_counter[f] += 1

    if b % 50 == 0 or b == 1 or b == B:
        print(f"[PROGRESS] {b}/{B} done. Current top1 = {top1}")


# =========================
# 5) 输出结果（绝对路径，避免权限问题）
# =========================
top1_freq = pd.DataFrame(top1_counter.most_common(), columns=["feature", "top1_count"])
top1_freq["top1_freq"] = top1_freq["top1_count"] / B

top3_freq = pd.DataFrame(top3_counter.most_common(), columns=["feature", "top3_count"])
top3_freq["top3_freq"] = top3_freq["top3_count"] / B

# 确保输出目录存在
os.makedirs(os.path.dirname(OUT_TOP1), exist_ok=True)

top1_freq.to_csv(OUT_TOP1, index=False)
top3_freq.to_csv(OUT_TOP3, index=False)

print("\n=== DONE ===")
print("Top-1 frequency (first 10):")
print(top1_freq.head(10).to_string(index=False))
print("\nTop-3 frequency (first 10):")
print(top3_freq.head(10).to_string(index=False))
print(f"\nSaved:\n  {OUT_TOP1}\n  {OUT_TOP3}")