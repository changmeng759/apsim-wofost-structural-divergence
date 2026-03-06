import pandas as pd
import numpy as np
from pathlib import Path

# ====== 1) 读入你的 scenario 表 ======
# TODO: 把这里改成你的文件路径
DATA_PATH = Path("/Users/mengchang/Desktop/scenario_table_44_A0.csv")  # 例如: Path("/Users/mengchang/Desktop/.../scenario_yields.csv")

df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# ====== 2) TODO：把下面四个列名改成你自己的 ======
COL_YEAR = "year"
COL_APSIM = "yield_APSIM"
COL_WOFOST = "yield_WOFOST"

# ====== 3) 计算 ΔY ======
df["deltaY"] = df[COL_WOFOST] - df[COL_APSIM]

# ====== 4) ΔY 统计 ======
deltaY = df["deltaY"].to_numpy(dtype=float)

mean_dy = np.mean(deltaY)
std_dy  = np.std(deltaY, ddof=1)  # 用样本标准差（推荐）
min_dy  = np.min(deltaY)
max_dy  = np.max(deltaY)

print("\n=== ΔY summary (t/ha) ===")
print(f"mean = {mean_dy:.4f}")
print(f"std  = {std_dy:.4f}")
print(f"min  = {min_dy:.4f}")
print(f"max  = {max_dy:.4f}")

# ====== 5) LOYO baseline RMSE + skill ======
# 说明：
# - 每次留出一个 year 的 4 个样本做测试
# - baseline: 用训练集 deltaY 的均值作为常数预测
# - model_rmse: 你需要填入你在 Fig.S4 里每折的 rmse_model（或整体 rmse_model）
#   这里先演示 baseline_rmse 的算法，并给你输出每折 baseline RMSE
years = sorted(df[COL_YEAR].unique().tolist())

fold_rows = []
for y in years:
    train = df[df[COL_YEAR] != y].copy()
    test  = df[df[COL_YEAR] == y].copy()

    y_train = train["deltaY"].to_numpy(dtype=float)
    y_test  = test["deltaY"].to_numpy(dtype=float)

    y_train_mean = float(np.mean(y_train))        # baseline predictor
    baseline_rmse = float(np.sqrt(np.mean((y_test - y_train_mean) ** 2)))

    fold_rows.append({
        "heldout_year": y,
        "n_test": len(test),
        "train_mean_deltaY": y_train_mean,
        "baseline_rmse": baseline_rmse
    })

fold = pd.DataFrame(fold_rows)

print("\n=== LOYO baseline by year ===")
print(fold.to_string(index=False))

# overall baseline RMSE：把所有 test 点拼起来算（更合理）
# 做法：对每个点，用它所在 fold 的 train_mean 预测，然后统一算 RMSE
preds = []
truth = []
for y in years:
    train = df[df[COL_YEAR] != y]
    test  = df[df[COL_YEAR] == y]
    y_train_mean = float(train["deltaY"].mean())

    preds.extend([y_train_mean] * len(test))
    truth.extend(test["deltaY"].tolist())

preds = np.array(preds, dtype=float)
truth = np.array(truth, dtype=float)

overall_baseline_rmse = float(np.sqrt(np.mean((truth - preds) ** 2)))

print("\n=== Overall LOYO baseline RMSE ===")
print(f"{overall_baseline_rmse:.4f} t/ha")

print("\nNEXT STEP:")
print("1) 你需要从 Fig.S4/你的代码里拿到 overall rmse_model（LOYO 下）")
print("2) 然后用 skill = 1 - (rmse_model**2 / overall_baseline_rmse**2)")