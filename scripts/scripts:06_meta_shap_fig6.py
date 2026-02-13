from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. 基础配置
# ==============================================================================
# 骨架文件
path_merged = '/data/L2/APSIM_WOFOST_year_summary_FINAL.csv'
# WOFOST 补全文件
path_wofost_source = 'wofost_LAI_biomass_yield_2014_2024.csv'
# APSIM 补全文件
path_apsim_source = 'APSIM_N_response_master.xlsx'

# 图片保存路径
save_path = 'Figure6_Merged_Final.png'

print("🚀 开始执行修复版任务 (v3.0 - 修复大小写问题)...")

# ==============================================================================
# 2. 辅助函数：标准化管理措施 & 年份
# ==============================================================================
def normalize_columns(df):
    # 1. 去除空格
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 强制修复年份列名 (year -> Year)
    # 只要列名里包含 'year' (忽略大小写)，就把它重命名为 'Year'
    for c in df.columns:
        if c.lower() == 'year':
            df.rename(columns={c: 'Year'}, inplace=True)
            break
            
    # 3. 找 Sowing 和 Nitrogen
    sow_col = next((c for c in df.columns if 'sow' in c.lower()), None)
    fert_col = next((c for c in df.columns if any(k in c.lower() for k in ['fert', 'nitro', 'n_rate', 'n_level'])), None)
    
    if sow_col and fert_col:
        # 标准化 Sowing
        df[sow_col] = df[sow_col].astype(str).apply(lambda x: 'SowLate' if 'Late' in x else 'SowNormal')
        
        # 标准化 Nitrogen
        def norm_fert(x):
            s = str(x).lower()
            if 'high' in s or '100' in s: return 'High'
            return 'Normal'
        df[fert_col] = df[fert_col].apply(norm_fert)
        return df, sow_col, fert_col
    return df, None, None

# ==============================================================================
# 3. 读取并“侦察”数据
# ==============================================================================

# --- 读取骨架 ---
print(f"\n1️⃣ 读取骨架文件...")
df_main = pd.read_csv(path_merged)
df_main, col_sow_main, col_n_main = normalize_columns(df_main)
print(f"   骨架列名: {df_main.columns.tolist()}")

# --- 读取 WOFOST ---
print(f"\n2️⃣ 读取 WOFOST 补充文件...")
try:
    df_w = pd.read_csv(path_wofost_source)
    df_w, col_sow_w, col_n_w = normalize_columns(df_w) # 这里会把 year 变成 Year
    print(f"   [调试] WOFOST 现列名: {df_w.columns.tolist()}")
    
    # 查找 Biomass (tagp, bio, total, wso, final_biomass)
    col_bio_w = next((c for c in df_w.columns if any(k in c.lower() for k in ['tagp', 'bio', 'wso', 'total'])), None)
    
    if col_bio_w:
        print(f"   ✅ 自动锁定 WOFOST Biomass 列: {col_bio_w}")
    else:
        print("   ❌ 没找到 Biomass 列")
except Exception as e:
    print(f"   ⚠️ 读取失败: {e}")
    df_w, col_bio_w = None, None

# --- 读取 APSIM ---
print(f"\n3️⃣ 读取 APSIM 补充文件...")
try:
    df_a = pd.read_excel(path_apsim_source)
    df_a, col_sow_a, col_n_a = normalize_columns(df_a) # 这里会把 year 变成 Year
    print(f"   [调试] APSIM 现列名: {df_a.columns.tolist()}")
    
    # 查找 Biomass
    col_bio_a = next((c for c in df_a.columns if any(k in c.lower() for k in ['bio', 'above', 'payload'])), None)
    
    if col_bio_a:
        print(f"   ✅ 自动锁定 APSIM Biomass 列: {col_bio_a}")
    else:
        print("   ❌ 没找到 Biomass 列")
except Exception as e:
    print(f"   ⚠️ 读取失败: {e}")
    df_a, col_bio_a = None, None

# ==============================================================================
# 4. 合并数据
# ==============================================================================
print("\n🔄 正在合并...")
df_final = df_main.copy()
# 确保骨架里的 Year 是整数
df_final['Year'] = df_final['Year'].astype(int)

# 合并 WOFOST Biomass
if df_w is not None and col_bio_w:
    # 此时 df_w 里一定是 'Year' (大写)，因为经过了 normalize_columns
    temp_w = df_w[['Year', col_sow_w, col_n_w, col_bio_w]].copy()
    temp_w.columns = ['Year', col_sow_main, col_n_main, 'max_biomass_wofost']
    temp_w['Year'] = temp_w['Year'].astype(int)
    
    df_final = pd.merge(df_final, temp_w, on=['Year', col_sow_main, col_n_main], how='left')
    print("   ✅ 已合入 WOFOST Biomass")

# 合并 APSIM Biomass
if df_a is not None and col_bio_a:
    temp_a = df_a[['Year', col_sow_a, col_n_a, col_bio_a]].copy()
    temp_a.columns = ['Year', col_sow_main, col_n_main, 'max_biomass_apsim']
    temp_a['Year'] = temp_a['Year'].astype(int)
    
    df_final = pd.merge(df_final, temp_a, on=['Year', col_sow_main, col_n_main], how='left')
    print("   ✅ 已合入 APSIM Biomass")

# ==============================================================================
# 5. 准备画图数据
# ==============================================================================
# 目标：Delta Y
# 自动寻找产量列
col_y_w = next((c for c in df_final.columns if 'yield_wofost' in c.lower()), None)
col_y_a = next((c for c in df_final.columns if 'yield_apsim' in c.lower()), None)

if col_y_w and col_y_a:
    df_final['Yield_Diff'] = df_final[col_y_w] - df_final[col_y_a]
else:
    print("❌ 致命错误：找不到产量列，请检查骨架文件列名！")
    exit()

# 特征
candidates = {
    'WOFOST Peak LAI': next((c for c in df_final.columns if 'lai_wofost' in c.lower()), None),
    'APSIM Peak LAI': next((c for c in df_final.columns if 'lai_apsim' in c.lower()), None),
    'WOFOST Peak Biomass': 'max_biomass_wofost' if 'max_biomass_wofost' in df_final.columns else None,
    'APSIM Peak Biomass': 'max_biomass_apsim' if 'max_biomass_apsim' in df_final.columns else None
}
# 过滤掉 None
final_features = {k: v for k, v in candidates.items() if v is not None}
X = df_final[list(final_features.values())]
X.columns = list(final_features.keys())
y = df_final['Yield_Diff']

# 简单的缺失值填充
X = X.fillna(X.mean())

print(f"\n📊 最终特征: {X.columns.tolist()}")

# ==============================================================================
# 6. 画图
# ==============================================================================
print("🤖 训练模型中...")
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title('Drivers of Yield Difference (WOFOST - APSIM)')
plt.tight_layout()

print(f"💾 正在保存至: {save_path}")
plt.savefig(save_path, dpi=300)
print("✅ 成功！请去桌面查看图片。")