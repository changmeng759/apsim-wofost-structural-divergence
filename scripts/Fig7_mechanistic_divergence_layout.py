

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -------------------------------------------------
# 数据：来自 Fig5 真实 SHAP 值
# -------------------------------------------------
APSIM_data = {"TAGP": 6603.8934, "LAI": 3.0525}
WOFOST_data = {"TAGP": 14457.3056, "LAI": 8.2871}

# 归一化
def normalize(dic):
    total = sum(dic.values())
    return {k: v / total for k, v in dic.items()}

APSIM_norm = normalize(APSIM_data)
WOFOST_norm = normalize(WOFOST_data)

# -------------------------------------------------
# Figure Layout
# -------------------------------------------------
fig = plt.figure(figsize=(22, 10))
gs = gridspec.GridSpec(
    2,
    3,
    width_ratios=[1, 1, 1.2],
    height_ratios=[0.2, 1],
    wspace=0.35
)

# 主标题 Panel
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(
    0.5, 0.45,
    "Fig7 — Meta-SHAP mechanistic divergence between APSIM & WOFOST",
    fontsize=24, fontweight="bold", ha="center", va="center"
)

# =======================
# 7A Panel
# =======================
axA = fig.add_subplot(gs[1, 0])
axA.set_title("7A — Normalized SHAP (APSIM vs WOFOST)", fontsize=16)

labels = ["TAGP", "LAI"]
x = np.arange(len(labels))
bar_width = 0.35

axA.bar(x - bar_width/2, [APSIM_norm["TAGP"], APSIM_norm["LAI"]],
        width=bar_width, label="APSIM", color="#1f77b4")
axA.bar(x + bar_width/2, [WOFOST_norm["TAGP"], WOFOST_norm["LAI"]],
        width=bar_width, label="WOFOST", color="#ff7f0e")

axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=14)
axA.set_ylabel("Normalized SHAP (0–1)", fontsize=14)
axA.legend(fontsize=12)

# =======================
# 7B Panel
# =======================
axB = fig.add_subplot(gs[1, 1])
axB.set_title("7B — SHAP difference (normalized)", fontsize=16)

diff = [
    WOFOST_norm["TAGP"] - APSIM_norm["TAGP"],
    WOFOST_norm["LAI"] - APSIM_norm["LAI"]
]

axB.bar(labels, diff, color="#1f77b4")
axB.axhline(0, color="black", linewidth=1)
axB.set_ylabel("ΔSHAP (WOFOST − APSIM)", fontsize=14)
axB.set_xticklabels(labels, fontsize=14)

# =======================
# 7C Panel（最终修复版，仅此版本）
# =======================
axC = fig.add_subplot(gs[1, 2])
axC.set_title(
    "7C — Mechanistic pathway divergence\n(weighted by normalized SHAP)",
    fontsize=16,
    pad=15
)
axC.axis("off")

# 坐标位置设计
y_APSIM = 0.72
y_WOFOST = 0.32
x_start = 0.05
x_mid = 0.40
x_end = 0.70

# --- APSIM Path ---
axC.text(x_start, y_APSIM, "APSIM", fontsize=20, color="#1f77b4", fontweight="bold")

axC.text(x_mid - 0.12, y_APSIM, "TAGP (%.2f)" % APSIM_norm["TAGP"],
         fontsize=17, color="#1f77b4")
axC.annotate("", xy=(x_mid, y_APSIM), xytext=(x_mid - 0.12, y_APSIM),
             arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4"))

axC.text(x_end - 0.12, y_APSIM, "LAI (%.2f)" % APSIM_norm["LAI"],
         fontsize=17, color="#1f77b4")
axC.annotate("", xy=(x_end, y_APSIM), xytext=(x_end - 0.12, y_APSIM),
             arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4"))

axC.text(x_end + 0.15, y_APSIM, "Yield", fontsize=20, color="#1f77b4")

# --- WOFOST Path ---
axC.text(x_start, y_WOFOST, "WOFOST", fontsize=20, color="#ff7f0e", fontweight="bold")

axC.text(x_mid - 0.12, y_WOFOST, "LAI (%.2f)" % WOFOST_norm["LAI"],
         fontsize=17, color="#ff7f0e")
axC.annotate("", xy=(x_mid, y_WOFOST), xytext=(x_mid - 0.12, y_WOFOST),
             arrowprops=dict(arrowstyle="->", lw=2, color="#ff7f0e"))

axC.text(x_end - 0.12, y_WOFOST, "TAGP (%.2f)" % WOFOST_norm["TAGP"],
         fontsize=17, color="#ff7f0e")
axC.annotate("", xy=(x_end, y_WOFOST), xytext=(x_end - 0.12, y_WOFOST),
             arrowprops=dict(arrowstyle="->", lw=2, color="#ff7f0e"))

axC.text(x_end + 0.15, y_WOFOST, "Yield", fontsize=20, color="#ff7f0e")

# -------------------------------------------------
# 保存
# -------------------------------------------------
OUT_DIR_FIG7 = ROOT / "outputs" / "fig7"
os.makedirs(OUT_DIR_FIG7, exist_ok=True)
OUT = OUT_DIR_FIG7 / "Fig7_MetaSHAP_Final.png"

plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("🎉 Perfect Figure7 saved to:", OUT)