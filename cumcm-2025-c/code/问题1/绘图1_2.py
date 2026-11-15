import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats


EXCEL_PATH = "数据清洗.xlsx"
SHEET_NAME = "Sheet1"
COL_BMI = "孕妇BMI"
COL_Y   = "Y染色体浓度"

ENABLE_WINSOR = True
LOW_Q, HIGH_Q = 0.01, 0.99

OUT_PNG = "BMI_vs_Y浓度_散点回归.png"
OUT_SVG = "BMI_vs_Y浓度_散点回归.svg"

SCATTER_COLOR = "#4C78A8"
FITLINE_COLOR = "#E45756"
BACKGROUND_COLOR = "#FFFFFF"

#字体
def set_chinese_font():
    preferred = [
        "SimHei", "Microsoft YaHei", "微软雅黑", "PingFang SC",
        "Hiragino Sans GB", "Source Han Sans SC", "Noto Sans CJK SC"
    ]
    avail = set(f.name for f in font_manager.fontManager.ttflist)
    for name in preferred:
        if name in avail:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

# ========= 工具函数 =========
def polyfit_with_r2(x, y, deg=2):
    p = np.poly1d(np.polyfit(x, y, deg))
    y_hat = p(x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return p, r2

# ========= 绘图函数 =========
def draw_bmi_vs_y(df):
    x = pd.to_numeric(df[COL_BMI], errors="coerce").astype(float).values
    y = pd.to_numeric(df[COL_Y],   errors="coerce").astype(float).values

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        raise ValueError("有效样本不足，请检查数据。")

    if ENABLE_WINSOR:
        x_low, x_high = np.nanquantile(x, LOW_Q),  np.nanquantile(x, HIGH_Q)
        y_low, y_high = np.nanquantile(y, LOW_Q),  np.nanquantile(y, HIGH_Q)
        keep = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
        x, y = x[keep], y[keep]

    p, r2 = polyfit_with_r2(x, y, deg=2)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 250)
    ys = p(xs)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    # LaTeX 方程字符串
    a, b, c = p.c
    eq_str = (
        f"$y = {a:.4g}x^2 + {b:.4g}x + {c:.4g}$\n"
        f"$R^2 = {r2:.4f}$\n"
        f"Pearson r = {pearson_r:.4f} (p={pearson_p:.2g})\n"
        f"Spearman $\\rho$ = {spearman_rho:.4f} (p={spearman_p:.2g})"
    )

    fig = plt.figure(figsize=(8.5, 6), dpi=180, facecolor=BACKGROUND_COLOR)
    ax = plt.gca()
    ax.set_facecolor("#FAFAFA")

    ax.scatter(
        x, y, s=28, alpha=0.88,
        edgecolors="white", linewidths=0.6,
        color=SCATTER_COLOR, label="样本"
    )
    ax.plot(xs, ys, color=FITLINE_COLOR, linewidth=2.4, label="二次拟合")

    ax.text(
        0.98, 0.02, eq_str,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="#DDDDDD")
    )

    ax.set_title("BMI 与 Y染色体浓度 的关系", fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("孕妇 BMI", fontsize=13)
    ax.set_ylabel("Y染色体浓度", fontsize=13)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.34)
    ax.legend(frameon=False, fontsize=12, loc="best")
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_SVG, bbox_inches="tight")
    plt.close()

    print("\n=== 输出完成 ===")
    print(f"图像已保存为：{OUT_PNG}, {OUT_SVG}")

# ========= 主程序 =========
if __name__ == "__main__":
    set_chinese_font()
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"未找到 Excel：{EXCEL_PATH}")
    data = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    for col in [COL_BMI, COL_Y]:
        if col not in data.columns:
            raise KeyError(f"缺少列：{col}")
    draw_bmi_vs_y(data)
