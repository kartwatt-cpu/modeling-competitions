import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


EXCEL_PATH = "数据清洗.xlsx"   
SHEET_NAME = "Sheet1"        
COL_Y = "Y染色体浓度"          
COL_GA = "孕周(周)"            
COL_BMI = "孕妇BMI"            


OUT1_PNG = "Y染色体浓度_随孕周散点回归.png"
OUT1_SVG = "Y染色体浓度_随孕周散点回归.svg"
OUT2_PNG = "Y染色体浓度_随BMI平方散点回归.png"
OUT2_SVG = "Y染色体浓度_随BMI平方散点回归.svg"


SCATTER_COLOR = "#4C78A8"
FITLINE_COLOR = "#E45756"
BACKGROUND_COLOR = "#FFFFFF"

#工具函数
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

def polyfit_with_r2(x, y, deg=2):

    p = np.poly1d(np.polyfit(x, y, deg))
    y_hat = p(x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return p, r2

def draw_scatter_with_fit(
    x, y, xlabel, ylabel, title,
    out_png, out_svg,
    scatter_color=SCATTER_COLOR,
    fit_color=FITLINE_COLOR,
    deg=2
):

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 5:
        raise ValueError(f"有效样本太少（{len(x_clean)}），请检查数据。")

    p, r2 = polyfit_with_r2(x_clean, y_clean, deg=deg)

    x_min, x_max = np.nanmin(x_clean), np.nanmax(x_clean)
    xs = np.linspace(x_min, x_max, 250)
    ys = p(xs)


    fig = plt.figure(figsize=(8.5, 6), dpi=180, facecolor=BACKGROUND_COLOR)
    ax = plt.gca()
    ax.set_facecolor("#FAFAFA")

    ax.scatter(
        x_clean, y_clean,
        s=28, alpha=0.85, edgecolors="white", linewidths=0.6,
        color=scatter_color, label="样本"
    )
    ax.plot(xs, ys, color=fit_color, linewidth=2.4, label=f"{deg}次拟合")


    if deg == 2:
        a, b, c = p.c
        eq_str = f"y = {a:.4g}·x² + {b:.4g}·x + {c:.4g}\nR² = {r2:.4f}"
    else:

        terms = []
        for i, coef in enumerate(p.c):
            power = deg - i
            if power >= 2:
                terms.append(f"{coef:.4g}·x^{power}")
            elif power == 1:
                terms.append(f"{coef:.4g}·x")
            else:
                terms.append(f"{coef:.4g}")
        eq_str = "y = " + " + ".join(terms) + f"\nR² = {r2:.4f}"


    ax.text(
        0.98, 0.02, eq_str,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#DDDDDD")
    )


    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)


    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.tight_layout()


    ax.legend(frameon=False, fontsize=12, loc="best")


    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"已导出：{out_png} 与 {out_svg}")

#主流程
def main():
    set_chinese_font()

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"未找到 Excel：{EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    for col in [COL_Y, COL_GA, COL_BMI]:
        if col not in df.columns:
            raise KeyError(f"未找到列：{col}（请检查表头与 SHEET_NAME）")


    y = pd.to_numeric(df[COL_Y], errors="coerce").values
    ga = pd.to_numeric(df[COL_GA], errors="coerce").values
    bmi = pd.to_numeric(df[COL_BMI], errors="coerce").values
    bmi2 = bmi ** 2

    draw_scatter_with_fit(
        x=ga,
        y=y,
        xlabel="孕周（周）",
        ylabel="Y染色体浓度",
        title="Y染色体浓度与孕周的关系",
        out_png=OUT1_PNG,
        out_svg=OUT1_SVG,
        deg=2
    )

    draw_scatter_with_fit(
        x=bmi2,
        y=y,
        xlabel="BMI²",
        ylabel="Y染色体浓度",
        title="Y染色体浓度与BMI平方的关系",
        out_png=OUT2_PNG,
        out_svg=OUT2_SVG,
        deg=2
    )

    print("\n✅ 全部完成！")

if __name__ == "__main__":
    main()
