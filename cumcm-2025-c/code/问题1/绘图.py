
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from scipy.optimize import curve_fit
from textwrap import dedent


EXCEL_PATH = '附件.xlsx'   
SHEET_NAME = None        
COL_WEEK = None           
COL_BMI  = None         
COL_Y    = None          


MAX_BMI = 40.0
MAX_Y_FRACTION = 0.18     
BINS = [0, 30, 35, 40]
BIN_LABELS = ['<30', '30–35', '35–40']


OUT_DIR = 'figs'
FIG1_NAME = 'Q1_Fig1_BMI_facets'  
FIG2_NAME = 'Q1_Fig2_BMI_compare'  
DPI = 600

# 鲜艳配色
PALETTE = {
    '<30':   '#1f77b4',  # 蓝
    '30–35': '#ff7f0e',  # 橙
    '35–40': '#d62728',  # 红
    'scatter': '#2ca02c',# 绿（散点）
}
BG_COLOR = '#ffffff'
GRID_COLOR = '#d9d9d9'

#中文字体与全局样式
def set_chinese_font():
    """
    自动选择本机可用的中文字体：SimHei / Microsoft YaHei / PingFang SC / STHeiti / Source Han Sans CN / Noto Sans CJK SC
    """
    candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti',
                  'Source Han Sans CN', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
    have = set(f.name for f in fontManager.ttflist)
    for fam in candidates:
        if fam in have:
            mpl.rcParams['font.sans-serif'] = [fam]
            break
    else:
       
        mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False  

set_chinese_font()

mpl.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

#工具函数
def gestational_weeks_to_float(s):
  
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    m = re.match(r'^\s*(\d+)\s*(?:w|周)\s*\+?\s*(\d+)?', s)
    if m:
        weeks = int(m.group(1))
        days = int(m.group(2)) if m.group(2) else 0
        return weeks + days / 7.0
    m = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', s)  # 12+3
    if m:
        weeks = int(m.group(1)); days = int(m.group(2))
        return weeks + days / 7.0
    m = re.match(r'^\s*(\d+)\s*(?:w|周)?\s*$', s)   
    if m:
        return float(m.group(1))
    return np.nan

def logistic(t, L, k, t0):
  
    z = np.clip(-k * (t - t0), -700, 700)
    return L / (1.0 + np.exp(z))

def guess_columns(df):
   
    candidates = {
        'week': ['检测孕周','孕周','检测孕周(周+天)','孕周(周+天)','J','week','gestation','gestational age'],
        'bmi':  ['孕妇BMI','BMI','K','bmi'],
        'y':    ['Y染色体浓度','Y 浓度','V','y_fraction','y conc','y浓度']
    }
    def find_col(kind):
        for name in candidates[kind]:
            for col in df.columns:
                if str(col).strip().lower() == str(name).strip().lower():
                    return col
        key = {'week':'周', 'bmi':'bmi', 'y':'y'}[kind]
        for col in df.columns:
            if key in str(col).lower():
                return col
        return None
    return (COL_WEEK or find_col('week'),
            COL_BMI  or find_col('bmi'),
            COL_Y    or find_col('y'))

def read_sheet_auto(path, sheet_name=None):
  
    xls = pd.ExcelFile(path)
    if sheet_name:
        return pd.read_excel(path, sheet_name=sheet_name)
    cand = None
    for s in xls.sheet_names:
        s_low = s.lower()
        if ('男' in s) or ('胎' in s_low) or ('male' in s_low):
            cand = s; break
    if cand is None:
        cand = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=cand)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

#读取与清洗
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f'未找到 Excel 文件：{EXCEL_PATH}')

df_raw = read_sheet_auto(EXCEL_PATH, SHEET_NAME).copy()
w_col, b_col, y_col = guess_columns(df_raw)
if any(c is None for c in (w_col, b_col, y_col)):
    raise ValueError(f'无法自动识别列名，请在顶部设置 COL_WEEK/COL_BMI/COL_Y。\n列清单：{list(df_raw.columns)}')

df = df_raw[[w_col, b_col, y_col]].rename(columns={w_col:'检测孕周', b_col:'孕妇BMI', y_col:'Y染色体浓度'})
df['孕周(周)'] = df['检测孕周'].apply(gestational_weeks_to_float)

y_raw = pd.to_numeric(df['Y染色体浓度'], errors='coerce')
df['Y_fraction'] = y_raw / 100.0 if (y_raw > 1.0).mean() > 0.5 else y_raw

df_clean = df.dropna(subset=['孕周(周)','孕妇BMI','Y_fraction']).copy()
df_clean = df_clean[(df_clean['孕妇BMI'] < MAX_BMI) & (df_clean['Y_fraction'] <= MAX_Y_FRACTION)]
df_clean = df_clean[(df_clean['孕周(周)'] >= 10.0) & (df_clean['孕周(周)'] <= 30.0)]
df_clean['BMI组'] = pd.cut(df_clean['孕妇BMI'], bins=BINS, labels=BIN_LABELS, include_lowest=True, right=False)
df_clean = df_clean[~df_clean['BMI组'].isna()].copy()

print(f'清洗后样本数：{len(df_clean)}')
print(df_clean['BMI组'].value_counts().sort_index())

#分组 Logistic 拟合
def fit_logistic_by_group(df_group, init=None, bounds=None, maxfev=200000):
    t = df_group['孕周(周)'].to_numpy()
    y = df_group['Y_fraction'].to_numpy()
    if init is None:
        L0 = float(np.clip(np.quantile(y, 0.95), 0.08, 0.25))
        init = [L0, 0.6, 18.0]
    if bounds is None:
        bounds = ([0.0, 0.0, 0.0], [0.35, 5.0, 60.0])
    popt, pcov = curve_fit(logistic, t, y, p0=init, bounds=bounds, maxfev=maxfev)
    return popt, pcov

fit_params, r2_by_group = {}, {}
for lab in BIN_LABELS:
    sub = df_clean[df_clean['BMI组'] == lab]
    if len(sub) < 10:
        print(f'组 {lab} 样本过少，跳过拟合。'); continue
    p0 = {'<30':[0.14,0.6,16.0],'30–35':[0.14,0.6,18.0],'35–40':[0.12,0.7,20.0]}.get(lab,[0.14,0.6,18.0])
    popt, pcov = fit_logistic_by_group(sub, init=p0)
    fit_params[lab] = (popt, pcov)
    L, k, t0 = popt
    yhat = logistic(sub['孕周(周)'].to_numpy(), L, k, t0)
    r2_by_group[lab] = r2_score(sub['Y_fraction'].to_numpy(), yhat)
    print(f'BMI {lab}: L={L:.4f}, k={k:.4f}, t0={t0:.2f}, R²={r2_by_group[lab]:.3f}')

#图1：三联分面图（排版优化，避免重叠）
os.makedirs(OUT_DIR, exist_ok=True)
t_grid = np.linspace(df_clean['孕周(周)'].min(), df_clean['孕周(周)'].max(), 400)

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9))  

fig.suptitle("胎儿 Y 染色体浓度随孕周的增长（按 BMI 分组）", y=1.06, fontsize=16)

for ax, lab in zip(axes, BIN_LABELS):
    sub = df_clean[df_clean['BMI组'] == lab]
   
    ax.scatter(sub['孕周(周)'], sub['Y_fraction']*100, s=18,
               alpha=0.65, linewidths=0.3, edgecolors='white',
               color=PALETTE['scatter'], label='观测点')
    if lab in fit_params:
        L, k, t0 = fit_params[lab][0]
        ax.plot(t_grid, logistic(t_grid, L, k, t0)*100,
                lw=2.6, color=PALETTE[lab], label='Logistic 拟合')
        txt = f"L={L*100:.1f}%，  k={k:.3f},  t0={t0:.1f}周\n$R^2$={r2_by_group[lab]:.3f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='#f7f7f7', ec='#cfcfcf'),
                fontsize=10)
  
    ax.set_title(f"BMI {lab}", pad=16)
    ax.set_xlabel("孕周（周）")
    ax.set_ylabel("Y 染色体浓度（%）")
    ax.set_ylim(0, max(18, (df_clean['Y_fraction']*100).max()*1.1))
    ax.legend(loc='lower right', frameon=True)

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1.25))

plt.tight_layout(rect=[0, 0, 1, 0.93])

png_path1 = os.path.join(OUT_DIR, f"{FIG1_NAME}.png")
svg_path1 = os.path.join(OUT_DIR, f"{FIG1_NAME}.svg")
fig.savefig(png_path1, dpi=DPI, bbox_inches='tight')
fig.savefig(svg_path1, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"已保存：\n- {png_path1}\n- {svg_path1}")

#图2：同轴对比 
fig2, ax2 = plt.subplots(figsize=(7.6, 4.8))
ax2.set_title("不同 BMI 组 Logistic 曲线对比", pad=14, fontsize=15)


for lab in BIN_LABELS:
    if lab in fit_params:
        L, k, t0 = fit_params[lab][0]
        ax2.plot(t_grid, logistic(t_grid, L, k, t0)*100,
                 lw=2.8, color=PALETTE[lab], label=f"BMI {lab} 拟合")


ax2.scatter(df_clean['孕周(周)'], df_clean['Y_fraction']*100,
            s=10, alpha=0.28, color='#6a6a6a', label='观测点（全部）')

ax2.set_xlabel("孕周（周）")
ax2.set_ylabel("Y 染色体浓度（%）")
ax2.set_ylim(0, max(18, (df_clean['Y_fraction']*100).max()*1.1))
ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2.5))
ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1.25))
ax2.legend(loc='lower right', frameon=True, ncol=1)


ax2.axhline(4.0, ls='--', lw=1.2, color='#888888')
ax2.text(ax2.get_xlim()[0] + 0.2, 4.2, "4% 阈值", color='#666666')

plt.tight_layout()

png_path2 = os.path.join(OUT_DIR, f"{FIG2_NAME}.png")
svg_path2 = os.path.join(OUT_DIR, f"{FIG2_NAME}.svg")
fig2.savefig(png_path2, dpi=DPI, bbox_inches='tight')
fig2.savefig(svg_path2, dpi=DPI, bbox_inches='tight')
plt.close(fig2)
print(f"已保存：\n- {png_path2}\n- {svg_path2}")

# 控制台摘要
summary = dedent(f"""

样本总数（清洗后）：{len(df_clean)}
各 BMI 组样本数：
{df_clean['BMI组'].value_counts().sort_index().to_string()}

Logistic 拟合参数（L, k, t0）与 R²：
""").strip()
print(summary)
for lab in BIN_LABELS:
    if lab in fit_params:
        L, k, t0 = fit_params[lab][0]
        print(f"- BMI {lab}: L={L:.4f}, k={k:.4f}, t0={t0:.2f}, R²={r2_by_group[lab]:.3f}")
    else:
        print(f"- BMI {lab}: 样本不足，未拟合。")

print("\n图像已导出为 PNG(600dpi) 与 SVG（矢量），可放心插入 Word/论文。")
