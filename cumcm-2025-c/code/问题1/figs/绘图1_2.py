# -*- coding: utf-8 -*-
"""
多元线性回归（含交互与二次项）诊断图脚本
模型：log(Cy) ~ GA * BMI + Age + BMI^2
输出：三张高清图（PNG 600dpi + SVG），中英双语标签，中文字体自适应。
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import statsmodels.formula.api as smf

# ===================== 配置 =====================
EXCEL_PATH = '附件.xlsx'         # 请确保与脚本在同一目录
SHEET_NAME = None               # None 自动猜测“男胎/胎/male”等工作表；或改为具体表名
COLS_HINT = {
    'GA':  ['检测孕周','孕周','gestational age','ga','J'],
    'BMI': ['孕妇BMI','BMI','bmi','K'],
    'Age': ['年龄','age','C'],
    'Y':   ['Y染色体浓度','y染色体浓度','y浓度','y','V'],
}
OUT_DIR = 'figs'
DPI = 600

# 文件名
F1 = 'Q1_LR_Fig1_Residuals'
F2 = 'Q1_LR_Fig2_StdResiduals'
F3 = 'Q1_LR_Fig3_3D_Surface'

# ===================== 中文字体 & 样式 =====================
def set_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti',
                  'Source Han Sans CN', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
    have = set(f.name for f in fontManager.ttflist)
    for fam in candidates:
        if fam in have:
            mpl.rcParams['font.sans-serif'] = [fam]
            break
    else:
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 兜底
    mpl.rcParams['axes.unicode_minus'] = False

set_chinese_font()
BG_COLOR = '#ffffff'
GRID_COLOR = '#d9d9d9'
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

# ===================== 工具函数 =====================
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

def pick_column(df, aliases):
    cols = [c for c in df.columns]
    # 先全等匹配
    for a in aliases:
        for c in cols:
            if str(c).strip().lower() == a.strip().lower():
                return c
    # 再子串匹配
    for a in aliases:
        for c in cols:
            if a.strip().lower() in str(c).strip().lower():
                return c
    return None

def gestational_weeks_to_float(s):
    """将 '12w+3' / '12周+3天' / '12+3' / '12w' / '12周' / '12' 转为小数周；失败返回 np.nan。"""
    if pd.isna(s): return np.nan
    s = str(s).strip().lower()
    # 12w+3 或 12周+3天
    m = re.match(r'^\s*(\d+)\s*(?:w|周)\s*\+?\s*(\d+)?', s)
    if m:
        weeks = int(m.group(1))
        days = int(m.group(2)) if m.group(2) else 0
        return weeks + days / 7.0
    # 12+3
    m = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', s)
    if m:
        weeks = int(m.group(1)); days = int(m.group(2))
        return weeks + days / 7.0
    # 仅周数
    m = re.match(r'^\s*(\d+)\s*(?:w|周)?\s*$', s)
    if m:
        return float(m.group(1))
    return np.nan

# ===================== 读取与预处理 =====================
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f'未找到 Excel 文件：{EXCEL_PATH}')

df_raw = read_sheet_auto(EXCEL_PATH, SHEET_NAME)
GA_col  = pick_column(df_raw, COLS_HINT['GA'])
BMI_col = pick_column(df_raw, COLS_HINT['BMI'])
Age_col = pick_column(df_raw, COLS_HINT['Age'])
Y_col   = pick_column(df_raw, COLS_HINT['Y'])
if any(c is None for c in (GA_col, BMI_col, Age_col, Y_col)):
    raise ValueError(f'无法识别列，请检查：\n{list(df_raw.columns)}')

df = df_raw[[GA_col, BMI_col, Age_col, Y_col]].copy()
df.columns = ['GA_raw','BMI','Age','Y_conc']

# 孕周转小数周
df['GA'] = df['GA_raw'].apply(gestational_weeks_to_float)

# 统一 Y 单位：若大多 >1，认为是百分数，转为小数
y_raw = pd.to_numeric(df['Y_conc'], errors='coerce')
df['Y'] = y_raw / 100.0 if (y_raw > 1.0).mean() > 0.5 else y_raw

# 清洗
df = df.dropna(subset=['GA','BMI','Age','Y']).copy()
df = df[(df['BMI'] < 40.0) & (df['Y'] >= 0) & (df['Y'] <= 0.18)]
# 合理孕周范围（略放宽）
df = df[(df['GA'] >= 10.0) & (df['GA'] <= 30.0)]

# 目标变量对数
df['logY'] = np.log(df['Y'])

print(f"清洗后样本数：{len(df)}")
print(df[['GA','BMI','Age','Y','logY']].head(5))

# ===================== 拟合模型 =====================
# 模型：logY ~ GA * BMI + Age + I(BMI**2)
model = smf.ols('logY ~ GA * BMI + Age + I(BMI**2)', data=df).fit()
print(model.summary())

# 影响度量（标准化残差等）
infl = model.get_influence()
std_resid = infl.resid_studentized_internal  # 内部学生化残差
fitted = model.fittedvalues
resid = model.resid

# ===================== 创建输出目录 =====================
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== 图1：残差 vs 拟合 =====================
fig1, ax1 = plt.subplots(figsize=(7.5, 4.6))
ax1.scatter(fitted, resid, s=18, alpha=0.75, color='#1f77b4', edgecolors='white', linewidths=0.3,
            label='Residuals')
ax1.axhline(0, color='#d62728', linestyle='--', linewidth=1.2)

ax1.set_title("残差 vs 拟合值 | Residuals vs Fitted", pad=14, fontsize=15)
ax1.set_xlabel("拟合的 ln(Y 浓度) | Fitted ln(Y)", fontsize=12)
ax1.set_ylabel("残差 | Residual", fontsize=12)

# 提示文本
ax1.text(0.01, 0.98, "理想：点云应围绕 0 随机分布\nIdeal: Random scatter around zero",
         transform=ax1.transAxes, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.35', fc='#f7f7f7', ec='#cfcfcf'), fontsize=10)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, f'{F1}.png'), dpi=DPI, bbox_inches='tight')
fig1.savefig(os.path.join(OUT_DIR, f'{F1}.svg'), dpi=DPI, bbox_inches='tight')
plt.close(fig1)

# ===================== 图2：标准化残差 vs 拟合 =====================
fig2, ax2 = plt.subplots(figsize=(7.5, 4.6))
ax2.scatter(fitted, std_resid, s=18, alpha=0.75, color='#ff7f0e', edgecolors='white', linewidths=0.3,
            label='Standardized Residuals')
ax2.axhline(0, color='#444444', linestyle='--', linewidth=1.0)
# ±2, ±3 参考线
for y in [2, -2, 3, -3]:
    ax2.axhline(y, color='#888888', linestyle=':', linewidth=1.0)

ax2.set_title("标准化残差 vs 拟合值 | Standardized Residuals vs Fitted", pad=14, fontsize=15)
ax2.set_xlabel("拟合的 ln(Y 浓度) | Fitted ln(Y)", fontsize=12)
ax2.set_ylabel("标准化残差 | Standardized Residual", fontsize=12)

ax2.text(0.01, 0.98, "多数点应在 ±2 内；超出 ±3 可能为离群\nMost within ±2; beyond ±3 may be outliers",
         transform=ax2.transAxes, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.35', fc='#f7f7f7', ec='#cfcfcf'), fontsize=10)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, f'{F2}.png'), dpi=DPI, bbox_inches='tight')
fig2.savefig(os.path.join(OUT_DIR, f'{F2}.svg'), dpi=DPI, bbox_inches='tight')
plt.close(fig2)

# ===================== 图3：3D 拟合曲面（GA×BMI），Age 固定中位数 =====================
GA_min, GA_max = df['GA'].min(), df['GA'].max()
BMI_min, BMI_max = df['BMI'].min(), df['BMI'].max()
Age_med = float(df['Age'].median())

GA_grid = np.linspace(GA_min, GA_max, 60)
BMI_grid = np.linspace(BMI_min, BMI_max, 60)
GA_mesh, BMI_mesh = np.meshgrid(GA_grid, BMI_grid)
Age_mesh = np.full_like(GA_mesh, Age_med)

# 构造 DataFrame 进行预测
grid_df = pd.DataFrame({
    'GA': GA_mesh.ravel(),
    'BMI': BMI_mesh.ravel(),
    'Age': Age_mesh.ravel()
})
grid_df['BMI2'] = grid_df['BMI']**2  # 仅用于说明；formula 已有 I(BMI**2)
# 使用公式接口的 predict（自动处理交互与二次项）
pred_logY = model.predict(grid_df)
Z = pred_logY.values.reshape(GA_mesh.shape)

fig3 = plt.figure(figsize=(8.4, 6.6))
ax3 = fig3.add_subplot(111, projection='3d')
surf = ax3.plot_surface(GA_mesh, BMI_mesh, Z, rstride=1, cstride=1, linewidth=0,
                        antialiased=True, alpha=0.9, cmap='viridis')
# 叠加稀疏散点（观测）
idx = np.random.choice(len(df), size=min(800, len(df)), replace=False)
ax3.scatter(df.iloc[idx]['GA'], df.iloc[idx]['BMI'], df.iloc[idx]['logY'],
            s=8, alpha=0.3, color='k')

ax3.set_title("三维拟合曲面（Age固定中位数）| 3D Fitted Surface (Age at median)", pad=14, fontsize=14)
ax3.set_xlabel("孕周（周） | GA (weeks)", labelpad=10)
ax3.set_ylabel("体质指数 | BMI", labelpad=10)
ax3.set_zlabel("ln(Y 浓度) | ln(Y)", labelpad=10)
fig3.colorbar(surf, shrink=0.6, aspect=16, pad=0.1, label='Predicted ln(Y)')

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, f'{F3}.png'), dpi=DPI, bbox_inches='tight')
fig3.savefig(os.path.join(OUT_DIR, f'{F3}.svg'), dpi=DPI, bbox_inches='tight')
plt.close(fig3)

print("✅ 已生成三张图（PNG+SVG）：")
print(f"- {os.path.join(OUT_DIR, F1)}.png / .svg")
print(f"- {os.path.join(OUT_DIR, F2)}.png / .svg")
print(f"- {os.path.join(OUT_DIR, F3)}.png / .svg")
