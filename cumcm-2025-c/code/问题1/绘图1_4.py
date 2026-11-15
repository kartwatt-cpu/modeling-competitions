# -*- coding: utf-8 -*-
"""
NIPT 问题一：Logistic 模型 3D 可视化（完整整合版）
- 读取 附件.xlsx（自动猜测含“男/胎/male”的工作表）
- 清洗：孕周(周+天)->小数周、去极端 BMI 与异常 Y 浓度
- 按三组 BMI (<30, 30–35, 35–40) 分别拟合 Logistic: Y(t)=L/(1+exp(-k*(t-t0)))
- 沿 BMI 维对 (L,k,t0) 线性插值，构造任意 BMI 的连续预测
- 输出 3D 曲面：
  1) GA×BMI→预测 Y 浓度曲面
  2) 4% 阈值达标孕周曲面（主视角/俯视/侧视三张图）
"""

import os, re
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit

# ---------------- 字体与风格（中文自适配） ----------------
def set_chinese_font():
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
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.color': '#d9d9d9',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ---------------- 配置 ----------------
EXCEL_PATH = '附件.xlsx'
SHEET_NAME = None
OUT_DIR = 'figs'
DPI = 600

# BMI 分组（中心点用于参数插值）
BMI_BINS    = [0, 30, 35, 40]
BMI_LABELS  = ['<30', '30–35', '35–40']
BMI_CENTERS = {'<30': 27.5, '30–35': 32.5, '35–40': 37.5}

# 阈值（达标时间曲面用）
THRESHOLD = 0.04  # 4%

# ---------------- 工具函数 ----------------
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

def pick_column(df, candidates):
    # 严格匹配
    for a in candidates:
        for c in df.columns:
            if str(c).strip().lower() == a.strip().lower():
                return c
    # 子串匹配
    for a in candidates:
        for c in df.columns:
            if a.strip().lower() in str(c).strip().lower():
                return c
    return None

def gestational_weeks_to_float(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower()
    m = re.match(r'^\s*(\d+)\s*(?:w|周)\s*\+?\s*(\d+)?', s)
    if m:
        weeks = int(m.group(1)); days = int(m.group(2)) if m.group(2) else 0
        return weeks + days/7.0
    m = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', s)
    if m:
        return int(m.group(1)) + int(m.group(2))/7.0
    m = re.match(r'^\s*(\d+)\s*(?:w|周)?\s*$', s)
    if m:
        return float(m.group(1))
    return np.nan

def logistic(t, L, k, t0):
    z = np.clip(-k*(t - t0), -700, 700)
    return L / (1 + np.exp(z))

def fit_logistic(x, y, p0=None, bounds=None, maxfev=200000):
    if p0 is None:
        L0 = float(np.clip(np.quantile(y, 0.95), 0.08, 0.25))
        p0 = [L0, 0.6, 18.0]
    if bounds is None:
        bounds = ([0.0, 0.0, 0.0], [0.35, 5.0, 60.0])
    popt, pcov = curve_fit(logistic, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    return popt, pcov

def interp_param_by_bmi(bmi, centers, param_dict):
    # centers: dict(label->center_bmi) ; param_dict: dict(label->value)
    items = sorted([(centers[k], param_dict[k]) for k in centers.keys()])
    xs = np.array([it[0] for it in items], float)
    ys = np.array([it[1] for it in items], float)
    if bmi <= xs[0]:
        x1,x2,y1,y2 = xs[0], xs[1], ys[0], ys[1]
    elif bmi >= xs[-1]:
        x1,x2,y1,y2 = xs[-2], xs[-1], ys[-2], ys[-1]
    else:
        j = np.searchsorted(xs, bmi)
        x1,x2 = xs[j-1], xs[j]
        y1,y2 = ys[j-1], ys[j]
    t = 0 if x2==x1 else (bmi - x1)/(x2 - x1)
    return y1 + t*(y2 - y1)

# ---------------- 读取与清洗 ----------------
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f'未找到 Excel：{EXCEL_PATH}')

df_raw = read_sheet_auto(EXCEL_PATH, SHEET_NAME)
GA_col  = pick_column(df_raw, ['检测孕周','孕周','gestational age','ga','J'])
BMI_col = pick_column(df_raw, ['孕妇BMI','BMI','bmi','K'])
Y_col   = pick_column(df_raw, ['Y染色体浓度','y染色体浓度','y浓度','y','V'])
if any(c is None for c in (GA_col, BMI_col, Y_col)):
    raise ValueError(f'无法识别所需列，请检查：{list(df_raw.columns)}')

df = df_raw[[GA_col, BMI_col, Y_col]].copy().rename(
    columns={GA_col:'GA_raw', BMI_col:'BMI', Y_col:'Y_raw'}
)
df['GA'] = df['GA_raw'].apply(gestational_weeks_to_float)

y_val = pd.to_numeric(df['Y_raw'], errors='coerce')
df['Y'] = y_val/100.0 if (y_val > 1.0).mean() > 0.5 else y_val

df = df.dropna(subset=['GA','BMI','Y']).copy()
df = df[(df['BMI'] < 40.0) & (df['Y'] >= 0) & (df['Y'] <= 0.18)]
df = df[(df['GA'] >= 10.0) & (df['GA'] <= 30.0)]

# ---------------- 分组拟合 Logistic ----------------
df['BMI组'] = pd.cut(df['BMI'], bins=BMI_BINS, labels=BMI_LABELS, include_lowest=True, right=False)
fit_params = {}  # {label: (L,k,t0)}
for lab in BMI_LABELS:
    sub = df[df['BMI组']==lab]
    if len(sub) < 10:
        continue
    p0 = {'<30':[0.14,0.6,16.0],'30–35':[0.14,0.6,18.0],'35–40':[0.12,0.7,20.0]}.get(lab,[0.14,0.6,18.0])
    (L,k,t0),_ = fit_logistic(sub['GA'].to_numpy(), sub['Y'].to_numpy(), p0=p0)
    fit_params[lab] = (float(L), float(k), float(t0))

# 若有组不足，则用现有组的平均参数兜底
if len(fit_params) < len(BMI_LABELS) and len(fit_params)>0:
    Ls, ks, t0s = zip(*fit_params.values())
    fallback = (float(np.mean(Ls)), float(np.mean(ks)), float(np.mean(t0s)))
    for lab in BMI_LABELS:
        if lab not in fit_params:
            fit_params[lab] = fallback

print("各 BMI 组拟合参数(L,k,t0)：", fit_params)

L_map  = {lab: fit_params[lab][0] for lab in BMI_LABELS}
k_map  = {lab: fit_params[lab][1] for lab in BMI_LABELS}
t0_map = {lab: fit_params[lab][2] for lab in BMI_LABELS}

def params_at_bmi(b):
    Lb  = interp_param_by_bmi(b, BMI_CENTERS, L_map)
    kb  = interp_param_by_bmi(b, BMI_CENTERS, k_map)
    t0b = interp_param_by_bmi(b, BMI_CENTERS, t0_map)
    return Lb, kb, t0b

# ---------------- 构建网格并预测：图1 预测曲面 ----------------
GA_grid  = np.linspace(df['GA'].min(),  df['GA'].max(), 80)
BMI_grid = np.linspace(max(20.0, df['BMI'].min()), min(39.9, df['BMI'].max()), 80)
GA_mesh, BMI_mesh = np.meshgrid(GA_grid, BMI_grid)

Y_pred = np.zeros_like(GA_mesh, dtype=float)
for i in range(GA_mesh.shape[0]):
    for j in range(GA_mesh.shape[1]):
        Lb, kb, t0b = params_at_bmi(BMI_mesh[i,j])
        Y_pred[i,j] = logistic(GA_mesh[i,j], Lb, kb, t0b)

# 图1：GA×BMI→预测 Y 浓度曲面
os.makedirs(OUT_DIR, exist_ok=True)
fig1 = plt.figure(figsize=(8.4, 6.6), constrained_layout=True)
ax1 = fig1.add_subplot(111, projection='3d')

# 光照增强
cmap1 = plt.get_cmap('viridis')
norm1 = mpl.colors.Normalize(vmin=np.nanmin(Y_pred), vmax=np.nanmax(Y_pred))
ls1 = LightSource(azdeg=315, altdeg=35)
shaded1 = ls1.shade(Y_pred, cmap=cmap1, vert_exag=0.8, blend_mode='overlay',
                    vmin=np.nanmin(Y_pred), vmax=np.nanmax(Y_pred))

surf1 = ax1.plot_surface(GA_mesh, BMI_mesh, Y_pred*100.0, rstride=1, cstride=1,
                         linewidth=0, antialiased=True, alpha=0.96,
                         facecolors=shaded1, shade=False)
ax1.plot_wireframe(GA_mesh, BMI_mesh, Y_pred*100.0, rstride=8, cstride=8,
                   color='k', linewidth=0.25, alpha=0.25)

ax1.set_title("预测 Y 浓度 3D 曲面 | Predicted Y Concentration Surface", pad=14)
ax1.set_xlabel("孕周（周） | GA (weeks)", labelpad=10)
ax1.set_ylabel("体质指数 | BMI", labelpad=10)
ax1.set_zlabel("Y 浓度（%） | Y (%)", labelpad=10)

mappable1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
mappable1.set_array([])
cbar1 = fig1.colorbar(mappable1, ax=ax1, shrink=0.64, aspect=18, pad=0.06)
cbar1.set_label("预测 Y 浓度（%） | Predicted Y (%)", fontsize=11)

ax1.view_init(elev=28, azim=-125)
fig1.savefig(os.path.join(OUT_DIR, "Q1_Logistic_3D_PredSurface.png"), dpi=DPI, bbox_inches='tight')
fig1.savefig(os.path.join(OUT_DIR, "Q1_Logistic_3D_PredSurface.svg"), bbox_inches='tight')
plt.close(fig1)

# ---------------- 图2：4% 阈值达标孕周 曲面（v3 三视角） ----------------
# 达标孕周解析：t* = t0 + (1/k) * ln(L/Y* - 1)，0<Y*<L 且 k>0
GA_thresh = np.full_like(BMI_grid, np.nan, dtype=float)
for idx, b in enumerate(BMI_grid):
    Lb, kb, t0b = params_at_bmi(b)
    if (THRESHOLD > 0) and (Lb is not None) and (kb is not None) and (THRESHOLD < Lb) and (kb > 0):
        GA_thresh[idx] = t0b + (1.0/kb) * np.log(Lb/THRESHOLD - 1.0)
    else:
        GA_thresh[idx] = np.nan  # 阈值不可达或无效

GA_thresh_mesh = np.tile(GA_thresh[:, None], (1, len(GA_grid)))
Z = GA_thresh_mesh
X = BMI_mesh
Y = GA_mesh

# 屏蔽无效值，准备光照
Z_masked = ma.masked_invalid(Z)
mask_unreach = Z_masked.mask
zmin = np.nanmin(Z)
zmax = np.nanmax(Z)
cmap2 = plt.get_cmap('plasma')
norm2 = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
ls2 = LightSource(azdeg=315, altdeg=35)
shaded2 = ls2.shade(Z_masked.filled(zmin), cmap=cmap2, vert_exag=0.8,
                    blend_mode='overlay', vmin=zmin, vmax=zmax)

xlim = (np.nanmin(X), np.nanmax(X))
ylim = (np.nanmin(Y), np.nanmax(Y))
zlim = (zmin, zmax)

def draw_surface_views(view_name, elev, azim, outfile_stub):
    """
    生成三维曲面图（含光照、线框、三向等高线投影、不可达遮罩）并导出。
    """
    if np.all(np.isnan(Z)):
        print(f"[warn] {outfile_stub}: Z 全为 NaN，跳过绘图。")
        return

    fig = plt.figure(figsize=(8.8, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # 曲面 + 线框
    surf = ax.plot_surface(X, Y, Z_masked, rstride=1, cstride=1, linewidth=0,
                           antialiased=True, alpha=0.96, facecolors=shaded2, shade=False)
    ax.plot_wireframe(X, Y, Z_masked, rstride=8, cstride=8,
                      color='k', linewidth=0.2, alpha=0.25)

    # 三向等高线投影
    z_offset = zlim[0] - 0.08 * (zlim[1] - zlim[0])
    x_offset = xlim[0] - 0.04 * (xlim[1] - xlim[0])
    y_offset = ylim[0] - 0.04 * (ylim[1] - ylim[0])

    ax.contour(X, Y, Z_masked, zdir='z', offset=z_offset, levels=10, cmap='gray',
               linewidths=0.8, alpha=0.9)
    ax.contour(X, Y, Z_masked, zdir='x', offset=x_offset, levels=10, cmap='gray',
               linewidths=0.8, alpha=0.9)
    ax.contour(X, Y, Z_masked, zdir='y', offset=y_offset, levels=10, cmap='gray',
               linewidths=0.8, alpha=0.9)

    # 不可达区域 -> 底面半透明遮罩
    if mask_unreach.any():
        ax.contourf(X, Y, mask_unreach.astype(float), zdir='z', offset=z_offset,
                    levels=[0.5, 1.5], colors=['#bbbbbb'], alpha=0.35)

    # 标注与轴
    ax.set_title(f"4% 阈值达标孕周 3D 曲面 | {view_name}", pad=16)
    ax.set_xlabel("体质指数 | BMI", labelpad=12)
    ax.set_ylabel("孕周网格 | GA grid (display)", labelpad=12)
    ax.set_zlabel("达标孕周（周） | GA* (weeks)", labelpad=14)

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(z_offset, zlim[1])
    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    # 颜色条显式绑定到当前 3D 轴
    cbar = fig.colorbar(surf, ax=ax, shrink=0.64, aspect=18, pad=0.06)
    cbar.set_label("达标孕周（周） | GA* (weeks)\n（若 L≤阈值则不可达，底面灰色遮罩）", fontsize=11)

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"{outfile_stub}.png"), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f"{outfile_stub}.svg"), bbox_inches='tight')
    plt.close(fig)

# 导出三视角
draw_surface_views("主视角 Perspective", elev=28, azim=-125,
                   outfile_stub="Q1_Logistic_3D_ThresholdSurface_v3_main")
draw_surface_views("俯视 Top-down",   elev=89, azim=-120,
                   outfile_stub="Q1_Logistic_3D_ThresholdSurface_v3_top")
draw_surface_views("侧视 Side",        elev=12, azim=180,
                   outfile_stub="Q1_Logistic_3D_ThresholdSurface_v3_side")

print("✅ 输出完成：")
print("- figs/Q1_Logistic_3D_PredSurface.(png|svg)")
print("- figs/Q1_Logistic_3D_ThresholdSurface_v3_main.(png|svg)")
print("- figs/Q1_Logistic_3D_ThresholdSurface_v3_top.(png|svg)")
print("- figs/Q1_Logistic_3D_ThresholdSurface_v3_side.(png|svg)")
