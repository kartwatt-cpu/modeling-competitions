# -*- coding: utf-8 -*-
"""
胎儿Y染色体浓度 vs 孕周（按BMI分组）
- 数据来源: 附件.xlsx
- 男胎孕妇 (Y染色体浓度非空)
- KMeans分BMI三组: 低/中/高
- 绘制散点 + Logistic拟合曲线
- 添加4%阈值线
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# -----------------------------
# 0) 设置
# -----------------------------
FILE_PATH = "附件.xlsx"   # Excel 文件名
Y_THRESHOLD = 0.04       # 4% 阈值

# 中文字体（根据你电脑实际情况，常用：SimHei、Microsoft YaHei、PingFang SC）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "font.size": 13,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -----------------------------
# 1) 读表 & 列选择
# -----------------------------
def pick_col(df, candidates, letter_hint=None):
    for c in candidates:
        if c in df.columns:
            return c
    if letter_hint:
        for col in df.columns:
            if str(col).strip().upper() == letter_hint.upper():
                return col
    low_cols = [str(c).lower() for c in df.columns]
    for c in candidates:
        cl = str(c).lower()
        for i, lc in enumerate(low_cols):
            if cl in lc:
                return df.columns[i]
    raise KeyError(f"找不到列: {candidates}")

df = pd.read_excel(FILE_PATH)

col_code = pick_col(df, ["孕妇代码","样本编号","受检者编号","ID"], "B")
col_gw   = pick_col(df, ["孕周","检测孕周","孕妇本次检测时的孕周"], "J")
col_bmi  = pick_col(df, ["孕妇BMI","孕妇 BMI 指标","BMI"], "K")
col_yc   = pick_col(df, ["Y染色体浓度","Y 浓度","Y_ratio"], "V")

# 男胎
df_m = df[df[col_yc].notna()].copy()

# 孕周转换
def parse_gw_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    s = str(x).strip().replace("周","w").replace("天","d").replace("＋","+")
    m = re.search(r"(\d+)w\+?(\d*)", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2)) if m.group(2)!="" else 0
        return w + d/7.0
    m2 = re.search(r"(\d+)\+(\d+)", s)
    if m2:
        return int(m2.group(1)) + int(m2.group(2))/7.0
    try:
        return float(s)
    except:
        return np.nan

df_m = df_m.rename(columns={col_code:"code", col_bmi:"BMI", col_yc:"Yfrac"})
df_m["孕周"] = df_m[col_gw].apply(parse_gw_to_float)
df_m = df_m[df_m["孕周"].between(10,30)].copy()

# -----------------------------
# 2) KMeans 分组
# -----------------------------
df_m = df_m.sort_values(["code","孕周"])
first_bmi = df_m.groupby("code", as_index=False).first()[["code","BMI"]].dropna()

K = 3
km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(first_bmi[["BMI"]].values)
centers = km.cluster_centers_.flatten()
order = np.argsort(centers)
remap = {int(order[i]): i for i in range(K)}
labels_ord = np.array([remap[int(l)] for l in km.labels_])

first_bmi["group"] = labels_ord
df_m = df_m.merge(first_bmi[["code","group"]], on="code", how="left")

group_info = {
    0: {"label":"低 BMI 组", "c":"#1f77b4"},
    1: {"label":"中 BMI 组", "c":"#2ca02c"},
    2: {"label":"高 BMI 组", "c":"#d62728"},
}

# -----------------------------
# 3) Logistic 拟合
# -----------------------------
def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k*(x - x0)))

def fit_logistic(x, y):
    mask = (y>=0)&(y<=1)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x)<5: return None
    L0 = max(np.percentile(y,95)*1.2, 0.08)
    k0, x00 = 0.3, np.median(x)
    try:
        popt, _ = curve_fit(logistic, x, y, p0=[L0,k0,x00], maxfev=20000)
        return popt
    except: return None

# -----------------------------
# 4) 绘图
# -----------------------------
fig, ax = plt.subplots(figsize=(10,6))
grid = np.linspace(df_m["孕周"].min(), df_m["孕周"].max(), 400)

for g in [0,1,2]:
    sub = df_m[df_m["group"]==g]
    ax.scatter(sub["孕周"], sub["Yfrac"]*100, s=18, alpha=0.5,
               c=group_info[g]["c"], label=group_info[g]["label"])
    pars = fit_logistic(sub["孕周"].values, sub["Yfrac"].values)
    if pars is not None:
        ax.plot(grid, logistic(grid,*pars)*100, lw=2.5, c=group_info[g]["c"])

# 阈值线
ax.axhline(Y_THRESHOLD*100, ls="--", lw=1.8, c="gray")
ax.text(df_m["孕周"].min()+0.2, Y_THRESHOLD*100+0.3, "4% 阈值线", color="gray")

ax.set_xlim(10,30); ax.set_ylim(0,25)
ax.set_xlabel("孕周（周）"); ax.set_ylabel("Y 染色体浓度（%）")
ax.set_title("不同 BMI 组孕周与 Y 染色体浓度关系图")
ax.legend()

plt.tight_layout()
plt.savefig("不同BMI组_Y浓度孕周关系图.png", dpi=220)
plt.show()
