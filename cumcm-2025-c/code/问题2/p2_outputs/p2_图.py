# -*- coding: utf-8 -*-
"""
NIPT 问题二 - 生成所有图表
1) 肘部法则
2) 浓度-孕周散点 + Logistic 拟合（分 BMI 组）
3) 首次达标孕周小提琴图
4) 误差敏感性箱线图（±0.5%）
"""

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 0. 路径与中文显示 =====
FILE_PATH = "附件.xlsx"            # 如文件名不同，请改这里
OUT_DIR   = "./p2_outputs"
FIG_DIR   = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# 中文与负号
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 1. 读表 & 列名自适配 =====
def pick_col(df, candidates, letter_hint=None):
    # 完全匹配
    for c in candidates:
        if c in df.columns:
            return c
    # 按表头是 A/B/C... 的情况
    if letter_hint is not None:
        for col in df.columns:
            if str(col).strip().upper() == letter_hint.upper():
                return col
    # 模糊包含
    low_cols = [str(c).lower() for c in df.columns]
    for c in candidates:
        cl = str(c).lower()
        for i, lc in enumerate(low_cols):
            if cl in lc:
                return df.columns[i]
    raise KeyError(f"未找到列：{candidates} 或列字母：{letter_hint}")

df = pd.read_excel(FILE_PATH)

col_code = pick_col(df, ["孕妇代码","样本编号","受检者编号","ID","code"], "B")
col_gw   = pick_col(df, ["孕周","检测孕周","孕妇本次检测时的孕周","孕周（周+天）","孕周(周+天)"], "J")
col_bmi  = pick_col(df, ["孕妇BMI","孕妇 BMI 指标","BMI"], "K")
col_yc   = pick_col(df, ["Y染色体浓度","Y 浓度","Y浓度","Y-浓度","Y_ratio","Y chromosome concentration"], "V")
try:
    col_times = pick_col(df, ["检测抽血次数","抽血次数","采血次数"], "I")
except KeyError:
    col_times = None

# ===== 2. 预处理：男胎筛选、孕周转换 =====
male_df = df[df[col_yc].notna()].copy()

def parse_gw_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)) and not isinstance(x,bool): return float(x)
    s = str(x).strip()
    s = s.replace("周","w").replace("天","d").replace("W","w").replace("D","d")
    s = s.replace("＋","+").replace("（","(").replace("）",")")
    m = re.search(r"(\d+)\s*w\s*\+?\s*(\d*)", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2)) if m.group(2)!="" else 0
        return round(w + d/7.0, 4)
    m2 = re.search(r"(\d+)\s*\+\s*(\d+)", s)
    if m2:
        w = int(m2.group(1)); d = int(m2.group(2))
        return round(w + d/7.0, 4)
    m3 = re.search(r"(\d+)\s*w$", s)
    if m3:
        return float(m3.group(1))
    try:
        return float(s)
    except:
        return np.nan

male_df["孕周(周)"] = male_df[col_gw].apply(parse_gw_to_float)
male_df = male_df[(male_df["孕周(周)"].between(8,30, inclusive="both"))].copy()
male_df = male_df.rename(columns={col_code:"孕妇代码", col_bmi:"孕妇BMI", col_yc:"Y染色体浓度"})
male_df["检测抽血次数"] = df.loc[male_df.index, col_times] if col_times else np.nan
male_df = male_df.sort_values(["孕妇代码","孕周(周)","检测抽血次数"])

# ===== 3. KMeans 聚类（K=3）+ 肘部法则图 =====
from sklearn.cluster import KMeans

first_bmi = male_df.groupby("孕妇代码", as_index=False).first()[["孕妇代码","孕妇BMI"]]
first_bmi = first_bmi.dropna(subset=["孕妇BMI"]).copy()

# 肘部法则：Inertia 随 K 变化
inertias = []
Ks = list(range(1,10))
for k in Ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(first_bmi[["孕妇BMI"]].values)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(Ks, inertias, marker="o")
plt.xlabel("聚类数 K"); plt.ylabel("簇内误差平方和 Inertia")
plt.title("肘部法则：BMI 聚类数选择")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_肘部法则_KMeans_BMI.png"), dpi=200)
plt.close()

# 采用 K=3 聚类，并按簇中心从小到大重标号
K = 3
kmeans = KMeans(n_clusters=K, n_init=20, random_state=42).fit(first_bmi[["孕妇BMI"]].values)
centers_raw = kmeans.cluster_centers_.flatten()
order = np.argsort(centers_raw)
remap = {int(order[i]): i for i in range(K)}
labels_ord = np.array([remap[int(l)] for l in kmeans.labels_])

first_bmi["BMI_cluster"] = labels_ord
first_bmi["cluster_center"] = [centers_raw[order[i]] for i in labels_ord]
mother_info = first_bmi.set_index("孕妇代码")[["孕妇BMI","BMI_cluster","cluster_center"]].copy()
male_df = male_df.join(mother_info[["BMI_cluster"]], on="孕妇代码")

# ===== 4. 计算首次达标孕周 thr_week（Y>=0.04） =====
thr_map = {}
for code, g in male_df.groupby("孕妇代码"):
    g = g.sort_values("孕周(周)")
    hit = g[g["Y染色体浓度"] >= 0.04]
    thr_map[code] = float(hit["孕周(周)"].iloc[0]) if len(hit) else None

thr_df = pd.DataFrame({"孕妇代码": list(thr_map.keys()),
                       "thr_week": list(thr_map.values())})
thr_df = thr_df.join(mother_info[["BMI_cluster"]], on="孕妇代码")
thr_hit = thr_df.dropna(subset=["thr_week"]).copy()

# 各组 P90 作为推荐周
rec_times = {}
for c in sorted(thr_hit["BMI_cluster"].unique()):
    gw = thr_hit.loc[thr_hit["BMI_cluster"]==c, "thr_week"].values
    rec_times[c] = float(np.percentile(gw, 90))

# ===== 5. 图：不同 BMI 组首次达标孕周的小提琴图 =====
plt.figure(figsize=(7,4.5))
data_plot = [thr_hit.loc[thr_hit["BMI_cluster"]==c, "thr_week"].values
             for c in sorted(rec_times.keys())]
parts = plt.violinplot(data_plot, showmeans=True, showmedians=True)
plt.xticks(range(1, len(rec_times)+1), [f"组{c}" for c in sorted(rec_times.keys())])
plt.axhline(12, ls="--", lw=1, alpha=0.5, label="12周")
plt.ylabel("首次达标孕周（周）")
plt.title("不同 BMI 组首次达标孕周分布（Y≥4%）")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig02_首次达标孕周_小提琴图.png"), dpi=200)
plt.close()

# ===== 6. 图：浓度-孕周散点 + Logistic 拟合（分组） =====
# 用 scipy 进行 Logistic 拟合；如没有 scipy，可改用 statsmodels 或简化为局部 LOESS
from scipy.optimize import curve_fit

def logistic(x, L, k, x0):
    # f(x) = L / (1 + exp[-k(x-x0)])
    return L / (1.0 + np.exp(-k*(x - x0)))

# 为了让拟合更稳健：每个组先采样（或全量）数据点
colors = {0:"#1f77b4", 1:"#2ca02c", 2:"#d62728"}  # 蓝/绿/红
labels = {0:"低BMI组", 1:"中BMI组", 2:"高BMI组"}

plt.figure(figsize=(8,5))
for c, sub in male_df.dropna(subset=["BMI_cluster"]).groupby("BMI_cluster"):
    xs = sub["孕周(周)"].values
    ys = sub["Y染色体浓度"].values
    plt.scatter(xs, ys*100, s=12, alpha=0.35, label=f"{labels.get(c,f'组{c}')}散点", c=colors.get(c,None))

    # 初值猜测：L 用该组上分位数（95%）*1.2，k=0.3，x0=组中位孕周
    L0  = np.percentile(ys, 95)*1.2 if len(ys)>0 else 0.15
    k0  = 0.30
    x00 = np.nanmedian(xs) if len(xs)>0 else 16.0
    p0 = [max(L0, 0.08), k0, x00]  # L 至少给个不太小的值，避免拟合失败

    # 拟合时对极端点作裁剪（避免负值/超大值影响）
    mask = (ys>=0) & (ys<=1)
    try:
        popt, pcov = curve_fit(logistic, xs[mask], ys[mask], p0=p0, maxfev=20000)
        # 画拟合曲线
        grid = np.linspace(xs.min()-0.5, xs.max()+0.5, 300)
        yhat = logistic(grid, *popt)
        plt.plot(grid, yhat*100, lw=2, c=colors.get(c,None), label=f"{labels.get(c,f'组{c}')}拟合")
    except Exception as e:
        # 拟合失败也不影响其他图
        print(f"[警告] 组 {c} Logistic 拟合失败：{e}")

# 4% 阈值线
plt.axhline(4.0, ls="--", lw=1, c="gray", alpha=0.8, label="4% 阈值")
plt.xlabel("孕周（周）"); plt.ylabel("Y 染色体浓度（%）")
plt.title("不同 BMI 组：Y 浓度-孕周散点与Logistic拟合")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_浓度孕周_散点与Logistic拟合.png"), dpi=200)
plt.close()

# ===== 7. 误差敏感性：在各组“推荐周(P90)”的未达标率（±0.5%）箱线图 =====
# 先在各自推荐周做“真实值”线性插值
def interp_at_week(g, week):
    g = g.sort_values("孕周(周)")
    xs = g["孕周(周)"].values
    ys = g["Y染色体浓度"].values
    if len(xs)==0: return np.nan
    if week <= xs.min(): return float(ys[xs.argmin()])
    if week >= xs.max(): return float(ys[xs.argmax()])
    return float(np.interp(week, xs, ys))

actual_at_rec = {}
for c in sorted(rec_times.keys()):
    w = rec_times[c]
    for code, g in male_df[male_df["BMI_cluster"]==c].groupby("孕妇代码"):
        actual_at_rec[(code,c)] = interp_at_week(g, w)

# 蒙特卡洛
runs = 5000
rng = np.random.default_rng(20240905)
fail_dists = {c:[] for c in sorted(rec_times.keys())}

for _ in range(runs):
    fail_count = {c:0 for c in rec_times}
    tot_count  = {c:0 for c in rec_times}
    for (code, c), val in actual_at_rec.items():
        if np.isnan(val): continue
        measured = val + rng.uniform(-0.005, 0.005)  # ±0.5%
        tot_count[c] += 1
        if measured < 0.04:
            fail_count[c] += 1
    for c in rec_times:
        if tot_count[c]>0:
            fail_dists[c].append(fail_count[c]/tot_count[c]*100.0)

# 箱线图
plt.figure(figsize=(7,4.5))
data_box = [fail_dists[c] for c in sorted(rec_times.keys())]
plt.boxplot(data_box, positions=np.arange(1,len(data_box)+1), widths=0.6, showmeans=True)
plt.xticks(np.arange(1,len(data_box)+1), [f"组{c}" for c in sorted(rec_times.keys())])
plt.ylabel("推荐周未达标率（%）")
plt.title("±0.5% 测量误差下推荐周未达标率分布（蒙特卡洛）")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig04_推荐周未达标率_误差敏感性箱线图.png"), dpi=200)
plt.close()

print("✅ 所有图表已生成：", os.path.abspath(FIG_DIR))
