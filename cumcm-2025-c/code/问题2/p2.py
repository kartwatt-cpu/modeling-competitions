
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILE_PATH = "附件.xlsx"  # 改成你的 Excel 文件名或绝对路径
SAVE_DIR = "./p2_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def pick_col(df, candidates, letter_hint=None):
  
    # 1) 完全匹配
    for c in candidates:
        if c in df.columns:
            return c

    if letter_hint is not None:
        for col in df.columns:
            if str(col).strip().upper() == letter_hint.upper():
                return col

    low_cols = [str(c).lower() for c in df.columns]
    for c in candidates:
        cl = str(c).lower()
        for i, lc in enumerate(low_cols):
            if cl in lc:
                return df.columns[i]
    raise KeyError(f"未能在表头中找到字段：{candidates} 或列字母：{letter_hint}")

df = pd.read_excel(FILE_PATH)


col_code = pick_col(df, ["孕妇代码", "样本编号", "受检者编号", "ID", "code"], "B")
col_gw   = pick_col(df, ["孕周", "检测孕周", "孕妇本次检测时的孕周", "孕周（周+天）", "孕周(周+天)"], "J")
col_bmi  = pick_col(df, ["孕妇BMI", "孕妇 BMI 指标", "BMI"], "K")
col_yc   = pick_col(df, ["Y染色体浓度", "Y 浓度", "Y浓度", "Y-浓度", "Y chromosome concentration", "Y_ratio"], "V")

try:
    col_times = pick_col(df, ["检测抽血次数", "抽血次数", "采血次数"], "I")
except KeyError:
    col_times = None



male_df = df[df[col_yc].notna()].copy()

def parse_gw_to_float(x):

    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()

    s = s.replace("周", "w").replace("天", "d").replace("W", "w").replace("D", "d")
    s = s.replace("＋", "+").replace("（", "(").replace("）", ")")

    m = re.search(r"(\d+)\s*w\s*\+?\s*(\d*)", s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) != "" else 0
        return round(w + d/7.0, 4)
    m2 = re.search(r"(\d+)\s*\+\s*(\d+)", s)
    if m2:
        w = int(m2.group(1))
        d = int(m2.group(2))
        return round(w + d/7.0, 4)

    m3 = re.search(r"(\d+)\s*w$", s)
    if m3:
        return float(m3.group(1))

    try:
        return float(s)
    except:
        return np.nan

male_df["孕周(周)"] = male_df[col_gw].apply(parse_gw_to_float)


male_df = male_df[(male_df["孕周(周)"].between(8, 30, inclusive="both"))].copy()


keep_cols = ["孕妇代码", "孕周(周)", "孕妇BMI", "Y染色体浓度"]
male_df = male_df.rename(columns={col_code:"孕妇代码", col_bmi:"孕妇BMI", col_yc:"Y染色体浓度"})

if col_times and col_times in df.columns:
    male_df["检测抽血次数"] = df.loc[male_df.index, col_times]
else:
    male_df["检测抽血次数"] = np.nan  # 没有就留空

#KMeans 对 BMI 聚类（K=3，且按中心值排序重标号）
from sklearn.cluster import KMeans


male_df = male_df.sort_values(["孕妇代码", "孕周(周)", "检测抽血次数"])
first_bmi = male_df.groupby("孕妇代码", as_index=False).first()[["孕妇代码", "孕妇BMI"]].copy()


first_bmi = first_bmi[~first_bmi["孕妇BMI"].isna()].copy()

K = 3
kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
labels_raw = kmeans.fit_predict(first_bmi[["孕妇BMI"]].values)

#按簇中心从小到大重排标签（让 0=低BMI, 1=中BMI, 2=高BMI）
centers = kmeans.cluster_centers_.flatten()
order = np.argsort(centers)

remap = {int(order[i]): i for i in range(K)}
labels_ord = np.array([remap[int(l)] for l in labels_raw])

first_bmi["BMI_cluster"] = labels_ord
first_bmi["cluster_center"] = [centers[order[i]] for i in labels_ord]


mother_info = first_bmi.set_index("孕妇代码")[["孕妇BMI","BMI_cluster","cluster_center"]].copy()


male_df = male_df.join(mother_info[["BMI_cluster"]], on="孕妇代码")

#计算每位孕妇的“首次达标孕周”（Y>=0.04）
thr_map = {}  # {孕妇代码: 首次达标孕周 或 None}
for code, g in male_df.groupby("孕妇代码"):
    g = g.sort_values("孕周(周)")
    hit = g[g["Y染色体浓度"] >= 0.04]
    thr_map[code] = float(hit["孕周(周)"].iloc[0]) if len(hit) else None

thr_df = pd.DataFrame({
    "孕妇代码": list(thr_map.keys()),
    "thr_week": list(thr_map.values())
})
thr_df = thr_df.join(mother_info[["BMI_cluster"]], on="孕妇代码")

n_never = thr_df["thr_week"].isna().sum()
print(f"在观测范围内未达到 4% 阈值的孕妇：{n_never} 人")


thr_hit = thr_df.dropna(subset=["thr_week"]).copy()

#各 BMI 组的 P90 推荐时点 
rec_times = {}         
group_stats = []       

for c in sorted(thr_hit["BMI_cluster"].unique()):
    gw = thr_hit.loc[thr_hit["BMI_cluster"]==c, "thr_week"].values
    if len(gw)==0:
        continue
    p50 = np.percentile(gw, 50)
    p90 = np.percentile(gw, 90)
    rec_times[c] = float(p90)
    group_stats.append([c, len(gw), float(p50), float(p90)])

res_table = pd.DataFrame(group_stats, columns=["BMI组","达标样本数","达标孕周中位数","推荐时点P90(周)"])
print("\n各 BMI 组推荐检测时点（基于 P90）：")
print(res_table)


res_table.to_csv(os.path.join(SAVE_DIR, "BMI组_推荐检测时点_P90.csv"), index=False, encoding="utf-8-sig")

#可视化：阈值达标孕周的分布（箱线/小提琴）
plt.figure(figsize=(8,5))
data_plot = [thr_hit.loc[thr_hit["BMI_cluster"]==c, "thr_week"].values for c in sorted(rec_times.keys())]
plt.violinplot(data_plot, showmeans=True, showmedians=True)
plt.xticks(range(1, len(rec_times)+1), [f"组{c}" for c in sorted(rec_times.keys())])
plt.axhline(12, ls="--", lw=1, alpha=0.4, label="12周")
plt.ylabel("首次达标孕周（周）")
plt.title("不同 BMI 组首次达标孕周分布（Y≥4%）")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "不同BMI组_首次达标孕周_小提琴图.png"), dpi=200)
plt.close()


def est_fail_rate_at_week(df_group, week):
    """
    估算该组在给定孕周 week 的未达标比例：
    使用每个个体“最近的检测值（向前/向后最近）”做近似（亦可改为线性插值）。
    """
    fails, total = 0, 0
    for code, g in df_group.groupby("孕妇代码"):
        g = g.sort_values("孕周(周)")
        # 取离 week 最近的一次检测值
        idx = (g["孕周(周)"] - week).abs().idxmin()
        val = float(g.loc[idx, "Y染色体浓度"])
        if not np.isnan(val):
            total += 1
            if val < 0.04:
                fails += 1
    return (fails, total, fails/total if total>0 else np.nan)

fail_rows = []
for c in sorted(rec_times.keys()):
    sub = male_df[male_df["BMI_cluster"]==c]
    w  = rec_times[c]
    fails, total, rate = est_fail_rate_at_week(sub, w)
    fail_rows.append([c, w, total, fails, round(rate*100,2)])
fail_df = pd.DataFrame(fail_rows, columns=["BMI组","推荐周","样本数","未达标数","未达标比例(%)"])
print("\n经验未达标比例（基于最近测点近似）：")
print(fail_df)
fail_df.to_csv(os.path.join(SAVE_DIR, "推荐周_经验未达标比例.csv"), index=False, encoding="utf-8-sig")

#误差敏感性（±0.5%）蒙特卡洛

from collections import defaultdict

def interp_conc_at_week(g, week):

    g = g.sort_values("孕周(周)")
    xs = g["孕周(周)"].values
    ys = g["Y染色体浓度"].values
    if len(xs)==0:
        return np.nan
    if week <= xs.min():
        return float(ys[xs.argmin()])
    if week >= xs.max():
        return float(ys[xs.argmax()])
    # 线性插值
    return float(np.interp(week, xs, ys))


actual_at_rec = {}
cluster_id_of = {}
for c in sorted(rec_times.keys()):
    w = rec_times[c]
    for code, g in male_df[male_df["BMI_cluster"]==c].groupby("孕妇代码"):
        actual_at_rec[(code, c)] = interp_conc_at_week(g, w)
        cluster_id_of[code] = c

runs = 5000
rng = np.random.default_rng(20240905)
fail_dists = defaultdict(list)  # {cluster: [rate, ...]}

for _ in range(runs):
    fail_count = defaultdict(int)
    tot_count  = defaultdict(int)

    for (code, c), true_val in actual_at_rec.items():
        if np.isnan(true_val):
            continue
        #加测量误差（±0.5% = ±0.005 绝对比值）
        measured = true_val + rng.uniform(-0.005, 0.005)
        tot_count[c] += 1
        if measured < 0.04:
            fail_count[c] += 1
    for c in sorted(rec_times.keys()):
        if tot_count[c] > 0:
            fail_dists[c].append(fail_count[c]/tot_count[c])


sens_rows = []
for c in sorted(rec_times.keys()):
    arr = np.array(fail_dists[c])
    mean = arr.mean()*100
    p5, p95 = np.percentile(arr, [5, 95])*100
    sens_rows.append([c, round(mean,2), round(p5,2), round(p95,2)])
sens_df = pd.DataFrame(sens_rows, columns=["BMI组","未达标率均值(%)","P5(%)","P95(%)"])
print("\n±0.5% 误差下未达标率的蒙特卡洛结果：")
print(sens_df)
sens_df.to_csv(os.path.join(SAVE_DIR, "误差敏感性_未达标率_蒙特卡洛.csv"), index=False, encoding="utf-8-sig")

# 可视化分布
plt.figure(figsize=(8,5))
positions = np.arange(1, len(rec_times)+1)
data = [np.array(fail_dists[c])*100 for c in sorted(rec_times.keys())]
plt.boxplot(data, positions=positions, widths=0.6, showmeans=True)
plt.xticks(positions, [f"组{c}" for c in sorted(rec_times.keys())])
plt.ylabel("推荐周未达标率（%）")
plt.title("±0.5% 误差下未达标率的分布（推荐检测周）")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "误差敏感性_未达标率_箱线图.png"), dpi=200)
plt.close()

#辅助：各组 BMI 范围/人数小结
bmi_ranges = male_df.groupby("BMI_cluster")["孕妇BMI"].agg(["count","min","max","mean"]).reset_index()
bmi_ranges.to_csv(os.path.join(SAVE_DIR, "BMI组_范围与人数.csv"), index=False, encoding="utf-8-sig")

print("\n已完成。结果文件与图表保存在：", os.path.abspath(SAVE_DIR))
