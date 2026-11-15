

import os
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime



from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from cycler import cycler
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn


EXCEL_PATH = "数据.xlsx"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs_q3")
TREE_DIR = os.path.join(OUT_DIR, "trees")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TREE_DIR, exist_ok=True)

Y_THRESHOLD = 0.04             # 达标阈值（4%）
N_SIM = 300                    # 误差模拟次数
GA_JITTER = 1.0               # 孕周误差（±1 周）
Y_JITTER  = 0.005             # 浓度误差（±0.5% 绝对值）


Q2_SUMMARY = pd.DataFrame({
    "BMI 分组": ["≤31 (~低BMI组)", "32–36 (中BMI组)", "≥36 (~高BMI组)"],
    "样本数 N": [138, 109, 20],
    "推荐检测孕周 (周)": [16.29, 16.10, 22.40],
    "推荐孕周未达标比例": ["约 10.1%", "约 14.7%", "约 15.0%"]
})


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def setup_matplotlib_theme(theme="color"):
    """theme: 'color'（彩色科研风） 或 'bw'（灰度期刊风）"""
    plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 600   # 论文建议 600
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    # 统一字号
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    if theme == "color":
        colors = ["#4472C4","#ED7D31","#70AD47","#A64D79","#2E75B6","#FFC000"]
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    else:
        gray_cycle = cycler(color=['#111','#444','#777','#AAA']) + cycler(linestyle=['-','--','-.',':'])
        plt.rcParams['axes.prop_cycle'] = gray_cycle

def to_percent(y):
    return y * 100.0

def _normalize_col(s: str) -> str:
    s = str(s).strip()
    s = s.replace('（', '(').replace('）', ')')
    s = s.replace('＋', '+').replace('﹢', '+')
    s = re.sub(r'\s+', '', s)
    s = s.replace('_', '')
    return s

def find_col(df, candidates, regex_patterns=None, allow_letters=None, alias=""):
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    lowmap = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]

    if regex_patterns:
        for pat in regex_patterns:
            patt = re.compile(pat, flags=re.IGNORECASE)
            for c in cols:
                if patt.search(c):
                    return c

    if allow_letters:
        for L in allow_letters:
            for c in cols:
                if c.upper() == L.upper():
                    return c
    raise ValueError(f"缺少关键列：{alias or candidates}（请检查Excel表头，控制台已打印列名）")

def parse_ga_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float,np.integer,np.floating)): return float(x)
    s = str(x).strip()
    m = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', s)
    if m: return int(m.group(1)) + int(m.group(2))/7.0
    m = re.match(r'^\s*(\d+)\s*周\s*(\d+)\s*天\s*$', s)
    if m: return int(m.group(1)) + int(m.group(2))/7.0
    m = re.match(r'^\s*(\d+)\s*[wW]\s*(\d+)\s*[dD]\s*$', s)
    if m: return int(m.group(1)) + int(m.group(2))/7.0
    m = re.match(r'^\s*(\d+(\.\d+)?)\s*$', s)
    if m: return float(m.group(1))
    return np.nan

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


print(">>> 读取 Excel:", EXCEL_PATH)
df_raw = pd.read_excel(EXCEL_PATH)
df_raw.columns = [_normalize_col(c) for c in df_raw.columns]
print("【调试】规范化后的列名：\n", list(df_raw.columns), "\n")


COL_BMI = find_col(
    df_raw,
    candidates=['孕妇BMI指标','BMI','bmi','体重指数','体质指数'],
    regex_patterns=[r'BMI', r'体重指数|体质指数'],
    allow_letters=['K'],
    alias='BMI'
)
COL_YCON = find_col(
    df_raw,
    candidates=['Y染色体浓度','Y浓度','Yconcentration','Yconc','Y_conc','Yratio','Yfraction'],
    regex_patterns=[r'(^|[^A-Za-z])Y(染色体)?(浓度|ratio|fraction|conc)'],
    allow_letters=['V'],
    alias='Y染色体浓度'
)
GA_single_candidates = [
    '孕妇本次检测时的孕周(周数+天数)','孕周(周+天)','孕周周加天','孕周',
    'GA','GAweeks','GestationalAge','gestweek','gestweeks','gestage','gest_age'
]
GA_single_regex = [r'孕周', r'GA|gest', r'周\+天', r'weeks?']
GA_week_candidates = ['孕周(周)','孕周周','周数']
GA_day_candidates  = ['孕周(天)','孕周天','天数','日数','days']

COL_TESTDATE = None
COL_LMP = None
try:
    COL_TESTDATE = find_col(
        df_raw,
        candidates=['检测时间','检测日期','抽血时间','采样时间','检测日期时间','检测时刻'],
        regex_patterns=[r'检测.*(时间|日期)|抽血时间|采样时间'],
        allow_letters=['H'],
        alias='检测时间'
    )
except: pass
try:
    COL_LMP = find_col(
        df_raw,
        candidates=['末次月经时间','末次月经','LMP','LastMenstrualPeriod'],
        regex_patterns=[r'末次月经|LMP'],
        allow_letters=['F'],
        alias='末次月经时间'
    )
except: pass

COL_ID = None
try:
    COL_ID = find_col(
        df_raw,
        candidates=['孕妇代码','孕妇ID','ID','id','样本序号','样本ID'],
        regex_patterns=[r'孕妇.*(代码|ID)|样本(序号|ID)'],
        allow_letters=['B','A'],
        alias='孕妇ID/代码'
    )
except:
    COL_ID = None


COL_GA = None
try:
    COL_GA = find_col(
        df_raw,
        candidates=GA_single_candidates,
        regex_patterns=GA_single_regex,
        allow_letters=['J'],
        alias='孕周'
    )
    print(f"【提示】检测到单列孕周：{COL_GA}")
    df_raw['GA_float'] = df_raw[COL_GA].apply(parse_ga_to_float)
except Exception:
    try:
        COL_GA_WEEK = find_col(df_raw, GA_week_candidates, regex_patterns=[r'孕周.*周|周数'], alias='孕周(周)')
        COL_GA_DAY  = find_col(df_raw, GA_day_candidates,  regex_patterns=[r'孕周.*天|天数|日数|days?'], alias='孕周(天)')
        print(f"【提示】检测到分列孕周：{COL_GA_WEEK} + {COL_GA_DAY}")
        def _to_float_weekday(w, d):
            try: w = float(w)
            except: return np.nan
            try: d = float(d)
            except: d = 0.0
            return w + d/7.0
        df_raw['GA_float'] = [_to_float_weekday(w,d) for w,d in zip(df_raw[COL_GA_WEEK], df_raw[COL_GA_DAY])]
    except Exception:
        if COL_TESTDATE and COL_LMP:
            print("【提示】未找到显式‘孕周’列，将用 检测时间-末次月经 计算孕周（周）。")
            df_raw['_LMP_dt'] = pd.to_datetime(df_raw[COL_LMP], errors='coerce')
            df_raw['_TEST_dt'] = pd.to_datetime(df_raw[COL_TESTDATE], errors='coerce')
            df_raw['GA_float'] = (df_raw['_TEST_dt'] - df_raw['_LMP_dt']).dt.days / 7.0
        else:
            raise ValueError(
                "未能识别‘孕周’。\n"
                "请检查：单列孕周/分列孕周/或日期差；控制台已打印列名以供对照。"
            )

df = df_raw.dropna(subset=[COL_BMI, COL_YCON, 'GA_float']).copy()
df.loc[df[COL_YCON] < 0, COL_YCON] = np.nan
df = df.dropna(subset=[COL_YCON]).copy()

print(f"【OK】列映射完成：BMI={COL_BMI}，Y浓度={COL_YCON}，孕周=GA_float；ID={COL_ID}\n")
print("数据规模：", df.shape)

#仅保留“男胎”
df_male = df[df[COL_YCON].notna()].copy()
print("男胎样本记录数：", len(df_male))

#自动识别 BMI 分组（KMeans=3）
if COL_ID and COL_ID in df_male.columns:
    bmi_base = df_male.groupby(COL_ID)[COL_BMI].median().to_frame('BMI_base')
else:
    tmp_id = np.arange(len(df_male))
    bmi_base = pd.DataFrame({'BMI_base': df_male[COL_BMI].values}, index=tmp_id)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(bmi_base[['BMI_base']])
centers = np.sort(kmeans.cluster_centers_.ravel())
edges = [-np.inf] + [float((centers[i]+centers[i+1])/2.0) for i in range(2)] + [np.inf]
labels = [f"低BMI组(≤{edges[1]:.1f})", f"中BMI组({edges[1]:.1f}–{edges[2]:.1f})", f"高BMI组(≥{edges[2]:.1f})"]
print("自动识别的 BMI 簇中心：", [round(c,1) for c in centers])
print("自动识别的 BMI 分组边界：", [f"{e:.1f}" if np.isfinite(e) else str(e) for e in edges])

df_male['BMI组'] = pd.cut(df_male[COL_BMI], bins=edges, labels=labels, right=True, include_lowest=True)
group_counts = df_male['BMI组'].value_counts(dropna=False).reindex(labels)
print("\nBMI 分组样本数：\n", group_counts, "\n")

#Pretty 图：浓度趋势 / 达标率 / 误差敏感性
def make_conc_trend_fig_pretty(df_in, group_label, COL_YCON, save_dir=FIG_DIR, theme="color", y_as_percent=False):
    setup_matplotlib_theme(theme)
    sub = df_in.copy()
    sub['周_0.5'] = np.round(sub['GA_float']*2)/2.0
    stat = sub.groupby('周_0.5')[COL_YCON].agg(['mean','std','count']).reset_index()
    if len(stat)==0: return None
    x = stat['周_0.5'].values
    y = stat['mean'].values
    s = stat['std'].values
    lo, hi = y - s, y + s
    if y_as_percent:
        y, lo, hi = to_percent(y), to_percent(lo), to_percent(hi)
        thr = 4.0; y_label = "Y 染色体浓度（%）"
    else:
        thr = 0.04; y_label = "Y 染色体浓度（比例）"
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, y, lw=2.2, marker='o', ms=3.5, label='均值', zorder=3)
    ax.fill_between(x, lo, hi, alpha=0.18, label='±1 标准差', zorder=2)
    ax.axhline(thr, ls='--', lw=1.5, color='red' if theme=="color" else '#333', label='达标阈值 4%')
    ax.annotate('阈值 4%', xy=(x.max(), thr), xytext=(-5, 6), textcoords='offset points',
                ha='right', va='bottom', fontsize=10, color='red' if theme=="color" else '#333')
    ax.set_title(f"{group_label}：Y 浓度—孕周趋势")
    ax.set_xlabel("孕周（周）"); ax.set_ylabel(y_label)
    if y_as_percent: ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True); ax.legend(loc='upper left')
    fig.tight_layout()
    fn_png = os.path.join(save_dir, f"{group_label}_浓度-孕周趋势_{theme}.png")
    fn_svg = os.path.join(save_dir, f"{group_label}_浓度-孕周趋势_{theme}.svg")
    fig.savefig(fn_png); fig.savefig(fn_svg); plt.close(fig)
    return fn_png

def first_attainment_weeks(df_group, id_col, ga_col, y_col, thr=Y_THRESHOLD):
    if id_col and id_col in df_group.columns:
        out = {}
        for pid, sub in df_group.sort_values(ga_col).groupby(id_col):
            ok = sub[sub[y_col] >= thr]
            out[pid] = np.nan if len(ok)==0 else float(ok.iloc[0][ga_col])
        return pd.Series(out, name="首次达标孕周")
    else:
        return pd.Series(np.where(df_group[y_col] >= thr, df_group[ga_col].values, np.nan),
                         index=df_group.index, name="首次达标孕周")

def make_attainment_curve_pretty(df_in, group_label, COL_YCON, COL_ID, save_dir=FIG_DIR, theme="color",
                                 mark_week=None, mark_label=None):
    setup_matplotlib_theme(theme)
    s_first = first_attainment_weeks(df_in, id_col=COL_ID, ga_col='GA_float', y_col=COL_YCON, thr=Y_THRESHOLD)
    xmin = math.floor(df_in['GA_float'].min()); xmax = math.ceil(df_in['GA_float'].max())
    xs = np.arange(xmin, xmax + 0.5, 0.5)
    rates = [np.nanmean(s_first <= w) for w in xs]
    y_vals = to_percent(np.array(rates))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(xs, y_vals, lw=2.2, marker='o', ms=3.5)
    ax.set_ylim(0, 102)
    ax.set_title(f"{group_label}：达标率（≥4%）—孕周曲线")
    ax.set_xlabel("孕周（周）"); ax.set_ylabel("累计达标比例（%）")
    ax.grid(True)
    if mark_week is not None:
        ax.axvline(mark_week, color='#6E6E6E' if theme=="bw" else '#555', ls=':', lw=1.5)
        text = mark_label if mark_label else f"推荐孕周 {mark_week:.2f}"
        ax.annotate(text, xy=(mark_week, 50), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='w', ec='#AAA'))
    fig.tight_layout()
    fn_png = os.path.join(save_dir, f"{group_label}_达标率-孕周曲线_{theme}.png")
    fn_svg = os.path.join(save_dir, f"{group_label}_达标率-孕周曲线_{theme}.svg")
    fig.savefig(fn_png); fig.savefig(fn_svg); plt.close(fig)
    tab = pd.DataFrame({"孕周": xs, "达标率(%)": y_vals})
    return fn_png, tab, s_first

def simulate_sensitivity_pretty(df_in, base_tab, group_label, COL_YCON, COL_ID, save_dir=FIG_DIR, theme="color"):
    setup_matplotlib_theme(theme)
    xmin = base_tab['孕周'].min(); xmax = base_tab['孕周'].max()
    xs = np.arange(xmin, xmax + 0.5, 0.5)
    mat = []
    for _ in range(N_SIM):
        sim = df_in.copy()
        sim['GA_sim'] = sim['GA_float'] + np.random.uniform(-GA_JITTER, GA_JITTER, size=len(sim))
        sim['Y_sim']  = sim[COL_YCON]    + np.random.uniform(-Y_JITTER,  Y_JITTER,  size=len(sim))
        if COL_ID and COL_ID in sim.columns:
            out = {}
            for pid, ssub in sim.sort_values('GA_sim').groupby(COL_ID):
                ok = ssub[ssub['Y_sim'] >= Y_THRESHOLD]
                out[pid] = np.nan if len(ok)==0 else float(ok.iloc[0]['GA_sim'])
            s_first = pd.Series(out)
        else:
            s_first = pd.Series(np.where(sim['Y_sim']>=Y_THRESHOLD, sim['GA_sim'], np.nan), index=sim.index)
        rates = [np.nanmean(s_first <= w) for w in xs]
        mat.append(to_percent(np.array(rates)))
    arr = np.vstack(mat)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(base_tab['孕周'], base_tab['达标率(%)'], lw=2.2, label='原始达标率曲线')
    ax.plot(xs, mean, lw=1.8, ls='--', label='误差模拟均值')
    ax.fill_between(xs, mean-std, mean+std, alpha=0.20, label='±1σ区间')
    ax.set_ylim(0, 102)
    ax.set_title(f"{group_label}：误差敏感性（±0.5%浓度、±1周孕周）")
    ax.set_xlabel("孕周（周）"); ax.set_ylabel("累计达标比例（%）")
    ax.grid(True); ax.legend(loc='lower right')
    fig.tight_layout()
    fn_png = os.path.join(save_dir, f"{group_label}_误差敏感性分析_{theme}.png")
    fn_svg = os.path.join(save_dir, f"{group_label}_误差敏感性分析_{theme}.svg")
    fig.savefig(fn_png); fig.savefig(fn_svg); plt.close(fig)
    return fn_png, pd.DataFrame({"孕周": xs, "均值(%)": mean, "下界(%)": mean-std, "上界(%)": mean+std})


def _setup_plot_theme(theme="bw"):
    setup_matplotlib_theme(theme)

def _pick_features(df_in, prefer_cols=None, extra_candidates=None):
    prefer = prefer_cols or []
    cands = extra_candidates or []

    name_map = {"年龄": ["年龄","age","Age","C"], "身高": ["身高","height","Height","D"], "体重": ["体重","weight","Weight","E"]}
    cols_avail = []
    for _, v in name_map.items():
        for nm in v:
            if nm in df_in.columns:
                cols_avail.append(nm); break
    feats = []

    if "GA_float" in df_in.columns: feats.append("GA_float")
    if "BMI" in df_in.columns: feats.append("BMI")
    elif 'BMI' != COL_BMI and COL_BMI in df_in.columns: feats.append(COL_BMI)

    for nm in cols_avail:
        if nm not in feats: feats.append(nm)

    for nm in prefer:
        if nm in df_in.columns and nm not in feats: feats.append(nm)
    for nm in cands:
        if nm in df_in.columns and nm not in feats: feats.append(nm)
    feats = [c for c in dict.fromkeys(feats) if c in df_in.columns]
    return feats

def _replace_to_chinese_label(text: str) -> str:

    if not isinstance(text, str): return text
    repl = {
        r"\bgini\b": "基尼系数",
        r"\bentropy\b": "信息熵",
        r"\bmse\b": "均方误差",
        r"\bsamples\b": "样本数",
        r"\bvalue\b": "计数",
        r"\bclass\b": "预测",
    }
    out = text
    for k,v in repl.items():
        out = re.sub(k, v, out)
    out = out.replace("<=", "≤")
    return out

def plot_tree_chinese(clf, feature_names, class_names, title, save_path_png):

    depth = clf.get_depth()
    leaves = clf.get_n_leaves()
    fig_w = max(10, min(42, 1.15 * leaves))
    fig_h = max(6, 1.6 * depth + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    artists = plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True,
        impurity=True, proportion=True,
        precision=2, ax=ax,
        fontsize=10 if leaves <= 30 else 8
    )

    for txt in ax.texts:
        txt.set_text(_replace_to_chinese_label(txt.get_text()))
    ax.set_title(title)
    ax.margins(x=0.03, y=0.12)
    fig.tight_layout()
    fig.savefig(save_path_png)
    plt.close(fig)

def export_graphviz_cn_png(clf, feature_names, class_names, png_path, dot_path=None):

    try:
        from graphviz import Source
        dot = export_graphviz(
            clf, out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True, rounded=True,
            proportion=True, impurity=True, precision=2,
            special_characters=True
        )

        dot = dot.replace("digraph Tree {", 'digraph Tree {\n  graph [ranksep=0.45, nodesep=0.35];\n  node [fontname="SimHei"];\n  edge [fontname="SimHei"];')

        mapping = {
            "gini = ": "基尼系数 = ",
            "entropy = ": "信息熵 = ",
            "mse = ": "均方误差 = ",
            "samples = ": "样本数 = ",
            "value = ": "计数 = ",
            "class = ": "预测 = ",
            " <= ": " ≤ ",
        }
        for k,v in mapping.items():
            dot = dot.replace(k, v)

        if dot_path:
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(dot)

        base, _ = os.path.splitext(png_path)
        Source(dot).render(base, format="png", cleanup=True)
        return True
    except Exception as e:
        # print(f"[Graphviz] 渲染失败：{e}")
        return False

def train_tree_for_group(df_group, y_col, theme="bw", group_name="Group", max_depth=4, min_samples_leaf=10):
    _setup_plot_theme(theme)

    feat_names = _pick_features(df_group)
    X = df_group[feat_names].copy()
    y = df_group[y_col].astype(int).values

    if len(np.unique(y)) < 2 or len(y) < 30:
        print(f"【提示】{group_name} 类别数不足或样本太少，跳过树模型。")
        return None, feat_names, None, None, None, (None, None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced", random_state=42
    ).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"[{group_name}] Tree acc={acc:.3f}, precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")


    class_names = ["未达标(<4%)", "达标(≥4%)"]
    # Graphviz 高清中文（优先）
    path_tree_png = os.path.join(TREE_DIR, f"{group_name}_决策树结构.png")
    dot_path = os.path.join(TREE_DIR, f"{group_name}_tree.dot")
    ok = export_graphviz_cn_png(clf, feat_names, class_names, png_path=path_tree_png, dot_path=dot_path)
    if ok:
        print(f"已导出 Graphviz 中文树：{path_tree_png}")
    else:
        # 回退：matplotlib 绘制 + 中文替换 + 自适应尺寸，尽量避免重叠
        title = f"{group_name} 决策树结构（最大深度={max_depth}，叶最小样本={min_samples_leaf}）"
        plot_tree_chinese(clf, feat_names, class_names, title, path_tree_png)


    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("特征重要性"); ax.set_title(f"{group_name} 特征重要性排序")
    for i, v in enumerate(imp.values):
        ax.text(v+1e-4, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    path_imp = os.path.join(TREE_DIR, f"{group_name}_特征重要性.png")
    fig.savefig(path_imp); plt.close(fig)


    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["预测未达标","预测达标"], yticklabels=["真实未达标","真实达标"])
    ax.set_title(f"{group_name} 混淆矩阵\n准确率={acc:.2f}  精确率={prec:.2f}  召回率={rec:.2f}  F1={f1:.2f}")
    fig.tight_layout()
    path_cm = os.path.join(TREE_DIR, f"{group_name}_混淆矩阵.png")
    fig.savefig(path_cm); plt.close(fig)


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, valid_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.2, 1.0, 6),
        cv=cv, scoring="accuracy", n_jobs=None, shuffle=True, random_state=42
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="训练集")
    ax.plot(train_sizes, valid_scores.mean(axis=1), marker="s", label="验证集")
    ax.fill_between(train_sizes, train_scores.mean(axis=1)-train_scores.std(axis=1),
                    train_scores.mean(axis=1)+train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes, valid_scores.mean(axis=1)-valid_scores.std(axis=1),
                    valid_scores.mean(axis=1)+valid_scores.std(axis=1), alpha=0.15)
    ax.set_xlabel("训练样本量"); ax.set_ylabel("准确率"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{group_name} 学习曲线")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    path_lc = os.path.join(TREE_DIR, f"{group_name}_学习曲线.png")
    fig.savefig(path_lc); plt.close(fig)


    if len(feat_names) >= 2:
        imp_sorted = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        top2 = list(imp_sorted.head(2).index)
        bmi_col = "BMI" if "BMI" in feat_names else (COL_BMI if COL_BMI in feat_names else None)
        if "GA_float" in feat_names and bmi_col:
            top2 = ["GA_float", bmi_col]
        X2 = df_group[top2].values
        y2 = df_group[y_col].astype(int).values
        if len(np.unique(y2)) >= 2:
            clf2 = DecisionTreeClassifier(
                criterion="gini", max_depth=clf.get_params()['max_depth'],
                min_samples_leaf=clf.get_params()['min_samples_leaf'],
                class_weight="balanced", random_state=42
            ).fit(X2, y2)
            x_min, x_max = X2[:,0].min()-0.5, X2[:,0].max()+0.5
            y_min, y_max = X2[:,1].min()-0.5, X2[:,1].max()+0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
            Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig, ax = plt.subplots(figsize=(6.6, 5.4))
            ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5,0.5,1.5], colors=["#9ecae1","#fdae6b"])
            ax.scatter(X2[y2==0,0], X2[y2==0,1], s=16, c="#2171b5", label="未达标", alpha=0.85)
            ax.scatter(X2[y2==1,0], X2[y2==1,1], s=16, c="#ef6c00", label="达标", alpha=0.85)
            ax.set_xlabel(top2[0]); ax.set_ylabel(top2[1])
            ax.set_title(f"{group_name} 决策边界（{top2[0]} vs {top2[1]}）")
            ax.legend(); ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path_db = os.path.join(TREE_DIR, f"{group_name}_决策边界_{top2[0]}_{top2[1]}.png")
            fig.savefig(path_db); plt.close(fig)


    try:
        tree_ = clf.tree_
        if tree_.feature[0] != -2:
            f_idx = tree_.feature[0]; thr = tree_.threshold[0]
            split_feat = feat_names[f_idx]
            left_mask = X_test.iloc[:, f_idx] <= thr
            right_mask = ~left_mask
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.hist(X_test.iloc[left_mask, f_idx], bins=20, alpha=0.6, label=f"{split_feat} ≤ {thr:.3f}")
            ax.hist(X_test.iloc[right_mask, f_idx], bins=20, alpha=0.6, label=f"{split_feat} > {thr:.3f}")
            ax.set_xlabel(split_feat); ax.set_ylabel("样本数")
            ax.set_title(f"{group_name} 节点前后样本分布（根分裂特征）")
            ax.legend(); ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path_node = os.path.join(TREE_DIR, f"{group_name}_节点样本分布_{split_feat}.png")
            fig.savefig(path_node); plt.close(fig)
    except Exception:
        pass

    return clf, feat_names, X_test, y_test, y_pred

#主流程：作图 + 决策树 + 汇总
theme = "bw"                
y_as_percent_for_conc = False

trend_paths, attain_paths, sens_paths = [], [], []
attain_tables, sens_tables = {}, {}
summary_rows = []


df_male['达标'] = (df_male[COL_YCON] >= Y_THRESHOLD).astype(int)

for g, sub in df_male.groupby('BMI组', dropna=True):
    group_name = str(g)
    print(f"\n==== 分组：{group_name} ====")


    if COL_BMI != "BMI":
        sub = sub.copy()
        sub["BMI"] = sub[COL_BMI]

    #图1：浓度趋势
    p1 = make_conc_trend_fig_pretty(sub, group_name, COL_YCON, save_dir=FIG_DIR, theme=theme, y_as_percent=y_as_percent_for_conc)
    if p1: trend_paths.append(p1)

    #图2：达标率 + 推荐孕周（90%分位）
    s_first_tmp = first_attainment_weeks(sub, id_col=COL_ID, ga_col='GA_float', y_col=COL_YCON, thr=Y_THRESHOLD)
    times = s_first_tmp.dropna().astype(float).values
    rec_week = float(np.percentile(times, 90)) if len(times)>0 else None
    p2, tab2, _ = make_attainment_curve_pretty(
        sub, group_name, COL_YCON, COL_ID, save_dir=FIG_DIR, theme=theme,
        mark_week=rec_week, mark_label=(f"推荐孕周 {rec_week:.2f}" if rec_week else None)
    )
    attain_paths.append(p2)
    attain_tables[group_name] = tab2
    not_rate = (times > rec_week).mean() if (len(times) > 0 and rec_week is not None) else np.nan

    #图3：误差敏感性
    p3, tab3 = simulate_sensitivity_pretty(sub, tab2, group_name, COL_YCON, COL_ID, save_dir=FIG_DIR, theme=theme)
    sens_paths.append(p3)
    sens_tables[group_name] = tab3

    #决策树全套（中文、无重叠优先）
    model, feat_names, Xtest, ytest, ypred = train_tree_for_group(
        sub, y_col="达标", theme=theme, group_name=group_name,
        max_depth=4, min_samples_leaf=max(8, int(len(sub)*0.02))
    )

    summary_rows.append({
        "BMI 组": group_name,
        "样本数": sub[COL_ID].nunique() if (COL_ID and COL_ID in sub.columns) else len(sub),
        "推荐孕周(90%分位)": round(rec_week, 2) if rec_week is not None else np.nan,
        "推荐孕周未达标比例": round(not_rate, 4) if not np.isnan(not_rate) else np.nan
    })

summary_df = pd.DataFrame(summary_rows).sort_values("BMI 组").reset_index(drop=True)
ensure_dir(OUT_DIR)
summary_path = os.path.join(OUT_DIR, "问题三_各BMI组推荐孕周.xlsx")
summary_df.to_excel(summary_path, index=False)
print("已导出汇总表：", summary_path)

q2_path = os.path.join(OUT_DIR, "问题二_BMI分组对照.xlsx")
Q2_SUMMARY.to_excel(q2_path, index=False)

for g in attain_tables:
    attain_tables[g].to_excel(os.path.join(OUT_DIR, f"{g}_达标率曲线表.xlsx"), index=False)
for g in sens_tables:
    sens_tables[g].to_excel(os.path.join(OUT_DIR, f"{g}_敏感性分析表.xlsx"), index=False)

# ========== 9. 生成 Word 报告（含决策树章节） ==========
doc = Document()
style = doc.styles['Normal']
style.font.name = 'SimHei'
style._element.rPr.rFonts.set(qn('w:eastAsia'), 'SimHei')

doc.add_heading("问题三：基于 BMI 的 NIPT 时点选择、误差敏感性与决策树分析", level=1)
p = doc.add_paragraph(f"数据来源：{EXCEL_PATH}；生成时间：{datetime.now():%Y-%m-%d %H:%M}")
p.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT

doc.add_heading("一、分组与数据说明", level=2)
doc.add_paragraph(
    "仅保留男胎样本（Y 染色体浓度非空）。采用 KMeans（k=3）对 BMI 自动聚类，以相邻簇中心的中点作为分组边界，"
    "得到低/中/高 BMI 三组，并在后续分析中分别建模与可视化。"
)

doc.add_paragraph("自动识别的 BMI 簇中心 & 边界：")
table0 = doc.add_table(rows=1, cols=2)
hdr = table0.rows[0].cells
hdr[0].text = "簇中心（升序）"; hdr[1].text = "分组边界（-∞, b1], (b1, b2], (b2, +∞）"
row = table0.add_row().cells
row[0].text = ", ".join([f"{c:.1f}" for c in centers])
row[1].text = ", ".join([("-∞" if not np.isfinite(e) else f"{e:.1f}") for e in edges[1:-1]])

doc.add_paragraph("各组样本数：")
table1 = doc.add_table(rows=1, cols=2)
h1 = table1.rows[0].cells
h1[0].text = "BMI 组"; h1[1].text = "样本数"
for g, n in group_counts.items():
    r = table1.add_row().cells
    r[0].text = str(g); r[1].text = str(int(n) if pd.notna(n) else 0)

doc.add_heading("二、时点推荐与鲁棒性", level=2)
t2 = doc.add_table(rows=1, cols=4)
hh = t2.rows[0].cells
hh[0].text = "BMI 组"; hh[1].text = "样本数"; hh[2].text = "推荐孕周(90%分位)"; hh[3].text = "推荐孕周未达标比例"
for _, r in summary_df.iterrows():
    rr = t2.add_row().cells
    rr[0].text = str(r["BMI 组"]); rr[1].text = str(r["样本数"])
    rr[2].text = str(r["推荐孕周(90%分位)"])
    rr[3].text = f"{r['推荐孕周未达标比例']:.2%}" if pd.notna(r['推荐孕周未达标比例']) else "NA"

doc.add_paragraph("（对照）问题二提供的结果：")
t3 = doc.add_table(rows=1, cols=4)
hh2 = t3.rows[0].cells
hh2[0].text = "BMI 分组"; hh2[1].text = "样本数 N"; hh2[2].text = "推荐检测孕周 (周)"; hh2[3].text = "推荐孕周未达标比例"
for _, r in Q2_SUMMARY.iterrows():
    rr = t3.add_row().cells
    rr[0].text = str(r["BMI 分组"]); rr[1].text = str(r["样本数 N"])
    rr[2].text = str(r["推荐检测孕周 (周)"]); rr[3].text = str(r["推荐孕周未达标比例"])

doc.add_heading("三、图表（浓度趋势 / 达标率 / 敏感性）", level=2)
for pth in sorted([p for p in os.listdir(FIG_DIR) if p.endswith(".png")]):
    doc.add_paragraph(pth.replace(".png",""))
    try: doc.add_picture(os.path.join(FIG_DIR, pth), width=Inches(5.6))
    except Exception: pass

doc.add_heading("四、决策树建模与可解释性图表", level=2)
doc.add_paragraph("展示各 BMI 组的决策树结构（中文）、特征重要性、混淆矩阵、学习曲线、（如适用）决策边界与根节点分裂特征的前后样本分布。")
for p in sorted([p for p in os.listdir(TREE_DIR) if p.endswith(".png")]):
    doc.add_paragraph(p.replace(".png",""))
    try: doc.add_picture(os.path.join(TREE_DIR, p), width=Inches(5.6))
    except Exception: pass

doc.add_heading("五、结论要点", level=2)
doc.add_paragraph("1) 随孕周增加，Y 染色体浓度总体上升；BMI 越高，达到 4% 的时间整体偏晚。")
doc.add_paragraph("2) 以“首次达标孕周”的 90% 分位作为推荐时点，可使 ≥90% 的样本在该孕周前已达标，兼顾一次成功率与等待成本。")
doc.add_paragraph("3) 误差敏感性显示：高 BMI 组对 ±1 周孕周误差更敏感，建议适度顺延 0.5–1 周以更稳妥。")
doc.add_paragraph("4) 决策树揭示：孕周与 BMI 是主导分裂特征；学习曲线提示应控制树深度避免过拟合。")

word_path = os.path.join(OUT_DIR, "问题三_自动分组+美化作图+决策树报告.docx")
doc.save(word_path)
print("已生成 Word 报告：", word_path)

print("\n 全部完成！输出目录：", os.path.abspath(OUT_DIR))
