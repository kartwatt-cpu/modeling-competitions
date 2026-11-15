


import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_gestweek(j_value):
    if pd.isna(j_value):
        return np.nan
    try:
        return float(j_value)
    except Exception:
        pass
    s = str(j_value).strip()
    m = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', s)
    if m:
        w, d = float(m.group(1)), float(m.group(2))
        return w + d / 7.0
    m = re.match(r'^\s*(\d+)\s*[周Ww]\s*(\d+)\s*[天Dd]\s*$', s)
    if m:
        w, d = float(m.group(1)), float(m.group(2))
        return w + d / 7.0
    m = re.match(r'^\s*(\d+)\s*([周Ww])\s*$', s)
    if m:
        return float(m.group(1))
    m = re.match(r'^\s*(\d+)\s*[天Dd]\s*$', s)
    if m:
        return float(m.group(1)) / 7.0
    return np.nan

def logistic_curve(x, A, k, x0):
    return A / (1.0 + np.exp(-k * (x - x0)))

def fit_logistic(x, y):
    A0 = max(np.percentile(y, 80), (np.max(y) if len(y) > 0 else 0.1))
    if not np.isfinite(A0) or A0 <= 0: A0 = 0.1
    k0 = 0.8
    x0_0 = np.median(x) if len(x) else 12.0
    try:
        popt, pcov = curve_fit(
            logistic_curve, x, y,
            p0=[A0, k0, x0_0],
            bounds=([0.04, 0.01, 6.0], [1.0, 5.0, 40.0]),
            maxfev=20000
        )
        yhat = logistic_curve(x, *popt)
        resid = y - yhat
        resid_sd = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else 0.02
        return popt, pcov, yhat, resid_sd
    except Exception:
        return None

def solve_week_for_threshold(A, k, x0, threshold=0.04):
    if A <= threshold or k <= 0: return np.nan
    try:
        val = A / threshold - 1.0
        if val <= 0: return np.nan
        return x0 + (1.0 / k) * np.log(val)
    except Exception:
        return np.nan

def estimate_pass_prob(mean, sd, threshold=0.04):
    if sd <= 1e-8: return float(mean >= threshold)
    z = (threshold - mean) / sd
    return float(1.0 - norm.cdf(z))

def objective_J(t, t_threshold, mean_func, resid_sd, w1=1.0, w2=10.0, w3=1.0):
    if t < 12.0: return np.inf
    mean = float(mean_func(t))
    pass_prob = estimate_pass_prob(mean, resid_sd, threshold=0.04)
    R_early = 1.0 - pass_prob
    R_late = 0.0 if t <= 20.0 else 10.0 * (t - 20.0)
    return w1 * (t - t_threshold)**2 + w2 * R_early + w3 * R_late

def scan_optimal_week(t_threshold, mean_func, resid_sd, t_min=12.0, t_max=20.0, step=0.1):
    ts = np.arange(t_min, t_max + 1e-9, step)
    Js = [objective_J(t, t_threshold, mean_func, resid_sd) for t in ts]
    idx = int(np.nanargmin(Js))
    return float(ts[idx]), float(Js[idx])

def safe_make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def guess_col(df, name_candidates):

    candidates = []
    for c in df.columns:
        if isinstance(c, tuple):
            cs = " ".join([str(x) for x in c])
        else:
            cs = str(c)
        cs_norm = cs.strip().lower()
        for cand in name_candidates:
            if cand in cs_norm:
                candidates.append(c)
                break
    return candidates[0] if candidates else None


#主流程


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default="附件.xlsx")
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    excel_path, sheet_name, outdir = args.excel, args.sheet, args.outdir
    safe_make_dir(outdir)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到 Excel 文件：{excel_path}")


    _tmp = pd.read_excel(excel_path, sheet_name=(sheet_name if sheet_name is not None else 0))
    if isinstance(_tmp, dict):
        first_key = list(_tmp.keys())[0]
        print(f"[提示] 检测到多工作表，默认使用：{first_key}")
        df = _tmp[first_key]
    else:
        df = _tmp

  
    print("[信息] 读取数据成功。列名示例（前10个）：", list(df.columns)[:10])

   
    col_V = guess_col(df, ["y染色体浓度", "y染色体游离", "y 浓度", "浓度y", "y-浓度", "v", " y " , " y% ", " y%"])
    if col_V is None:
        col_V = guess_col(df, [" y", " y值", "y " , "y", "y_chrom", "chr y"])

    col_J = guess_col(df, ["孕周", "周数", "j", "gest", "gestation", "孕周（周数+天数）", "孕周(周数+天数)"])
    col_K = guess_col(df, ["bmi", "体质指数", "体重指数", "k"])

    if col_V is None or col_J is None or col_K is None:
        print("[错误] 列自动识别失败：需要包含 'Y染色体浓度/孕周/BMI' 相关列。")
        print("       请核对 Excel 列名，并考虑用 --sheet 指定正确工作表。")
        print("       当前所有列名：", list(df.columns))
        return

   
    n0 = df.shape[0]
    df = df.loc[~pd.isna(df[col_V])].copy()
    n1 = df.shape[0]
    print(f"[信息] 原始行数={n0}；保留男胎(Y存在)后={n1}")


    df["gest_week"] = df[col_J].apply(parse_gestweek)
    df["BMI"] = pd.to_numeric(df[col_K], errors="coerce")
    df["Y_frac"] = pd.to_numeric(df[col_V], errors="coerce")


    if df["Y_frac"].dropna().gt(1.0).mean() > 0.2:
        df.loc[df["Y_frac"].notna(), "Y_frac"] = df.loc[df["Y_frac"].notna(), "Y_frac"] / 100.0
        print("[提示] 检测到 Y 浓度像百分数，已自动 /100 转为小数。")


    before = df.shape[0]
    df = df.loc[df["gest_week"].between(6, 40, inclusive="both") & df["BMI"].between(15, 60, inclusive="both")].copy()
    df = df.loc[(df["Y_frac"] >= 0) & (df["Y_frac"] <= 1.0)].copy()
    after = df.shape[0]
    print(f"[信息] 基础清洗后样本数：{after}（清洗前 {before}）")

    if after < 10:
        print("[警告] 有效样本数 < 10，KMeans 可能失败。将直接使用分位数分箱（最多3组）。")

    #BMI 分组
    X = df[["BMI"]].dropna().to_numpy()
    n_samples = X.shape[0]
    if n_samples == 0:
        print("[错误] BMI 全为空或被过滤；请检查 BMI 列或放宽过滤范围。")
        return

    labels = None
    used_method = None

    if n_samples >= 2:
        # 允许的 k 值：2..min(5, n_samples)
        ks = list(range(2, min(5, n_samples) + 1))
        best_score, best_k, best_labels = -np.inf, None, None
        for k in ks:
            try:
                km = KMeans(n_clusters=k, n_init=20, random_state=42)
                lab = km.fit_predict(X)
                # silhouette 需要至少2个簇且每簇至少1个样本；若失败则跳过
                score = silhouette_score(X, lab) if len(set(lab)) > 1 else -np.inf
                if score > best_score:
                    best_score, best_k, best_labels = score, k, lab
            except Exception:
                continue
        if best_labels is not None:
            labels = best_labels
            used_method = f"KMeans(k={best_k}, silhouette={best_score:.3f})"


    if labels is None:
        q = min(3, max(2, n_samples))  # 最多3组，至少2组
        try:
            df["_tmp_rank"] = df["BMI"].rank(method="first")
            df["_bin"], bin_edges = pd.qcut(df["BMI"], q=q, retbins=True, labels=False, duplicates="drop")
            labels = df["_bin"].astype(int).to_numpy()
            used_method = f"qcut(q={q})"
            df.drop(columns=["_tmp_rank", "_bin"], inplace=True, errors="ignore")
        except Exception as e:
            print("[错误] KMeans 与 qcut 均失败：", repr(e))
            return


    df = df.loc[df["BMI"].notna()].copy()
    labels_series = pd.Series(labels, index=df["BMI"].dropna().index)
    df["cluster"] = labels_series
    df = df.loc[df["cluster"].notna()].copy()
    df["cluster"] = df["cluster"].astype(int)


    med_order = df.groupby("cluster")["BMI"].median().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(med_order)}
    df["cluster"] = df["cluster"].map(remap)

    print(f"[信息] 分组方法：{used_method}；各组样本量：", df.groupby("cluster").size().to_dict())

    #组内拟合与推荐
    safe_make_dir(outdir)
    summary_rows, group_info = [], {}

    for g, gdf in df.groupby("cluster"):
        gdf = gdf.sort_values("gest_week")
        x = gdf["gest_week"].to_numpy()
        y = gdf["Y_frac"].to_numpy()

        group_info[g] = {"bmi_min": float(gdf["BMI"].min()), "bmi_max": float(gdf["BMI"].max()), "n": int(gdf.shape[0])}

        fit_res = None
        if len(gdf) >= 4 and len(np.unique(x)) >= 3:
            fit_res = fit_logistic(x, y)

        if fit_res is None:

            resid_sd = float(np.nanstd(y, ddof=1)) if len(y) > 1 else 0.02
            def mean_func(t):

                mask = np.abs(x - t) <= 1.0
                if mask.any():
                    return float(np.nanmean(y[mask]))

                k = min(5, len(y))
                if k == 0: return 0.03
                idx = np.argsort(np.abs(x - t))[:k]
                return float(np.nanmean(y[idx]))

            t_grid = np.linspace(max(6.0, float(np.nanmin(x)) if len(x) else 6.0),
                                 min(30.0, float(np.nanmax(x)) if len(x) else 30.0), 240)
            mvals = np.array([mean_func(t) for t in t_grid])
            meets = np.where(mvals >= 0.04)[0]
            t_threshold = float(t_grid[meets[0]]) if len(meets) else np.nan

            if np.isfinite(t_threshold):
                t_rec, _ = scan_optimal_week(t_threshold, mean_func, resid_sd, 12.0, 20.0, 0.1)
            else:

                t_rec = 13.0
            A=k=x0=np.nan
            x_line = np.linspace(max(6.0, np.nanmin(x) if len(x) else 6.0), min(30.0, np.nanmax(x) if len(x) else 30.0), 200)
            y_line = np.array([mean_func(t) for t in x_line])
        else:
            (A, k, x0), pcov, yhat, resid_sd = fit_res
            t_threshold = solve_week_for_threshold(A, k, x0, threshold=0.04)
            def mean_func(t): return logistic_curve(np.array([t]), A, k, x0)[0]
            t_rec, _ = scan_optimal_week(t_threshold if np.isfinite(t_threshold) else 12.0,
                                         mean_func, resid_sd, 12.0, 20.0, 0.1)
            x_line = np.linspace(max(6.0, np.nanmin(x)), min(30.0, np.nanmax(x)), 300)
            y_line = logistic_curve(x_line, A, k, x0)

#画图
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x, y, alpha=0.7, label="observed")
        plt.plot(x_line, y_line, linewidth=2.0, label="fit/mean")
        upper = np.clip(y_line + (resid_sd if np.isfinite(resid_sd) else 0.02), 0, 1)
        lower = np.clip(y_line - (resid_sd if np.isfinite(resid_sd) else 0.02), 0, 1)
        plt.fill_between(x_line, lower, upper, alpha=0.2, label="±1 SD")
        plt.axhline(0.04, linestyle="--", linewidth=1.0)
        if np.isfinite(t_threshold): plt.axvline(t_threshold, linestyle="--", linewidth=1.0)
        if np.isfinite(t_rec): plt.axvline(t_rec, linestyle=":", linewidth=1.2)
        plt.xlabel("孕周（周）")
        plt.ylabel("Y 染色体浓度（比例）")
        plt.title(f"组 {g}: BMI [{group_info[g]['bmi_min']:.1f}, {group_info[g]['bmi_max']:.1f}]，n={group_info[g]['n']}")
        plt.legend()
        fig_path = os.path.join(outdir, f"group_{g}_curve.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()

        summary_rows.append({
            "group": g,
            "BMI_min": group_info[g]['bmi_min'],
            "BMI_max": group_info[g]['bmi_max'],
            "n": group_info[g]['n'],
            "A_hat": float(A) if np.isfinite(A) else np.nan,
            "k_hat": float(k) if np.isfinite(k) else np.nan,
            "x0_hat": float(x0) if np.isfinite(x0) else np.nan,
            "resid_sd": float(resid_sd) if np.isfinite(resid_sd) else np.nan,
            "t_threshold(FF>=4%)": float(t_threshold) if np.isfinite(t_threshold) else np.nan,
            "t_recommend(optimal)": float(t_rec) if np.isfinite(t_rec) else np.nan
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True)
    csv_path = os.path.join(outdir, "summary_q2.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n===== 问题二：BMI 分组与推荐 NIPT 时点（综合失败率与延迟风险） =====")
    print("分组与样本量：", summary_df[["group", "n"]].to_dict("records"))
    show_cols = ["group", "BMI_min", "BMI_max", "n", "t_threshold(FF>=4%)", "t_recommend(optimal)"]
    disp = summary_df[show_cols].copy()
    for col in ["t_threshold(FF>=4%)", "t_recommend(optimal)"]:
        disp[col] = disp[col].apply(lambda v: np.nan if pd.isna(v) else round(float(v), 1))
    print(disp.to_string(index=False))
    print(f"\n结果已保存到：{csv_path}")
    print(f"每组曲线图已输出到：{outdir}/group_*.png")
    print("\n说明：如 KMeans 因样本不足失败，脚本已自动回退到分位数分箱（qcut）。")
    print("      若仍为空，请检查：孕周/BMI/Y列名是否识别、是否被过滤范围排除。")

if __name__ == "__main__":
    main()
