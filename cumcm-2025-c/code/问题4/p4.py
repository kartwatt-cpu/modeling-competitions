import os, re, json, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.utils import resample


try:
    from imblearn.over_sampling import SMOTE
except:
    SMOTE = None


try:
    import optuna
except:
    optuna = None

matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False


EXCEL = "附件.xlsx"
OUTDIR = "p4_pro_outputs"
SEED = 2025
TEST_SIZE = 0.20
N_SPLITS = 5
N_TRIALS = 40  
USE_SMOTE = False  
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUTDIR, exist_ok=True)


def parse_gw(s):
    """解析 '13w+2' / '13周+2' / 13.5 --> float(周)"""
    if pd.isna(s): return np.nan
    s = str(s).strip()
    m = re.match(r'(\d+)\s*w(?:\s*\+\s*(\d+))?$', s)
    if m:
        w = int(m.group(1)); d = int(m.group(2) or 0); return w + d/7.0
    m2 = re.match(r'(\d+)\s*周(?:\s*\+\s*(\d+))?', s)
    if m2:
        w = int(m2.group(1)); d = int(m2.group(2) or 0); return w + d/7.0
    try:
        return float(s)
    except:
        return np.nan

def ensure_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少关键列：{miss}\n请核对 Excel 表头是否与题面一致，或检查是否存在隐藏空格/全半角。")

def female_mask_auto(df):

    y_z = "Y染色体的Z值"
    y_c = "Y染色体浓度"
    if df[y_z].isna().mean() > 0.7 and df[y_c].isna().mean() > 0.7:
        return df[y_z].isna() & df[y_c].isna()
    else:
        return (df[y_c].fillna(0) < 0.05) & (df[y_z].fillna(0) < 0.5)

def save_hist(series, title, path):
    plt.figure(figsize=(5.2,3.8))
    plt.hist(series.dropna().values, bins=30)
    plt.title(title); plt.xlabel(title); plt.ylabel("频数")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def corr_heatmap(df_num, title, path):
    corr = df_num.corr().values
    labels = df_num.columns.tolist()
    plt.figure(figsize=(9,7))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=320); plt.close()

def box_by_label(df, cols, path, title):
    plt.figure(figsize=(7,4.2))
    positions = np.arange(len(cols))*2.0
    width = 0.7
    data0 = [df.loc[df["label"]==0, c].dropna().values for c in cols]
    data1 = [df.loc[df["label"]==1, c].dropna().values for c in cols]
    bp0 = plt.boxplot(data0, positions=positions-width/2, widths=width, patch_artist=True)
    bp1 = plt.boxplot(data1, positions=positions+width/2, widths=width, patch_artist=True)
    for patch in bp0["boxes"]: patch.set_alpha(0.5)
    for patch in bp1["boxes"]: patch.set_alpha(0.5)
    plt.xticks(positions, cols)
    plt.legend([bp0["boxes"][0], bp1["boxes"][0]], ["正常(0)", "异常(1)"])
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=320); plt.close()

def bootstrap_ci(y, p, metric_fn, B=1000, alpha=0.05, random_state=SEED):
    rng = np.random.RandomState(random_state)
    n = len(y)
    stats = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        stats.append(metric_fn(y[idx], p[idx]))
    stats = np.array(stats)
    lo = np.percentile(stats, 100*alpha/2)
    hi = np.percentile(stats, 100*(1-alpha/2))
    return float(np.mean(stats)), float(lo), float(hi)


df = pd.read_excel(EXCEL)
df.columns = df.columns.str.strip() 

need = ["年龄","身高","体重","检测孕周","孕妇BMI","原始读段数","在参考基因组上比对的比例","重复读段的比例",
        "唯一比对的读段数（个）","唯一比对的读段数","GC含量",
        "13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值","Y染色体的Z值",
        "Y染色体浓度","X染色体浓度","13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量",
        "被过滤掉读段数的比例","染色体的非整倍体"]

if "唯一比对的读段数（个）" in df.columns:
    uniq_col = "唯一比对的读段数（个）"
elif "唯一比对的读段数" in df.columns:
    uniq_col = "唯一比对的读段数"
else:
    uniq_col = None
if uniq_col is None:
    ensure_cols(df, ["唯一比对的读段数（个）"])  # 触发报错

check_cols = [c for c in need if c!="唯一比对的读段数" and c!="唯一比对的读段数（个）"]
ensure_cols(df, [c for c in check_cols if c in df.columns])


if uniq_col != "唯一比对的读段数":
    df = df.rename(columns={uniq_col: "唯一比对的读段数"})


df["孕周数值"] = df["检测孕周"].apply(parse_gw)


fm = female_mask_auto(df)
d = df.loc[fm].copy()
d["label"] = d["染色体的非整倍体"].notna().astype(int)

print(f"[INFO] 女胎近似样本量：{d.shape[0]}  阳性(异常)={int(d['label'].sum())}  阴性(正常)={int((1-d['label']).sum())}")


FEATURES = [
    "年龄","身高","体重","孕妇BMI","孕周数值",
    "原始读段数","在参考基因组上比对的比例","重复读段的比例","唯一比对的读段数","GC含量",
    "13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值","X染色体浓度",
    "13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量","被过滤掉读段数的比例"
]
d = d.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)


for col in ["年龄","孕妇BMI","孕周数值","X染色体浓度","GC含量","被过滤掉读段数的比例"]:
    if col in d.columns:
        save_hist(d[col], f"{col} 分布", os.path.join(OUTDIR, f"hist_{col}.png"))
corr_cols = ["年龄","孕妇BMI","孕周数值","X染色体浓度","GC含量",
             "13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值"]
corr_heatmap(d[corr_cols], "特征相关性热力图", os.path.join(OUTDIR, "corr_heatmap.png"))
box_by_label(d, ["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值"],
             os.path.join(OUTDIR, "boxplot_Z_by_label.png"), "13/18/21号Z值在正常/异常组的分布")


X_all = d[FEATURES].values.astype(np.float32)
y_all = d["label"].values.astype(np.int64)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all)


scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim, h1=64, h2=32, dp1=0.15, dp2=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dp1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dp2),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_one(model, X, y, Xv, yv, pos_weight, lr=1e-3, wd=1e-4, max_epoch=120, patience=20, batch=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)).unsqueeze(1))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, verbose=False)
    best_auc, best_state, wait = -1, None, 0
    train_loss_hist, val_auc_hist = [], []
    for ep in range(1, max_epoch+1):
        model.train()
        lossv = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward(); opt.step()
            lossv += loss.item()*len(xb)
        train_loss = lossv/len(ds)
        #验证 AUC
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.from_numpy(Xv).to(device))).cpu().numpy().ravel()
        try:
            auc = roc_auc_score(yv, pv)
        except:
            auc = np.nan
        train_loss_hist.append(train_loss); val_auc_hist.append(auc)
        sched.step(auc if not np.isnan(auc) else 0.0)
        if auc>best_auc:
            best_auc, best_state, wait = auc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait>=patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loss_hist, val_auc_hist, best_auc

def choose_threshold(y_true, p, mode="f1"):

    ths = np.linspace(0.05, 0.95, 181)
    best, best_t = -1, 0.5
    for t in ths:
        pred = (p>=t).astype(int)
        if mode=="f1":
            s = f1_score(y_true, pred, zero_division=0)
        else:
            cm = confusion_matrix(y_true, pred)
            tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
            tpr = tp/(tp+fn+1e-9); tnr = tn/(tn+fp+1e-9)
            s = tpr + tnr - 1.0  # Youden's J
        if s>best: best, best_t = s, t
    return best_t, best


def crossval_score(params, X, y, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        if USE_SMOTE and SMOTE is not None:
            sm = SMOTE(random_state=SEED)
            Xtr, ytr = sm.fit_resample(Xtr, ytr)

        pos = (ytr==1).sum(); neg = (ytr==0).sum()
        pw = neg/max(1,pos)
        model = MLP(in_dim=X.shape[1],
                    h1=int(params["h1"]), h2=int(params["h2"]),
                    dp1=params["dp1"], dp2=params["dp2"])
        model, _, _, best_auc = train_one(model, Xtr, ytr, Xva, yva, pw,
                                          lr=params["lr"], wd=params["wd"],
                                          max_epoch=int(params["epochs"]),
                                          patience=int(params["patience"]),
                                          batch=int(params["batch"]))
        aucs.append(best_auc if not np.isnan(best_auc) else 0.5)
    return float(np.mean(aucs))

def suggest_params(trial):
    return {
        "h1": trial.suggest_categorical("h1", [32, 64, 96, 128]),
        "h2": trial.suggest_categorical("h2", [16, 32, 48, 64]),
        "dp1": trial.suggest_float("dp1", 0.05, 0.35),
        "dp2": trial.suggest_float("dp2", 0.05, 0.30),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 3e-3, log=True),
        "epochs": trial.suggest_int("epochs", 80, 200, step=20),
        "patience": trial.suggest_int("patience", 8, 30),
        "batch": trial.suggest_categorical("batch", [32, 64, 128]),
    }

if optuna is not None:
    def objective(trial):
        params = suggest_params(trial)
        score = crossval_score(params, X_tr, y_tr, n_splits=3)
        trial.report(score, step=0)
        return score
    study = optuna.create_study(direction="maximize", study_name="nipt_p4_pro")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    print("[OPTUNA] best_params:", best_params, " best_score:", study.best_value)
else:
    best_params = {"h1":64,"h2":32,"dp1":0.15,"dp2":0.10,"lr":1e-3,"wd":1e-4,"epochs":120,"patience":20,"batch":64}
    print("[WARN] 未安装 optuna，使用默认超参：", best_params)

#用最佳超参做 KFold 训练，并集成到测试集
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_probs_val, fold_thr, fold_models = [], [], []
val_histories = []
for k, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr), 1):
    Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
    ytr, yva = y_tr[tr_idx], y_tr[va_idx]
    if USE_SMOTE and SMOTE is not None:
        sm = SMOTE(random_state=SEED)
        Xtr, ytr = sm.fit_resample(Xtr, ytr)
    pos = (ytr==1).sum(); neg = (ytr==0).sum()
    pw = neg/max(1,pos)

    model = MLP(in_dim=X_tr.shape[1],
                h1=int(best_params["h1"]), h2=int(best_params["h2"]),
                dp1=float(best_params["dp1"]), dp2=float(best_params["dp2"]))
    model, tr_loss_hist, va_auc_hist, best_auc = train_one(
        model, Xtr, ytr, Xva, yva, pw,
        lr=float(best_params["lr"]), wd=float(best_params["wd"]),
        max_epoch=int(best_params["epochs"]), patience=int(best_params["patience"]),
        batch=int(best_params["batch"])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        pva = torch.sigmoid(model(torch.from_numpy(Xva).to(device))).cpu().numpy().ravel()
    thr_f1, _ = choose_threshold(yva, pva, mode="f1")
    fold_probs_val.append((yva, pva)); fold_thr.append(thr_f1); fold_models.append(model)
    val_histories.append((tr_loss_hist, va_auc_hist))

    plt.figure(figsize=(6,3.4))
    plt.plot(tr_loss_hist, label="训练损失")
    plt.plot(va_auc_hist, label="验证AUC")
    plt.title(f"Fold {k} 训练过程")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"fold{k}_train_curve.png"), dpi=300); plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
probs_test = []
for model in fold_models:
    model.eval()
    with torch.no_grad():
        pt = torch.sigmoid(model(torch.from_numpy(X_te).to(device))).cpu().numpy().ravel()
    probs_test.append(pt)
p_test = np.mean(np.vstack(probs_test), axis=0)

#选择最终阈值：聚合全部验证折预测，最大化F1
y_val_all = np.concatenate([y for y,_ in fold_probs_val])
p_val_all = np.concatenate([p for _,p in fold_probs_val])
final_thr, _ = choose_threshold(y_val_all, p_val_all, mode="f1")

y_pred_test = (p_test>=final_thr).astype(int)


acc = accuracy_score(y_te, y_pred_test)
pre = precision_score(y_te, y_pred_test, zero_division=0)
rec = recall_score(y_te, y_pred_test, zero_division=0)
f1  = f1_score(y_te, y_pred_test, zero_division=0)
auc = roc_auc_score(y_te, p_test)

print(f"[TEST-ENSEMBLE] AUC={auc:.4f}  Acc={acc:.4f}  Pre={pre:.4f}  Rec={rec:.4f}  F1={f1:.4f}  thr={final_thr:.2f}")

#混淆矩阵
cm = confusion_matrix(y_te, y_pred_test)
plt.figure(figsize=(4.6,4))
plt.imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xticks([0,1], ["预测0(正常)","预测1(异常)"])
plt.yticks([0,1], ["真实0(正常)","真实1(异常)"])
plt.title("测试集 混淆矩阵")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "test_confusion_matrix.png"), dpi=320); plt.close()

#ROC / PR
fpr, tpr, _ = roc_curve(y_te, p_test)
precision, recall, _ = precision_recall_curve(y_te, p_test)
plt.figure(figsize=(5.3,4))
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"测试集 ROC (AUC={auc:.3f})")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "test_roc.png"), dpi=320); plt.close()

plt.figure(figsize=(5.3,4))
plt.plot(recall, precision)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("测试集 PR 曲线")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "test_pr.png"), dpi=320); plt.close()

#置信区间（Bootstrap 95% CI）
mean_auc, lo_auc, hi_auc = bootstrap_ci(y_te, p_test, roc_auc_score, B=1000)
def acc_metric(y, p): return accuracy_score(y, (p>=final_thr).astype(int))
mean_acc, lo_acc, hi_acc = bootstrap_ci(y_te, p_test, acc_metric, B=1000)
print(f"[CI] AUC mean={mean_auc:.3f} 95%CI[{lo_auc:.3f},{hi_auc:.3f}]")
print(f"[CI] Acc mean={mean_acc:.3f} 95%CI[{lo_acc:.3f},{hi_acc:.3f}]")

#可解释性：Permutation Importance
def permutation_importance(model_list, X, y, metric_fn=roc_auc_score, repeats=20, seed=SEED):
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        probs = []
        for m in model_list:
            p = torch.sigmoid(m(torch.from_numpy(X))).numpy().ravel()
            probs.append(p)
    base = metric_fn(y, np.mean(np.vstack(probs), axis=0))
    imps = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            with torch.no_grad():
                pp = []
                for m in model_list:
                    p = torch.sigmoid(m(torch.from_numpy(Xp))).numpy().ravel()
                    pp.append(p)
            s = metric_fn(y, np.mean(np.vstack(pp), axis=0))
            drops.append(base - s)
        imps.append(np.mean(drops))
    return np.array(imps), base

imp, base = permutation_importance([m.to("cpu").eval() for m in fold_models], X_te, y_te, repeats=20)
order = np.argsort(-imp)
plt.figure(figsize=(7.2,5))
plt.barh(np.array(FEATURES)[order], imp[order])
plt.gca().invert_yaxis()
plt.xlabel("AUC 降幅（Permutation Importance）")
plt.title("特征重要性（置换）- 测试集")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "feature_importance_permutation_test.png"), dpi=320); plt.close()

#可解释性：SHAP（若可用则做，自动Fallback）
def shap_explain(models, X_bg, X_ref, features, out_png):
    try:
        import shap
    except Exception as e:
        print(f"[WARN] shap 不可用：{e}"); return False

    def f_predict(Xnp):
        Xtorch = torch.from_numpy(Xnp.astype(np.float32))
        with torch.no_grad():
            probs = []
            for m in models:
                p = torch.sigmoid(m(Xtorch)).numpy().ravel()
                probs.append(p)
        return np.mean(np.vstack(probs), axis=0)
    try:
        explainer = shap.KernelExplainer(f_predict, X_bg)
        sv = explainer.shap_values(X_ref, nsamples=200)
        sv = np.array(sv)
        if sv.ndim==3: sv = sv[0]
        mean_abs = np.mean(np.abs(sv), axis=0)
        order = np.argsort(-mean_abs)
        plt.figure(figsize=(7.2,5))
        plt.barh(np.array(features)[order], mean_abs[order])
        plt.gca().invert_yaxis()
        plt.xlabel("平均|SHAP|")
        plt.title("SHAP 特征重要性（KernelExplainer, 集成）")
        plt.tight_layout(); plt.savefig(out_png, dpi=320); plt.close()
        print("[INFO] SHAP 已导出。")
        return True
    except Exception as e:
        print(f"[WARN] SHAP 计算失败：{e}")
        return False


bg_idx = np.random.choice(len(X_tr), min(120, len(X_tr)), replace=False)
ref_idx = np.random.choice(len(X_te), min(200, len(X_te)), replace=False)
_ = shap_explain([m.to("cpu").eval() for m in fold_models],
                 X_tr[bg_idx], X_te[ref_idx], FEATURES,
                 os.path.join(OUTDIR, "feature_importance_shap_integrated.png"))


res = {
    "n_samples_total": int(len(d)),
    "n_test": int(len(y_te)),
    "positive_rate_test": float(y_te.mean()),
    "AUC_test": float(auc),
    "ACC_test": float(acc),
    "Precision_test": float(pre),
    "Recall_test": float(rec),
    "F1_test": float(f1),
    "thr_final": float(final_thr),
    "AUC_CI_95": [float(lo_auc), float(hi_auc)],
    "ACC_CI_95": [float(lo_acc), float(hi_acc)],
    "best_params": best_params
}
with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTDIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=2))

print(f"[DONE] 增强版输出完成，文件位于：./{OUTDIR}/")
