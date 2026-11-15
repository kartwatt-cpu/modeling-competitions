# p2_draft_fa_trade_stability.py
# ------------------------------------------------------------
# 子问题2升级版：5先发 + 5替补（位置一一对应）
# 统一考虑：选秀 Draft / 自由球员签约 FA / 交易 Trade
# 目标：竞技表现 + 商业价值 -（签约成本/交易费/变动惩罚/Replacement惩罚）
# ------------------------------------------------------------

import os, sys, re
import numpy as np
import pandas as pd
import unicodedata
from datetime import datetime

# ============================================================
# 0) 依赖：PuLP
# ============================================================
def ensure_pulp():
    try:
        import pulp  # noqa
        return
    except Exception:
        import subprocess
        print("[INFO] pulp 未安装，正在安装 pulp ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
        import pulp  # noqa

ensure_pulp()
import pulp

# ============================================================
# 1) 路径配置（确保同目录）
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALARY_FULL = os.path.join(BASE_DIR, "wnba_salaries_2025.csv")
PARAM_CSV   = os.path.join(BASE_DIR, "模型参数数据：model_parameters.csv")
MERGED_XLSX = os.path.join(BASE_DIR, "ICM_D_WNBA_ThreeFiles_Merged.xlsx")
TEAM = "Las Vegas Aces"

for fp in [SALARY_FULL, PARAM_CSV, MERGED_XLSX]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"找不到文件: {fp}")

print(f"[INFO] 使用薪资候选池：{os.path.basename(SALARY_FULL)}")

df_salary = pd.read_csv(SALARY_FULL)
params = pd.read_csv(PARAM_CSV)

# ============================================================
# 2) 名字规范化（解决 A’ja / A'ja / 拼接格式）
# ============================================================
def norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_salary_player(s: str) -> str:
    """
    适配你这份 csv：'Breanna Stewart B Stewart' -> 'Breanna Stewart'
    """
    s = str(s).strip()
    return re.sub(r"\s+[A-Z]\s+[A-Za-z'\-\.]+$", "", s).strip()

# ============================================================
# 3) 读取参数
# ============================================================
def get_param(symbol: str, default: float) -> float:
    if "symbol" not in params.columns:
        return float(default)
    row = params.loc[params["symbol"] == symbol]
    if len(row) == 0:
        return float(default)
    col = "suggested_value" if "suggested_value" in row.columns else row.columns[-1]
    return float(row[col].iloc[0])

S_cap = get_param("S_cap", 1507100.0)
pct_brand = get_param("pct_brand", 0.35)
w_pop  = float(pct_brand)
w_perf = 1.0 - w_pop
p_injury_base = get_param("p_injury", 0.20)

REQ_G, REQ_F, REQ_C = 2, 2, 1

# 稳定性（尽量少动）
MAX_NEW_TOTAL    = int(get_param("MAX_NEW_TOTAL", 6))     # 总引入（非本队）人数上限
MAX_NEW_STARTERS = int(get_param("MAX_NEW_STARTERS", 2))  # 新引入的先发人数上限
MIN_TEAM_STARTERS= int(get_param("MIN_TEAM_STARTERS", 3)) # 本队先发至少几人

# 成本与惩罚
SIGN_COST_RATE   = float(get_param("SIGN_COST_RATE", 0.06))    # FA签约成本比例
TRADE_FEE_RATE   = float(get_param("TRADE_FEE_RATE", 0.10))    # 交易费比例（可调）
CHANGE_PENALTY   = float(get_param("CHANGE_PENALTY", 0.06))    # 变动惩罚（非本队被选中）
EPS_SAL          = float(get_param("EPS_SAL", 0.02))           # 薪资轻惩罚

# 交易/选秀限制
ALLOW_TRADES = True
MAX_TRADES   = int(get_param("MAX_TRADES", 2))                 # 最多交易引入人数
ALLOW_DRAFT  = True
MAX_DRAFT    = int(get_param("MAX_DRAFT", 1))                  # 最多选秀引入人数（通常1-2）

# Replacement 兜底：尽量不用
ALLOW_REPLACEMENT_IF_INFEASIBLE = True
MAX_REPLACEMENTS = 1
MIN_SALARY_TRY = [65000, 60000, 55000]

# 位置权重：先发更重要
STARTER_PERF_W = 1.00
BENCH_PERF_W   = 0.70
STARTER_POP_W  = 1.00
BENCH_POP_W    = 0.85

# Replacement 强惩罚（保证尽量不出现）
REPL_HARD_PEN = 0.25   # 每选一个 replacement 扣掉的目标（量级较大）

# ============================================================
# 4) 列名标准化：适配 wnba_salaries_2025.csv
# ============================================================
colmap = {}
if "player" not in df_salary.columns:
    if "Player" in df_salary.columns: colmap["Player"] = "player"
    elif "name" in df_salary.columns: colmap["name"] = "player"
    elif "Name" in df_salary.columns: colmap["Name"] = "player"

if "salary_2025" not in df_salary.columns:
    if "2025 Salary" in df_salary.columns: colmap["2025 Salary"] = "salary_2025"
    elif "salary" in df_salary.columns: colmap["salary"] = "salary_2025"

df_salary = df_salary.rename(columns=colmap)
if "player" not in df_salary.columns:
    raise ValueError("薪资文件缺少 player 列（或 Player/name/Name）。")
if "salary_2025" not in df_salary.columns:
    raise ValueError("薪资文件缺少 salary_2025 列（或 2025 Salary）。")

df_salary["player"] = df_salary["player"].astype(str).str.strip()
df_salary["player_raw"] = df_salary["player"]
df_salary["player"] = df_salary["player"].apply(clean_salary_player)
df_salary["salary_2025"] = pd.to_numeric(df_salary["salary_2025"], errors="coerce").fillna(0.0)

print("[DEBUG] player 清洗前/后示例：")
print(df_salary[["player_raw","player"]].head(8).to_string(index=False))

# ============================================================
# 5) 从 merged.xlsx 补 team + position
# ============================================================
print("[INFO] 从 merged.xlsx 补齐球员位置与球队信息...")
xls = pd.ExcelFile(MERGED_XLSX)
sheet_name = None
for s in xls.sheet_names:
    if "MASTER_PLAYER_GAME" in s.upper():
        sheet_name = s
        break
if sheet_name is None:
    sheet_name = "30_MASTER_PLAYER_GAME"

game = pd.read_excel(MERGED_XLSX, sheet_name=sheet_name)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_player = pick_col(game, ["player","Player","PLAYER","athlete","name"])
col_pos    = pick_col(game, ["pos","Pos","POS","position","Position"])
col_team   = pick_col(game, ["team","Team","TEAM","franchise"])
col_active = pick_col(game, ["active","Active","ACTIVE"])

if col_player is None: raise ValueError("merged.xlsx 比赛表缺 player/name 列")
if col_pos is None:    raise ValueError("merged.xlsx 比赛表缺 pos/position 列")

g = game.copy()
if col_active is not None:
    g = g[g[col_active] == True].copy()

g[col_player] = g[col_player].astype(str).str.strip()
g[col_pos]    = g[col_pos].astype(str).str.strip()

pos_mode = g.groupby(col_player)[col_pos].agg(lambda x: x.value_counts().index[0]).reset_index()
pos_mode = pos_mode.rename(columns={col_player:"player", col_pos:"position"})

if col_team is not None:
    team_mode = g.groupby(col_player)[col_team].agg(lambda x: x.value_counts().index[0]).reset_index()
    team_mode = team_mode.rename(columns={col_player:"player", col_team:"team"})
else:
    team_mode = pd.DataFrame({"player": pos_mode["player"], "team": "Unknown"})

df_salary["player_key"] = df_salary["player"].apply(norm_name)
pos_mode["player_key"]  = pos_mode["player"].apply(norm_name)
team_mode["player_key"] = team_mode["player"].apply(norm_name)

df_salary = df_salary.merge(pos_mode[["player_key","position"]], on="player_key", how="left")
df_salary = df_salary.merge(team_mode[["player_key","team"]],     on="player_key", how="left")

hit = df_salary["position"].notna().mean()
print(f"[DEBUG] position 匹配成功率: {hit:.1%} | df_salary 行数: {len(df_salary)}")
if hit < 0.20:
    bad = df_salary[df_salary["position"].isna()][["player"]].head(15)
    print("[DEBUG] position 未匹配样例：")
    print(bad.to_string(index=False))

df_salary = df_salary[~df_salary["position"].isna()].copy()
if len(df_salary) == 0:
    raise RuntimeError("补 position 后 df_salary 为空：检查球员名匹配。")

def map_position(pos) -> str:
    s = str(pos).upper()
    if "C" in s: return "C"
    if "F" in s: return "F"
    if "G" in s: return "G"
    return "UNK"

df_salary["pos3"] = df_salary["position"].apply(map_position)
df_salary = df_salary[df_salary["pos3"]!="UNK"].reset_index(drop=True)

df_salary["team"] = df_salary["team"].fillna("Unknown")
df_salary["is_team"] = (df_salary["team"].astype(str).str.strip().str.lower() == TEAM.lower()).astype(int)

# ============================================================
# 6) FA / Trade 资格：优先用 is_fa 列；否则默认外部为 FA
# ============================================================
if "is_fa" in df_salary.columns:
    df_salary["is_fa"] = df_salary["is_fa"].astype(int)
else:
    # 默认：本队=0（不是FA），外部=1（当作可签FA）
    df_salary["is_fa"] = (1 - df_salary["is_team"]).astype(int)

# 如果你有明确的“不可签约球员”，你可以手动给 is_fa=0，让他们只能通过 Trade 引入

# ============================================================
# 7) 从 merged.xlsx 构建表现/带票代理（perf/pop）
# ============================================================
print("[INFO] 从 merged.xlsx 计算表现/带票代理...")

col_season = pick_col(g, ["season","Season","SEASON","year","Year"])
col_min    = pick_col(g, ["minutes","Minutes","MIN","min"])
col_pts    = pick_col(g, ["points","Points","PTS"])
col_reb    = pick_col(g, ["rebounds","Rebounds","REB"])
col_ast    = pick_col(g, ["assists","Assists","AST"])
col_stl    = pick_col(g, ["steals","Steals","STL"])
col_blk    = pick_col(g, ["blocks","Blocks","BLK"])
col_tov    = pick_col(g, ["turnovers","Turnovers","TOV"])
col_ticket = pick_col(g, ["ticket_revenue_proxy","Ticket_Revenue_Proxy","ticket_proxy","ticket_revenue"])

need = [col_season,col_min,col_pts,col_reb,col_ast,col_stl,col_blk,col_tov,col_ticket]
if any(c is None for c in need):
    raise ValueError("merged.xlsx 比赛表缺 perf/pop 构造所需列（MIN/PTS/REB/AST/.../ticket）。")

gg = g.copy()
gg[col_season] = pd.to_numeric(gg[col_season], errors="coerce")
gg = gg[gg[col_season] >= 2023].copy()

for c in [col_min,col_pts,col_reb,col_ast,col_stl,col_blk,col_tov,col_ticket]:
    gg[c] = pd.to_numeric(gg[c], errors="coerce").fillna(0.0)
gg = gg[gg[col_min] > 0].copy()

gg["perf_proxy"] = (
    gg[col_pts]
    + 1.2*gg[col_reb]
    + 1.5*gg[col_ast]
    + 2.0*gg[col_stl]
    + 2.0*gg[col_blk]
    - 2.0*gg[col_tov]
) / gg[col_min].clip(lower=1.0)

gg["pop_proxy"] = gg[col_ticket] / gg[col_min].clip(lower=1.0)

agg = gg.groupby(col_player).agg(
    perf_proxy=("perf_proxy","mean"),
    pop_proxy=("pop_proxy","mean"),
    mins=(col_min,"sum")
).reset_index().rename(columns={col_player:"player"})

df = df_salary.merge(agg, on="player", how="left")

# 兜底：没匹配到比赛数据的球员给小baseline，避免全0
df["perf_raw"] = df["perf_proxy"].fillna(0.10)
df["pop_raw"]  = df["pop_proxy"].fillna(0.10)

df["p_injury"] = p_injury_base
df["perf_eff"] = df["perf_raw"]*(1.0-df["p_injury"])

# 去重 + 薪资底线 + pop截尾（抑制套利）
MIN_SAL_FLOOR = 65000
df["salary_2025"] = pd.to_numeric(df["salary_2025"], errors="coerce").fillna(0.0)
df.loc[df["salary_2025"] < MIN_SAL_FLOOR, "salary_2025"] = MIN_SAL_FLOOR

df["player_key"] = df["player"].apply(norm_name)
df = df.sort_values("salary_2025", ascending=False).drop_duplicates("player_key", keep="first").copy()

# pop winsor + log1p
df["pop_raw"] = pd.to_numeric(df["pop_raw"], errors="coerce").fillna(0.0)
lo, hi = df["pop_raw"].quantile(0.01), df["pop_raw"].quantile(0.99)
df["pop_raw"] = np.log1p(df["pop_raw"].clip(lo, hi))

df = df.reset_index(drop=True)

def minmax_01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.full_like(x, 0.5, dtype=float)
    return (x - mn) / (mx - mn)

df["perf_norm"] = minmax_01(df["perf_eff"].values)
df["pop_norm"]  = minmax_01(df["pop_raw"].values)

# ============================================================
# 8) 生成 Draft Pool（虚拟新秀）
# ============================================================
def build_draft_pool():
    # 新秀工资（可用参数表替换）
    rookie_salary = float(get_param("ROOKIE_SALARY", 85000.0))

    # 新秀预期表现/商业（调小一点更现实；否则会被模型疯狂选）
    draft_perf = float(get_param("DRAFT_PERF_NORM", 0.55))
    draft_pop  = float(get_param("DRAFT_POP_NORM",  0.45))

    # 每个位置给1个候选（你也可加到2-3个）
    pool = []
    for pos in ["G","F","C"]:
        pool.append({
            "player": f"Draft_{pos}_Pick1",
            "team": "DRAFT",
            "position": pos,
            "pos3": pos,
            "salary_2025": rookie_salary,
            "is_team": 0,
            "is_fa": 0,            # 不是FA来源
            "source": "DRAFT",
            "p_injury": p_injury_base,
            "perf_eff": draft_perf,
            "pop_raw": draft_pop,
            "perf_norm": draft_perf,
            "pop_norm": draft_pop
        })
    return pd.DataFrame(pool)

# ============================================================
# 9) Replacement 兜底（尽量少）
# ============================================================
def build_replacement_pool(min_salary: float):
    pool=[]
    for pos,k in [("G",6),("F",6),("C",3)]:
        for j in range(k):
            pool.append({
                "player": f"Replacement_{pos}_{j+1}",
                "team": "REPL",
                "position": pos,
                "pos3": pos,
                "salary_2025": float(min_salary),
                "is_team": 0,
                "is_fa": 1,
                "source": "REPL",
                "p_injury": p_injury_base,
                "perf_eff": 0.05*(1-p_injury_base),
                "pop_raw": 0.05,
                "perf_norm": 0.05,
                "pop_norm": 0.05
            })
    return pd.DataFrame(pool)

# ============================================================
# 10) 构建候选集合：TEAM + FA + (TRADE) + (DRAFT)
# ============================================================
df["source"] = np.where(df["is_team"]==1, "TEAM", np.where(df["is_fa"]==1, "FA", "TRADE"))

df_used_base = df.copy()

if ALLOW_DRAFT:
    df_used_base = pd.concat([df_used_base, build_draft_pool()], ignore_index=True)

# 注意：Trade 候选就是 source=TRADE 的那部分
# 如果你现在 is_fa 默认都为1，那么外部都是 FA，TRADE池会为空，这是正常的（但Trade就没意义）

# ============================================================
# 11) 求解器（PuLP MILP）
# ============================================================
def solve(df_full, enable_repl_limit=False):
    df_full = df_full.reset_index(drop=True)
    n = len(df_full)
    idxs = list(range(n))

    def pos_mask(pos):
        return [1 if df_full.iloc[i]["pos3"] == pos else 0 for i in idxs]
    maskG, maskF, maskC = pos_mask("G"), pos_mask("F"), pos_mask("C")

    prob = pulp.LpProblem("P2_Draft_FA_Trade", pulp.LpMaximize)

    xS = pulp.LpVariable.dicts("starter", idxs, 0, 1, cat="Binary")
    xB = pulp.LpVariable.dicts("bench",   idxs, 0, 1, cat="Binary")
    keep = pulp.LpVariable.dicts("keep",   idxs, 0, 1, cat="Binary")

    for i in idxs:
        prob += keep[i] == xS[i] + xB[i]
        prob += xS[i] + xB[i] <= 1

    # 人数
    prob += pulp.lpSum([xS[i] for i in idxs]) == 5
    prob += pulp.lpSum([xB[i] for i in idxs]) == 5

    # 位置结构：先发2G2F1C，替补对应
    prob += pulp.lpSum([maskG[i]*xS[i] for i in idxs]) == REQ_G
    prob += pulp.lpSum([maskF[i]*xS[i] for i in idxs]) == REQ_F
    prob += pulp.lpSum([maskC[i]*xS[i] for i in idxs]) == REQ_C
    prob += pulp.lpSum([maskG[i]*xB[i] for i in idxs]) == REQ_G
    prob += pulp.lpSum([maskF[i]*xB[i] for i in idxs]) == REQ_F
    prob += pulp.lpSum([maskC[i]*xB[i] for i in idxs]) == REQ_C

    # 工资帽：对10人施加（更现实）
    prob += pulp.lpSum([keep[i]*float(df_full.at[i,"salary_2025"]) for i in idxs]) <= S_cap

    # 外部球员来源限制：
    # - 若 source=FA：允许（is_fa=1）
    # - 若 source=TRADE：只有 ALLOW_TRADES 时允许，并受 MAX_TRADES 限制
    # - 若 source=DRAFT：只有 ALLOW_DRAFT 时允许，并受 MAX_DRAFT 限制
    # - 若 source=TEAM：当然允许

    trade_idxs = []
    draft_idxs = []
    repl_idxs  = []
    for i in idxs:
        src = str(df_full.at[i,"source"])
        if src == "TRADE":
            trade_idxs.append(i)
            if not ALLOW_TRADES:
                prob += keep[i] == 0
        if src == "DRAFT":
            draft_idxs.append(i)
            if not ALLOW_DRAFT:
                prob += keep[i] == 0
        if src == "REPL":
            repl_idxs.append(i)

        # 若外部且 is_fa=0 且 不是TRADE/DRAFT，则禁用
        is_team = int(df_full.at[i,"is_team"])
        is_fa   = int(df_full.at[i,"is_fa"])
        if is_team==0 and is_fa==0 and src not in ("TRADE","DRAFT"):
            prob += keep[i] == 0

    # 稳定性：引入人数限制（非本队）
    new_total = pulp.lpSum([keep[i]*(1-int(df_full.at[i,"is_team"])) for i in idxs])
    new_starters = pulp.lpSum([xS[i]*(1-int(df_full.at[i,"is_team"])) for i in idxs])
    team_starters = pulp.lpSum([xS[i]*int(df_full.at[i,"is_team"]) for i in idxs])

    prob += new_total <= MAX_NEW_TOTAL
    prob += new_starters <= MAX_NEW_STARTERS
    if int(df_full["is_team"].sum()) > 0:
        prob += team_starters >= min(MIN_TEAM_STARTERS, int(df_full["is_team"].sum()))

    # 交易/选秀数量限制
    if trade_idxs:
        prob += pulp.lpSum([keep[i] for i in trade_idxs]) <= MAX_TRADES
    if draft_idxs:
        prob += pulp.lpSum([keep[i] for i in draft_idxs]) <= MAX_DRAFT

    # replacement 限制（可选）
    if enable_repl_limit and repl_idxs:
        prob += pulp.lpSum([keep[i] for i in repl_idxs]) <= MAX_REPLACEMENTS

    # 目标函数
    cost_scale = S_cap
    obj_terms = []

    for i in idxs:
        sal = float(df_full.at[i,"salary_2025"])
        perf = float(df_full.at[i,"perf_norm"])
        pop  = float(df_full.at[i,"pop_norm"])
        is_team = int(df_full.at[i,"is_team"])
        src = str(df_full.at[i,"source"])

        value = (
            xS[i]*(w_perf*STARTER_PERF_W*perf + w_pop*STARTER_POP_W*pop)
            + xB[i]*(w_perf*BENCH_PERF_W*perf + w_pop*BENCH_POP_W*pop)
        )

        # 成本项：FA签约费 / Trade交易费 / Draft成本（可设很小）/ Replacement硬惩罚
        sign_cost = 0.0
        if src == "FA" and is_team==0:
            sign_cost = SIGN_COST_RATE * sal
        trade_cost = 0.0
        if src == "TRADE":
            trade_cost = TRADE_FEE_RATE * sal
        draft_cost = 0.0
        if src == "DRAFT":
            draft_cost = float(get_param("DRAFT_COST", 25000.0))  # 固定成本（象征性）

        # 变动惩罚：非本队被选中
        change_pen = CHANGE_PENALTY if is_team==0 else 0.0

        # Replacement 强惩罚
        repl_pen = REPL_HARD_PEN if src == "REPL" else 0.0

        obj_terms.append(
            value
            - keep[i]*(sign_cost/cost_scale)
            - keep[i]*(trade_cost/cost_scale)
            - keep[i]*(draft_cost/cost_scale)
            - keep[i]*change_pen
            - EPS_SAL*(sal/S_cap)*keep[i]
            - repl_pen*keep[i]
        )

    prob += pulp.lpSum(obj_terms)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    status = prob.solve(solver)
    st = pulp.LpStatus[status]
    if st not in ("Optimal","Feasible"):
        return None, st

    starters = [i for i in idxs if pulp.value(xS[i]) > 0.5]
    bench    = [i for i in idxs if pulp.value(xB[i]) > 0.5]
    obj_val  = float(pulp.value(prob.objective))
    return (starters, bench, obj_val), st

# ============================================================
# 12) 两阶段：先不加 Replacement；不行再少量兜底
# ============================================================
df_base = df_used_base.copy().reset_index(drop=True)
sol, st = solve(df_base, enable_repl_limit=False)

min_salary_used = None
df_final = df_base

if sol is None and ALLOW_REPLACEMENT_IF_INFEASIBLE:
    print("[WARN] 无可行解：启用少量 Replacement 兜底（<=1）...")
    for ms in MIN_SALARY_TRY:
        df_try = pd.concat([df_base, build_replacement_pool(ms)], ignore_index=True).reset_index(drop=True)
        sol2, st2 = solve(df_try, enable_repl_limit=True)
        if sol2 is not None:
            sol, st = sol2, st2
            df_final = df_try
            min_salary_used = ms
            break

if sol is None:
    raise RuntimeError(f"仍无可行解（状态={st}）。建议：放宽工资帽/稳定性约束，或降低替补人数要求。")

starters_idx, bench_idx, obj_val = sol
df_final = df_final.reset_index(drop=True)

# ============================================================
# 13) 位置配对输出（Starter ↔ Bench）
# ============================================================
def group_by_pos(idxs_):
    groups = {"G": [], "F": [], "C": []}
    for i in idxs_:
        pos = df_final.iloc[i]["pos3"]
        if pos in groups:
            groups[pos].append(i)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda j: float(df_final.iloc[j]["perf_eff"]), reverse=True)
    return groups

Sg = group_by_pos(starters_idx)
Bg = group_by_pos(bench_idx)

pairs = []
for pos in ["G","F","C"]:
    for s_i, b_i in zip(Sg[pos], Bg[pos]):
        pairs.append((pos, s_i, b_i))

def pack(role, idxs_):
    cols = ["player","team","source","pos3","salary_2025","is_team","is_fa","perf_eff","p_injury","pop_raw"]
    out = df_final.iloc[idxs_][cols].copy()
    out["role"] = role
    return out

starter_df = pack("Starter", starters_idx).sort_values(["pos3","salary_2025"], ascending=[True,False])
bench_df   = pack("Bench",   bench_idx).sort_values(["pos3","salary_2025"], ascending=[True,False])

total_salary = float(starter_df["salary_2025"].sum() + bench_df["salary_2025"].sum())
team_keep_n  = int(starter_df["is_team"].sum() + bench_df["is_team"].sum())
new_n        = 10 - team_keep_n

trade_n = int((starter_df["source"]=="TRADE").sum() + (bench_df["source"]=="TRADE").sum())
draft_n = int((starter_df["source"]=="DRAFT").sum() + (bench_df["source"]=="DRAFT").sum())
repl_n  = int((starter_df["source"]=="REPL").sum()  + (bench_df["source"]=="REPL").sum())
fa_n    = int((starter_df["source"]=="FA").sum()    + (bench_df["source"]=="FA").sum())

print("\n==============================")
print("子问题2：Draft + FA + Trade + Stability（5先发 + 5替补）")
print("==============================")
print(f"求解状态: {st}")
print(f"工资帽 S_cap = {S_cap:,.0f}")
print(f"10人总薪资 = {total_salary:,.0f} (占工资帽 {total_salary/S_cap:.1%})")
print(f"保留本队人数 = {team_keep_n} | 外部引入人数 = {new_n}")
print(f"来源分解：FA={fa_n} | TRADE={trade_n} | DRAFT={draft_n} | REPL={repl_n}" +
      (f" (REPL底薪={min_salary_used})" if min_salary_used else ""))
print(f"目标权重: 竞技 {w_perf:.2f} / 商业 {w_pop:.2f}")
print(f"最优目标值: {obj_val:.6f}")

print("\n—— 先发5人 ——")
print(starter_df.to_string(index=False))

print("\n—— 候补5人 ——")
print(bench_df.to_string(index=False))

pair_df = pd.DataFrame([{
    "Pos": pos,
    "Starter": df_final.iloc[s_i]["player"],
    "Starter_Source": df_final.iloc[s_i]["source"],
    "Starter_Team": df_final.iloc[s_i]["team"],
    "Starter_Salary": float(df_final.iloc[s_i]["salary_2025"]),
    "Bench": df_final.iloc[b_i]["player"],
    "Bench_Source": df_final.iloc[b_i]["source"],
    "Bench_Team": df_final.iloc[b_i]["team"],
    "Bench_Salary": float(df_final.iloc[b_i]["salary_2025"]),
} for pos, s_i, b_i in pairs])

print("\n—— 位置一一对应（Starter ↔ Bench）——")
print(pair_df.to_string(index=False))

# 导出
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
starter_df.to_csv(os.path.join(BASE_DIR, f"p2_starers_{ts}.csv"), index=False, encoding="utf-8-sig")
bench_df.to_csv(os.path.join(BASE_DIR, f"p2_bench_{ts}.csv"),   index=False, encoding="utf-8-sig")
pair_df.to_csv(os.path.join(BASE_DIR, f"p2_pairs_{ts}.csv"),    index=False, encoding="utf-8-sig")
print("\n[OK] 已导出：p2_starers_*.csv / p2_bench_*.csv / p2_pairs_*.csv")







# ============================================================
# 6) 网格搜索：alpha × MAX_NEW_TOTAL → 可行域/敏感性热力图数据
#    建议放在脚本末尾（导出 starters/bench/pairs 之后）
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _read_alpha_range_from_params(param_csv="模型参数数据：model_parameters.csv", fallback=(0.2, 0.8)):
    try:
        dfp = pd.read_csv(param_csv, encoding="utf-8")
        row = dfp.loc[dfp["symbol"].astype(str).str.strip().str.lower() == "alpha"]
        if len(row) == 0:
            return fallback
        a_min = float(row["min_value"].iloc[0])
        a_max = float(row["max_value"].iloc[0])
        if not np.isfinite(a_min) or not np.isfinite(a_max) or a_min == a_max:
            return fallback
        return (a_min, a_max)
    except Exception:
        return fallback

def solve_with_current_globals(df_base_in):
    """
    复用你脚本现有的 replacement 兜底逻辑，保证网格点求解方式一致。
    返回: feasible(0/1), obj_value(float or nan)
    """
    # 第一轮：不用 replacement
    sol_local, st_local = solve(df_base_in, enable_repl_limit=False)

    # 第二轮：必要时启用少量 replacement（与主流程一致）
    if sol_local is None and ALLOW_REPLACEMENT_IF_INFEASIBLE:
        for ms in MIN_SALARY_TRY:
            df_try = pd.concat([df_base_in, build_replacement_pool(ms)], ignore_index=True)
            sol2, st2 = solve(df_try, enable_repl_limit=True)
            if sol2 is not None:
                sol_local, st_local = sol2, st2
                break

    if sol_local is None:
        return 0, np.nan
    else:
        starters, bench, objv = sol_local
        return 1, float(objv)

def run_grid_alpha_maxnew(
    df_base_in,
    alpha_list,
    max_new_list,
    out_csv="p2_grid_alpha_maxnew.csv",
    out_png="p2_heatmap_alpha_maxnew.png"
):
    global w_pop, w_perf, MAX_NEW_TOTAL, MAX_NEW_STARTERS

    records = []
    # 保留原始设置，跑完恢复，避免污染主结果
    _w_pop0, _w_perf0 = w_pop, w_perf
    _mnew0, _mnewS0  = MAX_NEW_TOTAL, MAX_NEW_STARTERS

    for a in alpha_list:
        # alpha = 商业权重
        w_pop  = float(a)
        w_perf = 1.0 - w_pop

        for mnew in max_new_list:
            MAX_NEW_TOTAL = int(mnew)
            # 一个常见且更稳的联动：先发引入人数不应超过总引入人数
            # 在函数开头你已经存了 _mnewS0 = MAX_NEW_STARTERS
            MAX_NEW_STARTERS = int(min(_mnewS0, MAX_NEW_TOTAL))

            feasible, objv = solve_with_current_globals(df_base_in)

            records.append({
                "alpha": float(a),
                "MAX_NEW_TOTAL": int(mnew),
                "objective_value": objv,
                "feasible": int(feasible)
            })

    # 恢复原设置
    w_pop, w_perf = _w_pop0, _w_perf0
    MAX_NEW_TOTAL, MAX_NEW_STARTERS = _mnew0, _mnewS0

    grid_df = pd.DataFrame(records)
    grid_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 网格结果已保存：{out_csv}")

    # --- 画热力图（可行=1时用 objective_value，否则留空/最小值） ---
    piv = grid_df.pivot(index="MAX_NEW_TOTAL", columns="alpha", values="objective_value")

    # 为了可视化，把不可行点填充为最小值（或你也可以填 NaN）
    data = piv.values.copy()
    finite_vals = data[np.isfinite(data)]
    if finite_vals.size == 0:
        print("[WARN] 全部网格点不可行，无法绘制有效热力图。")
        return grid_df

    fill_val = np.nanmin(finite_vals) - 0.05 * (np.nanmax(finite_vals) - np.nanmin(finite_vals) + 1e-9)
    data_filled = np.where(np.isfinite(data), data, fill_val)

    plt.figure(figsize=(10, 4.8))
    im = plt.imshow(
        data_filled,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    plt.colorbar(im, label="Objective Value (filled for infeasible)")

    plt.xticks(
        ticks=np.arange(len(piv.columns)),
        labels=[f"{x:.2f}" for x in piv.columns],
        rotation=0
    )
    plt.yticks(
        ticks=np.arange(len(piv.index)),
        labels=[str(int(y)) for y in piv.index]
    )
    plt.xlabel("alpha (commercial weight)")
    plt.ylabel("MAX_NEW_TOTAL (external acquisitions limit)")
    plt.title("Feasibility & Sensitivity Map: alpha × MAX_NEW_TOTAL")

    # 标记不可行点：打叉（可选）
    infeas = grid_df.loc[grid_df["feasible"] == 0, ["MAX_NEW_TOTAL", "alpha"]]
    alpha_to_col = {a:i for i,a in enumerate(piv.columns)}
    maxnew_to_row = {m:i for i,m in enumerate(piv.index)}
    xs, ys = [], []
    for _, r in infeas.iterrows():
        a = float(r["alpha"]); m = int(r["MAX_NEW_TOTAL"])
        if a in alpha_to_col and m in maxnew_to_row:
            xs.append(alpha_to_col[a])
            ys.append(maxnew_to_row[m])
    if len(xs) > 0:
        plt.scatter(xs, ys, marker="x")  # 不指定颜色，遵循你之前的作图习惯

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[OK] 热力图已保存：{out_png}")

    return grid_df


# ====== 真实执行：alpha 从参数表读范围；MAX_NEW_TOTAL 做网格 ======
a_lo, a_hi = _read_alpha_range_from_params("模型参数数据：model_parameters.csv", fallback=(0.20, 0.80))
alpha_list = np.linspace(a_lo, a_hi, 9)           # 9个点够画热力图且不会太慢
max_new_list = list(range(0, MAX_NEW_TOTAL + 1))  # 0..默认上限

# 注意：这里用 df_base（replacement 前的候选池）更能体现“稳定性约束”带来的可行域变化
_ = run_grid_alpha_maxnew(df_base, alpha_list, max_new_list)
