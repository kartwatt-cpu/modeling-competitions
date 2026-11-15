import os
import re
import unicodedata
import numpy as np
import pandas as pd
from itertools import combinations

# ============================================================
# 子问题3：联盟扩张（扩张选秀）- Minimax Regret 保护名单
# 数据源：
#   1) ICM_D_WNBA_ThreeFiles_Merged.xlsx  (比赛/票房代理/位置/球队)
#   2) wnba_salaries_2025.csv            (2025薪资)
#   3) 模型参数数据：model_parameters.csv (权重/工资帽等)
# 输出：
#   - 每个球员：剩余价值、概率、遗憾值
#   - 最优保护名单(6)
#   - 最大遗憾值 & 最可能被选走球员
# ============================================================

TEAM = "Las Vegas Aces"

MERGED_XLSX = r"ICM_D_WNBA_ThreeFiles_Merged.xlsx"
SALARY_CSV  = r"wnba_salaries_2025.csv"
PARAM_CSV   = r"模型参数数据：model_parameters.csv"

for fp in [MERGED_XLSX, SALARY_CSV, PARAM_CSV]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"找不到文件: {fp}")

# ----------------------------
# 工具函数：列名挑选 / 名字规范化
# ----------------------------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def norm_name(s: str) -> str:
    """强力名字标准化：弯引号、重音、标点、后缀、空格等"""
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
    修复你薪资表中的 Player 格式：
    'Breanna Stewart B Stewart' -> 'Breanna Stewart'
    'A'ja Wilson A Wilson' -> 'A'ja Wilson'
    """
    s = str(s).strip()
    s2 = re.sub(r"\s+[A-Z]\s+[A-Za-z'\-\.]+$", "", s).strip()
    return s2

def minmax_01(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = np.min(arr), np.max(arr)
    if mx == mn:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)

# ----------------------------
# 读取参数
# ----------------------------
params = pd.read_csv(PARAM_CSV)

def get_param(symbol: str, default: float) -> float:
    if "symbol" not in params.columns:
        return float(default)
    row = params.loc[params["symbol"] == symbol]
    if len(row) == 0:
        return float(default)
    col = "suggested_value" if "suggested_value" in row.columns else row.columns[-1]
    try:
        return float(row[col].iloc[0])
    except Exception:
        return float(default)

S_cap = get_param("S_cap", 1507100.0)  # 2025 工资帽
# 商业权重：优先读 commercial_value_weight，其次 pct_brand
w_pop = get_param("commercial_value_weight", get_param("pct_brand", 0.35))
w_pop = float(np.clip(w_pop, 0.0, 1.0))
w_perf = 1.0 - w_pop

# 扩张城市吸引力因子（可在论文里做敏感性：0.8/1.0/1.2）
CITY_FACTOR = 1.0

# ----------------------------
# 1) 读取薪资表（2025）
# ----------------------------
df_salary = pd.read_csv(SALARY_CSV)

# 兼容列名：player/Player/name + salary_2025/2025 Salary
col_player = pick_col(df_salary, ["player", "Player", "name", "Name"])
if col_player is None:
    raise ValueError("薪资文件缺少 player/Player/name 列。")

col_salary = pick_col(df_salary, ["salary_2025", "2025 Salary", "Salary", "salary"])
if col_salary is None:
    raise ValueError("薪资文件缺少 2025 Salary / salary_2025 列。")

df_salary = df_salary.rename(columns={col_player: "player_raw", col_salary: "salary_2025"})
df_salary["player_raw"] = df_salary["player_raw"].astype(str).str.strip()
df_salary["player"] = df_salary["player_raw"].apply(clean_salary_player)
df_salary["salary_2025"] = pd.to_numeric(df_salary["salary_2025"], errors="coerce").fillna(0.0)

# ----------------------------
# 2) 从 merged.xlsx 补齐 team + position + 构造 perf/pop 代理
# ----------------------------
print("[INFO] 从 merged.xlsx 读取比赛表，构造表现/带票代理...")
xls = pd.ExcelFile(MERGED_XLSX)
sheet_game = None
for s in xls.sheet_names:
    if "MASTER_PLAYER_GAME" in s.upper():
        sheet_game = s
        break
if sheet_game is None:
    sheet_game = "30_MASTER_PLAYER_GAME"

game = pd.read_excel(MERGED_XLSX, sheet_name=sheet_game)

g_player = pick_col(game, ["player","Player","PLAYER","athlete","name"])
g_pos    = pick_col(game, ["pos","Pos","POS","position","Position"])
g_team   = pick_col(game, ["team","Team","TEAM","franchise"])
g_active = pick_col(game, ["active","Active","ACTIVE"])

g_season = pick_col(game, ["season","Season","SEASON","year","Year"])
g_min    = pick_col(game, ["minutes","Minutes","MIN","min"])
g_pts    = pick_col(game, ["points","Points","PTS"])
g_reb    = pick_col(game, ["rebounds","Rebounds","REB","TRB"])
g_ast    = pick_col(game, ["assists","Assists","AST"])
g_stl    = pick_col(game, ["steals","Steals","STL"])
g_blk    = pick_col(game, ["blocks","Blocks","BLK"])
g_tov    = pick_col(game, ["turnovers","Turnovers","TOV"])
g_ticket = pick_col(game, ["ticket_revenue_proxy","Ticket_Revenue_Proxy","ticket_proxy","ticket_revenue"])

need_cols = [g_player, g_pos, g_team, g_season, g_min, g_pts, g_reb, g_ast, g_stl, g_blk, g_tov, g_ticket]
if any(c is None for c in need_cols):
    raise ValueError("merged.xlsx 的比赛表缺少构造 perf/pop 的关键列（player/pos/team/season/min/PTS/REB/AST/STL/BLK/TOV/ticket）。")

gg = game.copy()
if g_active is not None:
    gg = gg[gg[g_active] == True].copy()

gg[g_player] = gg[g_player].astype(str).str.strip()
gg[g_pos]    = gg[g_pos].astype(str).str.strip()
gg[g_team]   = gg[g_team].astype(str).str.strip()

gg[g_season] = pd.to_numeric(gg[g_season], errors="coerce")
for c in [g_min,g_pts,g_reb,g_ast,g_stl,g_blk,g_tov,g_ticket]:
    gg[c] = pd.to_numeric(gg[c], errors="coerce").fillna(0.0)

# 近两季更贴近现实（你也可以改成 >=2022）
gg = gg[gg[g_season] >= 2023].copy()
gg = gg[gg[g_min] > 0].copy()

# 表现代理：单位分钟净贡献（简洁可解释，适合论文）
gg["perf_proxy"] = (
    gg[g_pts]
    + 1.2*gg[g_reb]
    + 1.5*gg[g_ast]
    + 2.0*gg[g_stl]
    + 2.0*gg[g_blk]
    - 2.0*gg[g_tov]
) / gg[g_min].clip(lower=1.0)

# 带票代理：票房代理/分钟（你 merged 里提供的 ticket_revenue_proxy）
gg["pop_proxy"] = gg[g_ticket] / gg[g_min].clip(lower=1.0)

# 位置/球队：取众数（最常出现）
pos_mode = gg.groupby(g_player)[g_pos].agg(lambda x: x.value_counts().index[0]).reset_index()
pos_mode = pos_mode.rename(columns={g_player: "player", g_pos: "position"})
team_mode = gg.groupby(g_player)[g_team].agg(lambda x: x.value_counts().index[0]).reset_index()
team_mode = team_mode.rename(columns={g_player: "player", g_team: "team"})

# perf/pop：取均值
agg = gg.groupby(g_player).agg(
    perf_proxy=("perf_proxy", "mean"),
    pop_proxy=("pop_proxy", "mean"),
    mins=(g_min, "sum"),
).reset_index().rename(columns={g_player: "player"})

# ----------------------------
# 3) 合并：薪资表 x (team/pos/perf/pop)
# ----------------------------
# 用 key 合并（解决名字格式差异）
df_salary["player_key"] = df_salary["player"].apply(norm_name)
pos_mode["player_key"] = pos_mode["player"].apply(norm_name)
team_mode["player_key"] = team_mode["player"].apply(norm_name)
agg["player_key"] = agg["player"].apply(norm_name)

df = df_salary.merge(pos_mode[["player_key","position"]], on="player_key", how="left") \
             .merge(team_mode[["player_key","team"]], on="player_key", how="left") \
             .merge(agg[["player_key","perf_proxy","pop_proxy","mins"]], on="player_key", how="left")

# 只保留 Aces 球员（以 merged 的 team 为准）
df["team"] = df["team"].fillna("Unknown")
df_aces = df[df["team"].str.lower() == TEAM.lower()].copy()

if len(df_aces) == 0:
    # 如果 team 在 merged 中不是标准写法，给你一个提示输出
    print("[WARN] team 精确匹配不到 Aces。下面输出 merged 里出现的球队名示例（前30个唯一值）：")
    print(pd.Series(df["team"].unique()).dropna().head(30).to_string(index=False))
    raise RuntimeError("无法从 merged 数据中筛出 Las Vegas Aces，请检查 team 字段的写法。")

# 去掉无法构造 perf/pop 的球员（通常是没出场/名字对不上）
df_aces["perf_proxy"] = pd.to_numeric(df_aces["perf_proxy"], errors="coerce")
df_aces["pop_proxy"]  = pd.to_numeric(df_aces["pop_proxy"], errors="coerce")

# 对缺失者给很小的 baseline，避免空表（并且论文可解释：缺数据者保守估计）
df_aces["perf_proxy"] = df_aces["perf_proxy"].fillna(0.05)
df_aces["pop_proxy"]  = df_aces["pop_proxy"].fillna(0.05)

# 如果 Aces 球员很多，取 mins 最大的前 N（防止含训练营/短期名单噪声）
df_aces["mins"] = pd.to_numeric(df_aces["mins"], errors="coerce").fillna(0.0)
df_aces = df_aces.sort_values("mins", ascending=False).drop_duplicates("player_key", keep="first").copy()
df_aces = df_aces.head(14).copy()  # 一般 roster < 12，这里取14做鲁棒

# 重置索引（后续组合枚举用位置索引）
df_aces = df_aces.reset_index(drop=True)

# ----------------------------
# 4) 价值计算：剩余价值、被选中概率、遗憾值
# ----------------------------
# 归一化
df_aces["perf_norm"] = minmax_01(df_aces["perf_proxy"].values)
df_aces["pop_norm"]  = minmax_01(df_aces["pop_proxy"].values)

# 人气权重 pop_w：映射到 [0.8, 1.5]（论文好解释：明星更能带动商业）
df_aces["pop_w"] = 0.8 + 0.7 * df_aces["pop_norm"]

# 剩余价值：给一个“稳健、量纲清晰”的打分（0~100级别）
#   value = 100*(w_perf*perf_norm + w_pop*pop_norm) - 20*(salary/S_cap)
df_aces["salary_ratio"] = (df_aces["salary_2025"] / max(S_cap, 1.0)).clip(lower=0.0)
df_aces["remaining_value"] = 100.0 * (w_perf * df_aces["perf_norm"] + w_pop * df_aces["pop_norm"]) \
                             - 20.0 * df_aces["salary_ratio"]

# 被选中概率：基于剩余价值的启发式 + 城市吸引力因子
def pick_prob(v):
    if v >= 60: return 1.0
    if v >= 45: return 0.8
    if v >= 30: return 0.5
    if v >= 10: return 0.2
    return 0.05  # 不是0：扩张队也可能捡漏/补位置

df_aces["pick_prob_base"] = df_aces["remaining_value"].apply(pick_prob)
df_aces["pick_prob"] = (df_aces["pick_prob_base"] * CITY_FACTOR).clip(upper=1.0)

# 遗憾值：若不保护，该球员被选走带来的风险损失
df_aces["regret_value"] = df_aces["pick_prob"] * df_aces["remaining_value"]

# ----------------------------
# 5) Minimax Regret：选 6 人保护
# ----------------------------
players = df_aces["player"].tolist()
vals    = df_aces["remaining_value"].tolist()
probs   = df_aces["pick_prob"].tolist()

if len(players) < 7:
    raise RuntimeError(f"Aces 可用球员数不足以做保护6人（当前={len(players)}）。检查数据筛选或 mins 阈值。")

best_regret = float("inf")
best_set = None
best_worst_player = None

for protect_idx in combinations(range(len(players)), 6):
    worst = -1
    worst_player = None
    for i in range(len(players)):
        if i not in protect_idx:
            r = probs[i] * vals[i]
            if r > worst:
                worst = r
                worst_player = players[i]
    if worst < best_regret:
        best_regret = worst
        best_set = protect_idx
        best_worst_player = worst_player

protected = [players[i] for i in best_set]
unprotected = [p for p in players if p not in protected]

# ----------------------------
# 6) 输出与导出
# ----------------------------
print("\n==============================")
print("子问题3：Aces 扩张选秀保护策略（Minimax Regret）")
print("==============================")
print(f"工资帽 S_cap = {S_cap:,.0f} | 竞技权重={w_perf:.2f} 商业权重={w_pop:.2f} | 城市吸引力因子={CITY_FACTOR:.2f}")
print(f"Aces 候选球员数 = {len(players)}（按mins筛选/去重后）")

# 输出每个球员（剩余价值/遗憾值）
out_cols = ["player", "team", "position", "salary_2025", "mins",
            "perf_proxy", "pop_proxy", "perf_norm", "pop_norm", "pop_w",
            "salary_ratio", "remaining_value", "pick_prob", "regret_value"]

df_show = df_aces[out_cols].copy()
df_show = df_show.sort_values(["regret_value","remaining_value"], ascending=False)

print("\n—— 每位球员：剩余价值 & 遗憾值（假设不保护）——")
print(df_show.to_string(index=False))

# 最优名单
print("\n—— 最优保护名单（6人）——")
for p in protected:
    print(" -", p)

print("\n—— 未保护球员 ——")
print(unprotected)

print("\n—— 最可能造成最大损失的被选走球员 ——")
print(best_worst_player)
print("最大遗憾值（worst-case loss） =", round(best_regret, 4))

# 标记输出表
df_aces["is_protected"] = df_aces["player"].isin(protected).astype(int)
df_aces["is_worst_case_pick"] = (df_aces["player"] == best_worst_player).astype(int)

df_aces.sort_values(["is_protected","regret_value"], ascending=[False, False]) \
      .to_csv("p3_optimal_protection.csv", index=False, encoding="utf-8-sig")
df_show.to_csv("p3_aces_player_values.csv", index=False, encoding="utf-8-sig")

print("\n[INFO] 已导出：p3_aces_player_values.csv")
print("[INFO] 已导出：p3_optimal_protection.csv")
