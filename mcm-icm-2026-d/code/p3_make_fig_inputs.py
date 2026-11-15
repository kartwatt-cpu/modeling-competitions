# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 0) 文件路径（按你本地目录）
# =========================
TEAM_VAL_FILE   = Path("球队估值与运营数据：team_valuations.csv")
ATT_FILE        = Path("球队上座率数据：attendance_data.csv")
MEDIA_FILE      = Path("媒体版权合同数据：media_deals.csv")
PARAM_FILE      = Path("模型参数数据：model_parameters.csv")
DTP_FILE        = Path("ticket_pricing_recommendations_diff_price.csv")  # 可选：用于估计平均票价

TEAM_NAME = "Las Vegas Aces"

OUT_VAL = Path("valuation_trajectories.csv")
OUT_CF  = Path("cashflow_structure.csv")

# =========================
# 1) 读取参数字典
# =========================
def load_params(path: Path):
    df = pd.read_csv(path)
    # 用 parameter 做 key，更稳
    p = {}
    for _, r in df.iterrows():
        key = str(r["parameter"]).strip()
        p[key] = float(r["suggested_value"])
    return p

P = load_params(PARAM_FILE)

def gp(name, default):
    return float(P.get(name, default))

# 你题设里常用参数（若文件缺少则用默认）
DISCOUNT_RATE  = gp("Discount Rate", 0.09)
VAL_MULTIPLE   = gp("Valuation Multiple", 12.0)
G_SHORT        = gp("Revenue Growth Rate Short Term", 0.25)
G_LONG         = gp("Revenue Growth Rate Long Term", 0.12)
WIN_IMPACT     = gp("Win Rate Revenue Impact", 0.08)  # per +0.10 win pct
BRAND_PCT      = gp("Brand Value Pct of Valuation", 0.35)
INJURY_PROB    = gp("Injury Probability Per Season", 0.20)

SALARY_CAP     = gp("Salary Cap 2025", 1507100.0)

# =========================
# 2) 读取 baseline 运营数据
# =========================
tv = pd.read_csv(TEAM_VAL_FILE)
att = pd.read_csv(ATT_FILE)
md  = pd.read_csv(MEDIA_FILE)

row_tv = tv.loc[tv["team"] == TEAM_NAME].iloc[0]
row_att = att.loc[att["team"] == TEAM_NAME].iloc[0]

# baseline 估值与总收入（单位：M -> USD）
V0 = float(row_tv["valuation_2025_M"]) * 1e6
REV0 = float(row_tv["revenue_2024_M"]) * 1e6

# baseline 主场总到场人数（2024）
home_att_total = float(row_att["total_home_attendance_2024"])

# =========================
# 3) 从 DTP 输出估计平均票价（可选）
# =========================
if DTP_FILE.exists():
    df_dtp = pd.read_csv(DTP_FILE)
    # 优先用 base_price 的均值（每场基准票价）
    if "base_price" in df_dtp.columns:
        avg_price = float(df_dtp["base_price"].mean())
    elif "best_price" in df_dtp.columns:
        avg_price = float(df_dtp["best_price"].mean())
    else:
        avg_price = 50.0
else:
    avg_price = 50.0  # 没有 DTP 文件就用保守默认

# 估计 baseline 门票收入（用总到场人数×平均票价）
ticket_rev0 = home_att_total * avg_price

# =========================
# 4) baseline 媒体收入（按联盟合同均摊）
# =========================
# 2025 仍按 current deal
cur = md.loc[md["deal_period"].str.contains("Current", case=False, na=False)].iloc[0]
media_total_2025 = float(cur["annual_value_M"]) * 1e6
teams_2025 = 12
media_rev0 = media_total_2025 / teams_2025

# baseline 赞助收入 = 总收入 - 票房 - 媒体（若为负，强制下限）
sponsor_rev0 = max(REV0 - ticket_rev0 - media_rev0, 0.15 * REV0)

# baseline 成本：工资 + 运营（运营成本用比例近似）
salary_cost0 = SALARY_CAP
op_cost0 = 0.45 * REV0  # 可改：0.35~0.55 都合理

# =========================
# 5) 定义策略与情景（可直接写论文：policy lever + shock assumptions）
# =========================
strategies = {
    "Aggressive": {
        "win_pct": 0.75,
        "salary_ratio": 1.00,
        "marketing_boost": 0.06,   # 让赞助/票房因商业运作上扬
        "op_cost_ratio": 1.05,
    },
    "Balanced": {
        "win_pct": 0.65,
        "salary_ratio": 0.95,
        "marketing_boost": 0.03,
        "op_cost_ratio": 1.00,
    },
    "Conservative": {
        "win_pct": 0.55,
        "salary_ratio": 0.90,
        "marketing_boost": 0.01,
        "op_cost_ratio": 0.97,
    },
}

scenarios = {
    "Baseline": {
        "ticket_shock": 0.00,
        "sponsor_shock": 0.00,
        "injury_shock": 0.00,     # win_pct 降幅
    },
    "InjuryShock": {
        "ticket_shock": -0.08,    # 伤病导致观赛意愿下滑
        "sponsor_shock": -0.05,
        "injury_shock": -0.12,
    },
    "DemandShock": {
        "ticket_shock": -0.12,    # 需求整体下滑（市场/竞争/宏观）
        "sponsor_shock": -0.12,
        "injury_shock": 0.00,
    },
}

# =========================
# 6) 联盟扩张：媒体合同均摊（2026+）
# =========================
new_total = md.loc[md["deal_period"].str.contains("New Total", case=False, na=False)].iloc[0]
media_total_2026p = float(new_total["annual_value_M"]) * 1e6

teams_by_year = {2025: 12, 2026: 13, 2027: 14, 2028: 15}

def media_share(year):
    if year <= 2025:
        return media_total_2025 / teams_by_year[2025]
    else:
        return media_total_2026p / teams_by_year.get(year, 15)

# =========================
# 7) 生成估值轨迹（长表）
# =========================
years = [2025, 2026, 2027, 2028]
t_index = list(range(len(years)))  # time = 0,1,2,3

rows_val = []
rows_cf = []

base_win = strategies["Balanced"]["win_pct"]

for scen_id, scen in scenarios.items():
    for strat_id, strat in strategies.items():

        V = V0
        pop = 1.0  # 简化“品牌/人气指数”，用于估值修正
        peak = V0

        # 每个策略输出一个现金流结构（取 2026 作为“冲击后代表年”）
        cf_2026 = None

        for ti, year in zip(t_index, years):

            # ---- 收入增长率：前两年用 short，后两年用 long（可按你论文改）
            g = G_SHORT if year <= 2026 else G_LONG

            # ---- 竞技影响：win_pct 对收入的提升（WIN_IMPACT per +0.10）
            win_pct = strat["win_pct"] + scen["injury_shock"]
            win_pct = float(np.clip(win_pct, 0.35, 0.85))
            win_factor = 1.0 + WIN_IMPACT * ((win_pct - base_win) / 0.10)

            # ---- 门票收入：基于 baseline ticket_rev0 叠加增长、竞技、人气与冲击
            ticket = ticket_rev0 * ((1 + g) ** ti) * win_factor * (1.0 + scen["ticket_shock"])

            # ---- 媒体收入：按年份均摊（扩张后均摊下降/上升取决于合同）
            media = media_share(year)

            # ---- 赞助收入：受商业策略（marketing_boost）与冲击影响
            sponsor = sponsor_rev0 * ((1 + g) ** ti) * (1.0 + strat["marketing_boost"]) * (1.0 + scen["sponsor_shock"])
            sponsor = max(sponsor, 0.05 * REV0)

            # ---- 成本：工资与运营
            cost_salary = SALARY_CAP * strat["salary_ratio"]
            cost_op = op_cost0 * strat["op_cost_ratio"] * ((1 + 0.06) ** ti)  # 运营成本略通胀

            # ---- 总收入/估值（估值倍数×收入，并用品牌因子做修正）
            total_rev = ticket + media + sponsor

            # 人气/品牌指数演化（非常简化：竞技 + 商业推动，人气有上限）
            pop = pop * (1 + 0.03 * (win_pct - base_win) + strat["marketing_boost"] - 0.02 * abs(scen["ticket_shock"]))
            pop = float(np.clip(pop, 0.75, 1.30))

            V_model = VAL_MULTIPLE * total_rev
            V = V_model * (1.0 + BRAND_PCT * (pop - 1.0))  # 品牌占比越高，人气越能推估值

            # drawdown
            peak = max(peak, V)
            drawdown = (peak - V) / peak if peak > 0 else 0.0

            rows_val.append({
                "time": ti,
                "scenario_id": scen_id,
                "strategy_id": strat_id,
                "valuation": V,
                "drawdown": drawdown
            })

            # 记录 2026 年现金流结构（代表“冲击后”）
            if year == 2026:
                cf_2026 = {
                    "scenario_id": scen_id,
                    "strategy_id": strat_id,
                    "revenue_ticket": ticket,
                    "revenue_media": media,
                    "revenue_sponsor": sponsor,
                    "cost_salary": cost_salary,
                    "cost_operation": cost_op
                }

        rows_cf.append(cf_2026)

# 输出 CSV
df_out_val = pd.DataFrame(rows_val)
df_out_cf  = pd.DataFrame(rows_cf)

df_out_val.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")
df_out_cf.to_csv(OUT_CF, index=False, encoding="utf-8-sig")

print(f"[OK] 已生成: {OUT_VAL.resolve()}")
print(f"[OK] 已生成: {OUT_CF.resolve()}")
print("\nvaluation_trajectories.csv 预览：")
print(df_out_val.head())
print("\ncashflow_structure.csv 预览：")
print(df_out_cf.head())
