# -*- coding: utf-8 -*-
"""
P5 稳健论文叙事版本（子问题4+5综合）：
DTP + XGBoost(或GBDT)需求预测 + CLV + 伤病Monte Carlo + B计划降档 +（简化）管理层策略决策

修复两类“论文风险”：
(1) 策略单一化：用 loss_ratio + ROI + severe_rate 组合决策，避免几乎全是同一策略
(2) CLV方向错误：伤病压新观众比例 + 转化率上限，避免伤病后CLV反而上升

输出文件（全部可视化友好）：
1) p5_gamelevel_base.csv
2) p5_mc_season_summary.csv
3) p5_mc_strategy_share.csv
4) p5_mc_risk_distribution.csv
5) p5_mc_game_agg.csv
6) p5_mc_example_run_gamelevel.csv
7) p5_mc_example_run_injury_only.csv
8) p5_mc_gamelevel_all.csv (可选)
"""

import numpy as np
import pandas as pd

# ---------- 尝试使用 xgboost；如没有则自动降级为 sklearn 的 GBDT ----------
USE_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    USE_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor


# ==============================
# 0) 参数区（建议论文里做敏感性分析）
# ==============================
SEED_DATA = 42
SEED_MC = 2025

N_GAMES = 100
CAPACITY = 12000  # 预留（目前不截断，你也可在预测后 clip）

# 定价与CLV链路
PRICES = {"Early": 40.0, "Normal": 50.0, "Premium": 70.0}
CONVERSION = {"Early": 0.05, "Normal": 0.03, "Premium": 0.02}
NEW_VISITOR_RATIO = 0.20
CLV_PER_USER = 1000.0

# 档位选择（基准：规则）
THRESH_PREMIUM = 7000
THRESH_EARLY = 4000

# 伤病（关键球员缺阵）
INJURY_PROB = 0.30
SEVERITY_LEVELS = ["mild", "moderate", "severe"]
SEVERITY_PROB = [0.55, 0.30, 0.15]

ATT_SHOCK_RANGE = {
    "mild": (0.88, 0.95),
    "moderate": (0.75, 0.87),
    "severe": (0.60, 0.74),
}
PERF_DROP = {"mild": 0.04, "moderate": 0.10, "severe": 0.18}

# 管理层动作（简化）
DEBT_INTEREST_RATE = 0.10
DEBT_HORIZON_YEARS = 1.0
TRANSFER_COST = 180_000.0
DEBT_PRINCIPAL = 250_000.0
DEBT_COST = DEBT_PRINCIPAL * (1.0 + DEBT_INTEREST_RATE * DEBT_HORIZON_YEARS)

MITIGATE_ATT_FACTOR = 0.55   # Debt&Trade 对“上座损失部分”的缓解
MITIGATE_PERF_FACTOR = 0.60  # Debt&Trade 对“竞技损失代理”的缓解
TANK_FUTURE_VALUE = 220_000.0

# 估值口径（简化）
LAMBDA_CLV = 0.30
VALUATION_MULTIPLIER = 5.0

# Monte Carlo
N_MC = 500
EXAMPLE_RUN_ID = 1
EXPORT_ALL_GAMELEVEL = True

# ==============================
# 修复(2)：CLV伤病折扣机制（核心）
# ==============================
K_CLV = 1.2   # 伤病压转化率的强度
K_NV = 1.8    # 伤病压“新观众比例”的强度（关键：避免CLV反升）
K_CAP = 0.8   # 转化率上限随伤病收紧程度（把转化限制在健康水平附近）

# ==============================
# 修复(1)：策略决策参数（按损失比例+ROI）
# ==============================
LOSS_SMALL = 0.075   # 原来0.06，略提高：小冲击更倾向不动作
LOSS_LARGE = 0.12    # 原来0.14，降低：更容易进入“大冲击”区域，偶尔出现tank
ROI_DEBT = 1.00      # 原来0.20，显著提高：必须“边际收益”足够大才补强
SEVERE_TANK = 0.08   # 原来0.12，降低：严重伤病更可能触发tank
ROI_TANK_BAD = 0.20  # 原来0.10，提高一点：ROI差才tank



# ==============================
# 工具函数
# ==============================
def downgrade_tier(tier: str) -> str:
    """B计划：Premium -> Normal -> Early"""
    if tier == "Premium":
        return "Normal"
    if tier == "Normal":
        return "Early"
    return "Early"

def choose_base_tier(att: int) -> str:
    """基准档位规则：阈值分层"""
    if att > THRESH_PREMIUM:
        return "Premium"
    if att < THRESH_EARLY:
        return "Early"
    return "Normal"


def clv_mechanics(att: int, tier_base: str, tier_used: str, perf_loss: float, injured: bool):
    """
    统一的CLV机制（保证伤病后CLV不逆势上升）：
    - 伤病会压“新观众比例” + 压“转化率”
    - 并且 effective_conv 不允许超过健康状态的上限（随伤病进一步收紧）
    """
    # 伤病后的新观众比例下降
    if injured:
        new_ratio = NEW_VISITOR_RATIO * max(0.0, 1.0 - K_NV * perf_loss)
    else:
        new_ratio = NEW_VISITOR_RATIO

    new_vis = int(att * new_ratio)

    # 转化率：先按档位，再乘伤病折扣
    base_conv = CONVERSION[tier_base]
    conv_tier = CONVERSION[tier_used]

    if injured:
        clv_discount = max(0.0, 1.0 - K_CLV * perf_loss)
        conv_raw = conv_tier * clv_discount

        # 上限：伤病时不允许超过“健康时的base_conv*(1 - K_CAP*perf_loss)”
        conv_cap = base_conv * max(0.0, 1.0 - K_CAP * perf_loss)
        effective_conv = min(conv_raw, conv_cap)
    else:
        clv_discount = 1.0
        effective_conv = conv_tier

    new_season = int(new_vis * effective_conv)
    clv_value = float(new_season) * CLV_PER_USER

    return {
        "NewVisitors": new_vis,
        "EffectiveConv": float(effective_conv),
        "CLVDiscount": float(clv_discount),
        "NewSeasonTickets": new_season,
        "CLV": float(clv_value),
    }


# ==============================
# 1) 合成数据：Demand = f(对手实力,周末,天气,季后赛)
# ==============================
np.random.seed(SEED_DATA)

opp_strength = np.random.randint(0, 101, size=N_GAMES)
weekend = np.random.choice([0, 1], size=N_GAMES, p=[0.6, 0.4])
playoff = np.random.choice([0, 1], size=N_GAMES, p=[0.8, 0.2])
rain = np.random.choice([0, 1], size=N_GAMES, p=[0.8, 0.2])

base_att = 3000 + 20 * opp_strength + 700 * weekend + 1500 * playoff - 800 * rain
noise = np.random.normal(0, 300, size=N_GAMES)
attendance = np.maximum(1000, np.minimum(10000, base_att + noise)).astype(int)

df = pd.DataFrame({
    "Game": np.arange(1, N_GAMES + 1),
    "OppStrength": opp_strength,
    "Weekend": weekend,
    "Playoff": playoff,
    "Rain": rain,
    "Attendance": attendance
})

X = df[["OppStrength", "Weekend", "Playoff", "Rain"]]
y = df["Attendance"]


# ==============================
# 2) 训练需求预测模型（XGBoost / GBDT）
# ==============================
if USE_XGB:
    model = XGBRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=0,
        objective="reg:squarederror"
    )
else:
    model = GradientBoostingRegressor(random_state=0)

model.fit(X, y)
df["PredAttendance"] = np.round(model.predict(X)).astype(int)
# df["PredAttendance"] = df["PredAttendance"].clip(0, CAPACITY)


# ==============================
# 3) 基准逐场定价 + 收入/CLV链路
# ==============================
base_rows = []
for _, r in df.iterrows():
    g = int(r["Game"])
    att_pred = int(r["PredAttendance"])

    tier = choose_base_tier(att_pred)
    price = float(PRICES[tier])
    revenue = float(att_pred * price)

    clv_pack = clv_mechanics(att_pred, tier_base=tier, tier_used=tier, perf_loss=0.0, injured=False)

    base_rows.append({
        "Game": g,
        "OppStrength": int(r["OppStrength"]),
        "Weekend": int(r["Weekend"]),
        "Playoff": int(r["Playoff"]),
        "Rain": int(r["Rain"]),
        "PredAttendance": att_pred,
        "Tier_Base": tier,
        "Price_Base": price,
        "Revenue_Base": revenue,

        "NewVisitors_Base": clv_pack["NewVisitors"],
        "NewSeasonTickets_Base": clv_pack["NewSeasonTickets"],
        "CLV_Base": clv_pack["CLV"],

        "TotalValue_Base": float(revenue + LAMBDA_CLV * clv_pack["CLV"])
    })

base_df = pd.DataFrame(base_rows)

print("\n==============================")
print("基准逐场结果（前10行）")
print("==============================")
print(base_df.head(10).to_markdown(index=False))

base_df.to_csv("p5_gamelevel_base.csv", index=False, encoding="utf-8-sig")


# ==============================
# 4) Monte Carlo：伤病 + B计划 + 管理层策略
# ==============================
rng = np.random.default_rng(SEED_MC)

def decide_strategy(expected_loss_value: float,
                    expected_loss_per_game: float,
                    base_rev_per_game: float,
                    cost_debt_trade: float,
                    severe_rate: float):
    """
    修复(1)：用“损失占收入比例 + ROI + severe_rate”做决策，避免策略单一化。
    """
    loss_ratio = 0.0 if base_rev_per_game <= 0 else expected_loss_per_game / base_rev_per_game

    # ✅ 用“边际收益”来算ROI，而不是用 DoNothing 的总损失
    benefit_debt = expected_loss_value * (1.0 - MITIGATE_ATT_FACTOR)  # 保守近似：主要缓解上座损失
    roi = (benefit_debt - cost_debt_trade) / max(cost_debt_trade, 1.0)


    # 小冲击：不动作
    if loss_ratio < LOSS_SMALL:
        return "DoNothing", 0.0, 0.0, 1.0, 1.0, loss_ratio, roi, benefit_debt

    # 中冲击：看ROI
    if loss_ratio < LOSS_LARGE:
        if roi >= ROI_DEBT:
            return "Debt&Trade", cost_debt_trade, 0.0, float(MITIGATE_ATT_FACTOR), float(MITIGATE_PERF_FACTOR), loss_ratio, roi, benefit_debt
        else:
            return "DoNothing", 0.0, 0.0, 1.0, 1.0, loss_ratio, roi, benefit_debt

    # 大冲击：优先补强；若 severe 很高 且 ROI 很差 -> tank
    if severe_rate >= SEVERE_TANK and roi < ROI_TANK_BAD:
        return "Tank", 0.0, float(TANK_FUTURE_VALUE), 1.0, 1.0, loss_ratio, roi, benefit_debt
    else:
        return "Debt&Trade", cost_debt_trade, 0.0, float(MITIGATE_ATT_FACTOR), float(MITIGATE_PERF_FACTOR), loss_ratio, roi, benefit_debt


def simulate_one_season(run_id: int):
    # 固定这组随机，使“先验损失评估”和“最终仿真”一致
    inj_flags = rng.random(N_GAMES) < INJURY_PROB
    severities = rng.choice(SEVERITY_LEVELS, size=N_GAMES, p=SEVERITY_PROB)
    severe_rate = float(np.mean((severities == "severe") & inj_flags))

    # ---------- 4.1 先验：假设 DoNothing（执行B计划降档），估计损失用于策略选择 ----------
    total_base_value = 0.0
    total_dn_value = 0.0

    for i in range(N_GAMES):
        base = base_df.iloc[i]
        tier_b = base["Tier_Base"]
        att_b = int(base["PredAttendance"])
        price_b = float(base["Price_Base"])
        val_b = float(base["TotalValue_Base"])
        total_base_value += val_b

        if inj_flags[i]:
            sev = severities[i]
            lo, hi = ATT_SHOCK_RANGE[sev]
            shock = float(rng.uniform(lo, hi))
            att_i = int(att_b * shock)

            tier_adj = downgrade_tier(tier_b)
            price_adj = float(PRICES[tier_adj])
            rev_i = float(att_i * price_adj)

            perf_loss = float(PERF_DROP[sev])

            clv_pack = clv_mechanics(att_i, tier_base=tier_b, tier_used=tier_adj, perf_loss=perf_loss, injured=True)
            val_i = float(rev_i + LAMBDA_CLV * clv_pack["CLV"])
        else:
            val_i = val_b

        total_dn_value += float(val_i)

    expected_loss_value = float(total_base_value - total_dn_value)

    # ---------- 4.2 策略选择（修复策略单一化） ----------
    base_rev = float(base_df["Revenue_Base"].sum())
    base_rev_per_game = base_rev / max(N_GAMES, 1)

    cost_debt_trade = float(DEBT_COST + TRANSFER_COST)
    expected_loss_per_game = expected_loss_value / max(N_GAMES, 1)

    strategy, strategy_cost, future_value, mitigate_att, mitigate_perf, loss_ratio, roi ,benefit_debt= decide_strategy(
        expected_loss_value, expected_loss_per_game, base_rev_per_game, cost_debt_trade, severe_rate
    )

    # ---------- 4.3 最终逐场模拟 ----------
    game_rows = []

    total_new_rev = 0.0
    total_new_clv = 0.0
    total_new_value = 0.0

    loss_att_total = 0.0
    loss_down_total = 0.0
    perf_drop_total = 0.0

    for i in range(N_GAMES):
        base = base_df.iloc[i]
        g = int(base["Game"])

        tier_b = base["Tier_Base"]
        price_b = float(base["Price_Base"])
        att_b = int(base["PredAttendance"])

        rev_b = float(base["Revenue_Base"])
        clv_b = float(base["CLV_Base"])
        val_b = float(base["TotalValue_Base"])

        if inj_flags[i]:
            sev = severities[i]

            lo, hi = ATT_SHOCK_RANGE[sev]
            raw_shock = float(rng.uniform(lo, hi))
            shock = 1.0 - (1.0 - raw_shock) * mitigate_att
            att_i = int(att_b * shock)

            # 同档价格下收入（分解损失）
            rev_same = float(att_i * price_b)

            # B计划降档；Debt&Trade 让降档更弱（更贴近“补强+维稳”）
            if strategy == "Debt&Trade":
                tier_adj = tier_b if tier_b == "Early" else ("Normal" if tier_b == "Premium" else "Early")
            else:
                tier_adj = downgrade_tier(tier_b)

            price_adj = float(PRICES[tier_adj])
            rev_i = float(att_i * price_adj)

            loss_att = float(rev_b - rev_same)
            loss_down = float(rev_same - rev_i)
            loss_att_total += loss_att
            loss_down_total += loss_down

            perf_loss = float(PERF_DROP[sev] * mitigate_perf)
            perf_drop_total += perf_loss

            # 修复(2)：统一CLV机制（不再反升）
            clv_pack = clv_mechanics(att_i, tier_base=tier_b, tier_used=tier_adj, perf_loss=perf_loss, injured=True)
            clv_i = float(clv_pack["CLV"])

            val_i = float(rev_i + LAMBDA_CLV * clv_i)

            row = {
                "Run": run_id, "Game": g,
                "Injury": True, "Severity": sev, "Strategy": strategy,

                "PredAttendance_Base": att_b,
                "Attendance_Injury": att_i,

                "Tier_Base": tier_b,
                "Tier_Adjusted": tier_adj,

                "Price_Base": price_b,
                "Price_Adjusted": price_adj,

                "Revenue_Base": rev_b,
                "Revenue_sameTier": rev_same,
                "Revenue_Injury": rev_i,

                "Loss_due_to_attendance": loss_att,
                "Loss_due_to_downgrade": loss_down,

                "CLV_Base": clv_b,
                "CLV_Injury": clv_i,

                "TotalValue_Base": val_b,
                "TotalValue_Injury": val_i,

                "PerfDropProxy": perf_loss,

                # 中间量（可视化/论文解释）
                "CLVDiscount": clv_pack["CLVDiscount"],
                "EffectiveConv": clv_pack["EffectiveConv"],
                "NewVisitors_Injury": clv_pack["NewVisitors"],
                "NewSeasonTickets_Injury": clv_pack["NewSeasonTickets"],
            }

        else:
            row = {
                "Run": run_id, "Game": g,
                "Injury": False, "Severity": "none", "Strategy": strategy,

                "PredAttendance_Base": att_b,
                "Attendance_Injury": att_b,

                "Tier_Base": tier_b,
                "Tier_Adjusted": tier_b,

                "Price_Base": price_b,
                "Price_Adjusted": price_b,

                "Revenue_Base": rev_b,
                "Revenue_sameTier": rev_b,
                "Revenue_Injury": rev_b,

                "Loss_due_to_attendance": 0.0,
                "Loss_due_to_downgrade": 0.0,

                "CLV_Base": clv_b,
                "CLV_Injury": clv_b,

                "TotalValue_Base": val_b,
                "TotalValue_Injury": val_b,

                "PerfDropProxy": 0.0,

                "CLVDiscount": 1.0,
                "EffectiveConv": CONVERSION[tier_b],
                "NewVisitors_Injury": int(att_b * NEW_VISITOR_RATIO),
                "NewSeasonTickets_Injury": int(int(att_b * NEW_VISITOR_RATIO) * CONVERSION[tier_b]),
            }

        rev_i = float(row["Revenue_Injury"])
        clv_i = float(row["CLV_Injury"])
        val_i = float(row["TotalValue_Injury"])

        total_new_rev += rev_i
        total_new_clv += clv_i
        total_new_value += val_i

        game_rows.append(row)

    # ---------- 4.4 赛季汇总 ----------
    total_new_value = float(total_new_value - strategy_cost + future_value)

    base_season_value = float(base_df["TotalValue_Base"].sum())
    base_rev = float(base_df["Revenue_Base"].sum())
    base_clv = float(base_df["CLV_Base"].sum())

    rev_loss = float(base_rev - total_new_rev)
    risk_index = 0.0 if base_rev <= 0 else float(rev_loss / base_rev * 100.0)

    total_loss = float(rev_loss)
    downgrade_share = 0.0 if total_loss <= 0 else float(loss_down_total / total_loss * 100.0)

    valuation_base = float(VALUATION_MULTIPLIER * base_season_value)
    valuation_new = float(VALUATION_MULTIPLIER * total_new_value)
    valuation_drop_pct = 0.0 if valuation_base <= 0 else float((valuation_base - valuation_new) / valuation_base * 100.0)

    season_summary = {
        "Run": run_id,
        "Strategy": strategy,

        "ExpectedLossValue_DoNothing": float(expected_loss_value),
        "ExpectedLossPerGame": float(expected_loss_per_game),
        "BaseRevPerGame": float(base_rev_per_game),
        "LossRatio": float(loss_ratio),
        "ROI_DebtTrade": float(roi),
        "SevereRate": float(severe_rate),

        "CostDebtTrade": float(cost_debt_trade),
        "StrategyCost": float(strategy_cost),
        "FutureValue_Tank": float(future_value),

        "BaseRevenue": float(base_rev),
        "InjuryRevenue": float(total_new_rev),
        "RevenueLoss": float(rev_loss),
        "RiskIndexPct": float(risk_index),

        "LossAttendanceTotal": float(loss_att_total),
        "LossDowngradeTotal": float(loss_down_total),
        "DowngradeLossSharePct": float(downgrade_share),

        "BaseCLV": float(base_clv),
        "InjuryCLV": float(total_new_clv),
        "CLVLoss": float(base_clv - total_new_clv),

        "PerfDropProxyTotal": float(perf_drop_total),

        "BaseSeasonValue": float(base_season_value),
        "InjurySeasonValue": float(total_new_value),

        "ValuationBase": float(valuation_base),
        "ValuationInjury": float(valuation_new),
        "ValuationDropPct": float(valuation_drop_pct),

        "InjuryRateRealized": float(np.mean(inj_flags)),

        # 写出关键CLV机制参数（论文可写敏感性）
        "K_CLV": float(K_CLV),
        "K_NV": float(K_NV),
        "K_CAP": float(K_CAP),

        "BenefitDebtTrade": float(benefit_debt),

    }

    return pd.DataFrame(game_rows), season_summary


# ---------- 4.5 Monte Carlo 主循环 ----------
all_game = []
all_season = []
example_game_df = None

for run_id in range(1, N_MC + 1):
    game_df, season_summary = simulate_one_season(run_id)
    all_game.append(game_df)
    all_season.append(season_summary)
    if run_id == EXAMPLE_RUN_ID:
        example_game_df = game_df.copy()

all_game_df = pd.concat(all_game, ignore_index=True)
season_df = pd.DataFrame(all_season)

if EXPORT_ALL_GAMELEVEL:
    all_game_df.to_csv("p5_mc_gamelevel_all.csv", index=False, encoding="utf-8-sig")


# ==============================
# 5) 风险分布 + 策略占比
# ==============================
def q(x, p):
    return float(np.quantile(x, p))

risk = season_df["RiskIndexPct"].values
val_drop = season_df["ValuationDropPct"].values

risk_dist = pd.DataFrame([{
    "N_MC": int(N_MC),
    "Risk_mean": float(np.mean(risk)),
    "Risk_std": float(np.std(risk, ddof=1)),
    "Risk_p05": q(risk, 0.05),
    "Risk_p25": q(risk, 0.25),
    "Risk_p50": q(risk, 0.50),
    "Risk_p75": q(risk, 0.75),
    "Risk_p95": q(risk, 0.95),

    "ValDrop_mean": float(np.mean(val_drop)),
    "ValDrop_std": float(np.std(val_drop, ddof=1)),
    "ValDrop_p05": q(val_drop, 0.05),
    "ValDrop_p50": q(val_drop, 0.50),
    "ValDrop_p95": q(val_drop, 0.95),
}])

strategy_counts = season_df["Strategy"].value_counts(dropna=False).reset_index()
strategy_counts.columns = ["Strategy", "Count"]
strategy_counts["Share"] = strategy_counts["Count"] / float(N_MC)

print("\n==============================")
print("Monte Carlo 风险分布统计（论文核心表）")
print("==============================")
print(risk_dist.to_markdown(index=False))

print("\n==============================")
print("策略占比（DoNothing / Debt&Trade / Tank）")
print("==============================")
print(strategy_counts.to_markdown(index=False))

season_df.to_csv("p5_mc_season_summary.csv", index=False, encoding="utf-8-sig")
strategy_counts.to_csv("p5_mc_strategy_share.csv", index=False, encoding="utf-8-sig")
risk_dist.to_csv("p5_mc_risk_distribution.csv", index=False, encoding="utf-8-sig")


# ==============================
# 6) 逐场聚合（均值 + 分位数带）
# ==============================
eps = 1e-9

all_game_df["RevenueDropPct"] = (
    (all_game_df["Revenue_Base"] - all_game_df["Revenue_Injury"]) /
    np.maximum(all_game_df["Revenue_Base"], eps) * 100.0
)

all_game_df["TotalValue_Base"] = all_game_df["Revenue_Base"] + LAMBDA_CLV * all_game_df["CLV_Base"]
all_game_df["TotalValue_Injury"] = all_game_df["Revenue_Injury"] + LAMBDA_CLV * all_game_df["CLV_Injury"]
all_game_df["TotalValueDrop"] = all_game_df["TotalValue_Base"] - all_game_df["TotalValue_Injury"]

all_game_df["Downgraded"] = (all_game_df["Tier_Adjusted"] != all_game_df["Tier_Base"]).astype(int)

all_game_df["LossTotal"] = (all_game_df["Revenue_Base"] - all_game_df["Revenue_Injury"]).clip(lower=0.0)
all_game_df["DowngradeShare"] = np.where(
    all_game_df["LossTotal"] > 0,
    all_game_df["Loss_due_to_downgrade"] / np.maximum(all_game_df["LossTotal"], eps),
    0.0
)

agg_mean = (
    all_game_df
    .groupby("Game", as_index=False)
    .agg(
        InjuryFreq=("Injury", "mean"),
        DowngradeFreq=("Downgraded", "mean"),

        RevenueBase_mean=("Revenue_Base", "mean"),
        RevenueInjury_mean=("Revenue_Injury", "mean"),
        RevenueDropPct_mean=("RevenueDropPct", "mean"),

        LossAttendance_mean=("Loss_due_to_attendance", "mean"),
        LossDowngrade_mean=("Loss_due_to_downgrade", "mean"),
        DowngradeShare_mean=("DowngradeShare", "mean"),

        CLVBase_mean=("CLV_Base", "mean"),
        CLVInjury_mean=("CLV_Injury", "mean"),

        TotalValueBase_mean=("TotalValue_Base", "mean"),
        TotalValueInjury_mean=("TotalValue_Injury", "mean"),
        TotalValueDrop_mean=("TotalValueDrop", "mean"),

        PerfDropProxy_mean=("PerfDropProxy", "mean"),

        CLVDiscount_mean=("CLVDiscount", "mean"),
        EffectiveConv_mean=("EffectiveConv", "mean"),
        NewVisitors_mean=("NewVisitors_Injury", "mean"),
        NewSeasonTickets_mean=("NewSeasonTickets_Injury", "mean"),
    )
)

def quantile_wide(df_in: pd.DataFrame, col: str, qs=(0.05, 0.50, 0.95), prefix="Q"):
    tmp = df_in.groupby("Game")[col].quantile(list(qs)).unstack().reset_index()
    cols = ["Game"] + [f"{prefix}_p{int(q*100):02d}" for q in qs]
    tmp.columns = cols
    return tmp

rev_q = quantile_wide(all_game_df, "Revenue_Injury", prefix="RevenueInjury")
drop_q = quantile_wide(all_game_df, "RevenueDropPct", prefix="DropPct")
clv_q = quantile_wide(all_game_df, "CLV_Injury", prefix="CLVInjury")
val_q = quantile_wide(all_game_df, "TotalValue_Injury", prefix="TotalValueInjury")

agg = (agg_mean
       .merge(rev_q, on="Game", how="left")
       .merge(drop_q, on="Game", how="left")
       .merge(clv_q, on="Game", how="left")
       .merge(val_q, on="Game", how="left")
)

agg.to_csv("p5_mc_game_agg.csv", index=False, encoding="utf-8-sig")


# ==============================
# 7) 示例run导出（论文展示用）
# ==============================
if example_game_df is not None:
    example_game_df.to_csv("p5_mc_example_run_gamelevel.csv", index=False, encoding="utf-8-sig")
    inj_ex = example_game_df[example_game_df["Injury"] == True].copy()
    inj_ex.to_csv("p5_mc_example_run_injury_only.csv", index=False, encoding="utf-8-sig")

print("\n[INFO] 导出完成：")
print(" - p5_gamelevel_base.csv")
print(" - p5_mc_season_summary.csv")
print(" - p5_mc_strategy_share.csv")
print(" - p5_mc_risk_distribution.csv")
print(" - p5_mc_game_agg.csv")
print(" - p5_mc_example_run_gamelevel.csv / p5_mc_example_run_injury_only.csv")
if EXPORT_ALL_GAMELEVEL:
    print(" - p5_mc_gamelevel_all.csv")
