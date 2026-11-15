# p1_compare_3_strategies_rep_dd.py
# ------------------------------------------------
# 子问题1改进版（满足你的新要求）：
# 1) 三策略对比：Conservative / Aggressive / Optimal
# 2) 终值用最后K=6期“平均收入”估值（削弱末期投机）
# 3) 末期竞技声誉/延续价值：用最后K期平均(P,W)加终值bonus
# 4) 估值回撤惩罚：强推V平稳上升（避免“低谷/盆地”）
# 5) 输出主表：year,strategy,W,attend,P,V,D,C
# 6) 输出debug表：含f,p,u,annual_revenue等便于解释与复现
#
# 运行：python p1_compare_3_strategies_rep_dd.py

import numpy as np
import pandas as pd
from math import exp

try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError("需要 scipy：pip install scipy") from e


# =========================
# 0) 全局参数
# =========================
T = 3.0
steps_per_year = 6
N = int(T * steps_per_year)
dt = 1 / steps_per_year

# 终值窗口：K=6（最后一年）
K_TERMINAL = 6

# 工资帽（示例，可替换成你真实数据）
cap_2025 = 1.5071e6
cap_2026 = 1.60e6
cap_2027 = 1.70e6

def salary_cap(time):
    if time < 1.0:
        return cap_2025
    elif time < 2.0:
        return cap_2026
    else:
        return cap_2027

# 借贷利率（示例）
r_base = 0.05
r_raise = 0.08

def interest_rate(time):
    return r_base if time < 1.0 else r_raise

# 控制变量范围
min_frac, max_frac = 0.75, 1.00
p_min, p_max = 0.50, 2.00
u_min, u_max = 0.00, 1.00

# 票务与运营
baseline_price = 25.0
elasticity = 0.5
capacity = 12000
home_games_per_year = 20

overhead = 2.0e6
extra_per_attendee = 10.0

# 人气动态
k_W = 1.0
k_A = 0.3

# 折现率、估值倍数
rho = 0.05
multiple = 5.0

# 现金管理（避免C永远=0）
retain_cash_ratio = 0.40
max_repay_ratio = 0.80

# 初始状态
pop0 = 0.80
debt0 = 0.0
cash0 = 0.0


# =========================
# 1) 新增：两类“现实化”项（你要的重点）
# =========================
# (A) 末期竞技声誉/延续价值权重（美元量级）
# 解释：最后一年平均 P、W 越高，越有品牌/续约/商业的延续价值
ETA_P = 1.2e7   # 人气延续价值（可调：越大越不敢伤P）
ETA_W = 1.2e7   # 竞技延续价值（可调：越大越不敢末期砍f）

# (B) 估值回撤惩罚：强推 V 平稳上升
# 解释：对 V_{t-1} > V_t 的跌幅进行二次惩罚，避免“盆地”
LAMBDA_DD = 3.0e7  # 可调：越大越接近单调不降（过大可能让解更保守）
V_SCALE = 1.0e8    # 归一化尺度，避免数值过大/过小


# =========================
# 2) 仍保留：控制变量平滑惩罚（调整成本）
# =========================
LAMBDA_P = 0.12  # p平滑（建议略大，避免票价“跳边界”）
LAMBDA_F = 0.08  # f平滑
LAMBDA_U = 0.03  # u平滑

def smooth_penalty(x):
    f = x[0::3]
    p = x[1::3]
    u = x[2::3]
    df = np.diff(f); dp = np.diff(p); du = np.diff(u)
    return float(LAMBDA_F*np.sum(df**2) + LAMBDA_P*np.sum(dp**2) + LAMBDA_U*np.sum(du**2))


# =========================
# 3) 子模块：竞技、需求、收入、估值
# =========================
def win_rate_from_spending(f):
    # 线性近似：f=1 -> 0.8; f=0.75 -> 0.5
    W = 1.2*f - 0.4
    return float(np.clip(W, 0.0, 1.0))

def calc_attendance(pop, p):
    demand = pop * (p ** (-elasticity))
    return float(capacity if demand > 1 else capacity * demand)

def calc_annual_revenue(attendance, pop, time, p):
    ticket_rev = attendance * baseline_price * p * home_games_per_year
    extra_rev  = attendance * extra_per_attendee * home_games_per_year

    if time < 1.0:
        media = 2.0e6
    elif time < 2.0:
        media = 2.0e6 * 0.9
    else:
        media = 2.0e6 * 1.2

    sponsor = 1.0e6 * (0.5 + 0.5*pop)
    return float(ticket_rev + extra_rev + media + sponsor)

def enterprise_value_from_revenue(annual_revenue):
    return float(multiple * annual_revenue)

def equity_value(EV, debt, cash):
    return float(EV - debt + cash)


# =========================
# 4) 核心仿真：给定 x，返回（主轨迹、debug轨迹、目标分解）
# =========================
def simulate_core(x, strategy_name="TMP", include_debug=True):
    cash = float(cash0)
    debt = float(debt0)
    pop  = float(pop0)

    time = 0.0
    J_profit = 0.0

    rows_main = []
    rows_dbg = []

    rev_list = []
    P_list = []
    W_list = []
    Vnet_list = []

    for i in range(N):
        f = float(np.clip(x[3*i],     min_frac, max_frac))
        p = float(np.clip(x[3*i + 1], p_min,    p_max))
        u = float(np.clip(x[3*i + 2], u_min,    u_max))

        cap = salary_cap(time)
        r_d = interest_rate(time)

        W = win_rate_from_spending(f)
        attendance = calc_attendance(pop, p)
        annual_revenue = calc_annual_revenue(attendance, pop, time, p)

        rev_list.append(annual_revenue)
        P_list.append(pop)   # 注意：这里记录的是“期初pop”，也可改成更新后pop，差别很小
        W_list.append(W)

        salary_cost = f * cap
        annual_cost = salary_cost + overhead
        profit = (annual_revenue - annual_cost - debt*r_d) * dt

        # 现金/债务更新
        cash += profit
        if cash < 0:
            deficit = -cash
            debt += u * deficit
            cash = 0.0
        else:
            retained = retain_cash_ratio * cash
            available = cash - retained
            if debt > 0 and available > 0:
                repay = min(debt, available * max_repay_ratio)
                debt -= repay
                available -= repay
            cash = retained + available

        # 人气更新（离散化）
        pop_next = pop + (k_W*(W - pop) + k_A*(attendance/capacity - pop)) * dt
        pop_next = float(np.clip(pop_next, 0.0, 1.0))
        pop = pop_next

        # 估值展示（V = EV - D）
        EV = enterprise_value_from_revenue(annual_revenue)
        V_net = EV - debt
        Vnet_list.append(V_net)

        time_next = time + dt
        J_profit += exp(-rho * time_next) * profit

        rows_main.append({
            "year": round(time_next, 4),
            "strategy": strategy_name,
            "W": float(W),
            "attend": float(attendance),
            "P": float(pop),
            "V": float(V_net),
            "D": float(debt),
            "C": float(cash),
        })

        if include_debug:
            rows_dbg.append({
                "year": round(time_next, 4),
                "strategy": strategy_name,
                "f_salary_ratio": f,
                "p_ticket_multiplier": p,
                "u_debt_ratio": u,
                "cap": float(cap),
                "annual_revenue": float(annual_revenue),
                "annual_cost": float(annual_cost),
                "interest_rate": float(r_d),
                "profit_dt": float(profit),
                "W": float(W),
                "attend": float(attendance),
                "P": float(pop),
                "EV": float(EV),
                "V_net": float(V_net),
                "D": float(debt),
                "C": float(cash),
            })

        time = time_next

    # -------- 终值（K期平均收入） --------
    K = min(K_TERMINAL, len(rev_list))
    avg_rev_terminal = float(np.mean(rev_list[-K:]))
    EV_terminal = enterprise_value_from_revenue(avg_rev_terminal)
    terminal_equity = equity_value(EV_terminal, debt, cash)

    # -------- 末期竞技声誉/延续价值 bonus（用最后K期平均 P/W）--------
    # 用更新后pop更贴近“期末状态”，这里使用主表末K期的 P/W 更直观
    df_tmp = pd.DataFrame(rows_main)
    avgP = float(df_tmp["P"].tail(K).mean())
    avgW = float(df_tmp["W"].tail(K).mean())
    continuation_bonus = exp(-rho * T) * (ETA_P * avgP + ETA_W * avgW)

    # -------- 估值回撤惩罚（避免盆地）--------
    Varr = np.array(Vnet_list, dtype=float)
    dV = Varr[:-1] - Varr[1:]
    drops = np.maximum(0.0, dV) / V_SCALE
    drawdown_pen = LAMBDA_DD * float(np.sum(drops**2))

    df_main = pd.DataFrame(rows_main, columns=["year","strategy","W","attend","P","V","D","C"])
    df_dbg = pd.DataFrame(rows_dbg) if include_debug else pd.DataFrame()

    # 总目标（要最大化）
    J_total = J_profit + exp(-rho*T)*terminal_equity + continuation_bonus - drawdown_pen

    info = {
        "J_profit": float(J_profit),
        "terminal_equity_disc": float(exp(-rho*T)*terminal_equity),
        "continuation_bonus": float(continuation_bonus),
        "drawdown_pen": float(drawdown_pen),
        "avgP_lastK": float(avgP),
        "avgW_lastK": float(avgW),
    }
    return df_main, df_dbg, float(J_total), info


# =========================
# 5) 最优控制求解（最大化 J_total - smooth_penalty）
# =========================
def objective_for_minimize(x):
    _, _, J_total, _ = simulate_core(x, include_debug=False)
    return -(J_total - smooth_penalty(x))

def solve_optimal_control():
    x0 = np.array([0.90, 1.00, 0.50] * N, dtype=float)
    bounds = [(min_frac, max_frac), (p_min, p_max), (u_min, u_max)] * N

    res = minimize(
        objective_for_minimize,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-7, "disp": False},
    )
    return res


# =========================
# 6) 三条策略
# =========================
def build_constant_policy(f_const, p_const, u_const):
    return np.array([f_const, p_const, u_const] * N, dtype=float)

def get_three_strategies_x(opt_x):
    x_conservative = build_constant_policy(
        f_const=0.78,
        p_const=1.25,
        u_const=0.10
    )
    x_aggressive = build_constant_policy(
        f_const=1.00,
        p_const=0.75,
        u_const=0.85
    )
    x_optimal = opt_x.copy()
    return x_conservative, x_aggressive, x_optimal


# =========================
# 7) 主程序：输出主表 + debug表 + 摘要
# =========================
def summarize(df_main, J, info):
    last = df_main.iloc[-1]
    v = df_main["V"].values
    drops = int(np.sum(np.diff(v) < -1e-9))
    return {
        "strategy": df_main["strategy"].iloc[0],
        "J_total": float(J),
        "final_W": float(last["W"]),
        "final_P": float(last["P"]),
        "final_V": float(last["V"]),
        "final_D": float(last["D"]),
        "final_C": float(last["C"]),
        "V_drop_count": drops,
        "avgP_lastK": info.get("avgP_lastK", np.nan),
        "avgW_lastK": info.get("avgW_lastK", np.nan),
        "J_profit": info.get("J_profit", np.nan),
        "term_equity_disc": info.get("terminal_equity_disc", np.nan),
        "cont_bonus": info.get("continuation_bonus", np.nan),
        "dd_pen": info.get("drawdown_pen", np.nan),
    }

def main():
    print("=== 求解 Optimal（含：终值K=6均值 + 末期声誉bonus + 估值回撤惩罚）===")
    res = solve_optimal_control()
    print("优化成功：", res.success, "迭代次数：", res.nit)
    if not res.success:
        print("[WARN] 未完全收敛，将使用当前数值解（论文可写为数值近似最优）")

    x_con, x_agg, x_opt = get_three_strategies_x(res.x)

    print("\n=== 仿真三策略 ===")
    con_main, con_dbg, J_con, info_con = simulate_core(x_con, "Conservative", include_debug=True)
    agg_main, agg_dbg, J_agg, info_agg = simulate_core(x_agg, "Aggressive", include_debug=True)
    opt_main, opt_dbg, J_opt, info_opt = simulate_core(x_opt, "Optimal", include_debug=True)

    df_all = pd.concat([con_main, agg_main, opt_main], ignore_index=True)
    df_dbg = pd.concat([con_dbg, agg_dbg, opt_dbg], ignore_index=True)

    out_main = "p1_strategy_comparison_table.csv"
    out_dbg  = "p1_strategy_comparison_debug.csv"
    df_all.to_csv(out_main, index=False, encoding="utf-8-sig")
    df_dbg.to_csv(out_dbg,  index=False, encoding="utf-8-sig")
    print(f"[OK] 主表已保存：{out_main}")
    print(f"[OK] Debug表已保存：{out_dbg}")

    summary = pd.DataFrame([
        summarize(con_main, J_con, info_con),
        summarize(agg_main, J_agg, info_agg),
        summarize(opt_main, J_opt, info_opt),
    ])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n=== 摘要对比（含V下跌次数、末期声誉、分解项）===")
    print(summary.to_string(index=False))

    # 快速提示：如果你要“几乎严格单调上升”
    opt_drops = int(summary.loc[summary["strategy"]=="Optimal","V_drop_count"].iloc[0])
    if opt_drops > 0:
        print("\n[INFO] Optimal 仍出现少量V回撤：")
        print("  -> 优先把 LAMBDA_DD 提高（例如 3e7 -> 8e7）")
        print("  -> 或把 LAMBDA_P/LAMBDA_F 再提高一点（减少边界跳变）")
        print("  -> 或把 ETA_W/ETA_P 提高（末期更不敢牺牲竞技/人气）")
    else:
        print("\n[OK] Optimal 的 V 已实现平稳上升（或近似不下降）。")


if __name__ == "__main__":
    main()
