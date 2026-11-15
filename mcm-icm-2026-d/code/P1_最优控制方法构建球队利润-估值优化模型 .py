import math
import numpy as np
from math import exp
from scipy.optimize import minimize
import pandas as pd

# =========================
# 子问题1：最优控制 + 轨迹表输出
# =========================

# === 模型参数 ===
T = 3.0                             # 规划期3年（2025-2027）
steps_per_year = 6                  # 每年6个决策时段（约每两个月一次）
N = int(T * steps_per_year)         # 总时段数
dt = 1 / steps_per_year             # 单位：年

# 联盟财务参数
cap_2025 = 1.5071e6                 # 2025工资帽（美元）
cap_2026 = 1.6e6                    # 2026工资帽（假设增长）
cap_2027 = 1.7e6                    # 2027工资帽（假设增长）

def salary_cap(time):
    if time < 1.0:
        return cap_2025
    elif time < 2.0:
        return cap_2026
    else:
        return cap_2027

r_base = 0.05                       # 前期年利率5%
r_raise = 0.08                      # 后期年利率8%

def interest_rate(time):
    return r_base if time < 1.0 else r_raise

min_frac = 0.75                     # 最低薪资投入比例（工资帽75%）
max_frac = 1.0                      # 最高投入（工资帽100%）
baseline_price = 25.0               # 基准票价（美元）
elasticity = 0.5                    # 票价需求弹性
capacity = 12000                    # 球馆容量
overhead = 2.0e6                    # 固定运营成本（美元/年）
extra_per_attendee = 10.0           # 每位观众额外消费（美元）

# 人气动态参数
k_W = 1.0                           # 胜率对人气影响
k_A = 0.3                           # 上座率对人气影响

# 估值倍数
multiple = 5.0

# 胜率与薪资投入关系（线性近似）
def win_rate_from_spending(f):
    W = 1.2 * f - 0.4               # f=1 -> 0.8, f=0.5 -> 0.2
    return max(0.0, min(1.0, W))

def compute_attendance(pop, p):
    """由人气与票价弹性计算上座（截断于容量）"""
    demand_factor = pop * (p ** (-elasticity))
    attendance = capacity if demand_factor > 1 else capacity * demand_factor
    return attendance

def compute_annual_revenue(time, pop, p, attendance):
    """年化收入：门票 + 衍生 + 媒体 + 赞助（与你脚本一致）"""
    annual_ticket_rev = attendance * baseline_price * p * 20
    annual_extra_rev  = attendance * extra_per_attendee * 20

    if time < 1.0:
        media = 2.0e6
    elif time < 2.0:
        media = 2.0e6 * 0.9
    else:
        media = 2.0e6 * 1.2

    sponsor = 1.0e6 * (0.5 + 0.5 * pop)

    return annual_ticket_rev + annual_extra_rev + media + sponsor

def simulate_objective(x):
    """
    x: [f0,p0,u0, f1,p1,u1, ...]
    返回 -J（供最小化）
    """
    cash = 0.0
    debt = 0.0
    pop = 0.8
    total_utility = 0.0
    time = 0.0

    for i in range(N):
        f = max(min_frac, min(max_frac, x[3*i]))
        p = max(0.5, min(2.0, x[3*i + 1]))
        u = max(0.0, min(1.0, x[3*i + 2]))

        cap = salary_cap(time)
        r_d = interest_rate(time)

        W = win_rate_from_spending(f)
        attendance = compute_attendance(pop, p)
        annual_revenue = compute_annual_revenue(time, pop, p, attendance)

        salary_cost = f * cap
        annual_cost = salary_cost + overhead

        profit_year = annual_revenue - annual_cost - debt * r_d
        profit = profit_year * dt

        # 更新现金与债务
        cash += profit
        if cash < 0:
            deficit = -cash
            new_debt = u * deficit
            debt += new_debt
            cash = 0.0
        else:
            if debt > 0:
                repay = min(debt, cash)
                debt -= repay
                cash -= repay
            if cash > 0:
                cash = 0.0  # 分红清零（你的原设定）

        # 更新人气
        pop += (k_W * (W - pop) + k_A * (attendance/capacity - pop)) * dt
        pop = max(0.0, min(1.0, pop))

        time += dt
        total_utility += exp(-0.05 * time) * profit

    # 终值估值
    final_p = max(0.5, min(2.0, x[3*N - 2]))
    final_attendance = compute_attendance(pop, final_p)
    final_revenue = compute_annual_revenue(T - dt, pop, final_p, final_attendance)

    enterprise_value = multiple * final_revenue
    equity_value = enterprise_value - debt
    total_utility += exp(-0.05 * T) * equity_value

    return -total_utility

# =========================
# 1) 求解最优控制
# =========================
x0 = np.array([0.9, 1.0, 0.5] * N)
bounds = [(min_frac, max_frac), (0.5, 2.0), (0.0, 1.0)] * N

res = minimize(
    simulate_objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    options={'maxiter': 200}
)

print("优化成功：", res.success, "迭代次数：", res.nit)
x_opt = res.x

# =========================
# 2)（可选）3D surface 数据导出（你原来就有）
# =========================
def evaluate_constant_policy(f_const: float, p_const: float, u_const: float) -> float:
    x_const = np.array([f_const, p_const, u_const] * N, dtype=float)
    valuation = -simulate_objective(x_const)
    return valuation

salary_grid = np.linspace(min_frac, max_frac, 6)
ticket_grid = np.linspace(0.5, 2.0, 7)
debt_grid   = np.linspace(0.0, 1.0, 6)

surface_data = []
for f_const in salary_grid:
    for p_const in ticket_grid:
        for u_const in debt_grid:
            val = evaluate_constant_policy(f_const, p_const, u_const)
            surface_data.append([f_const, p_const, u_const, val])

surface_df = pd.DataFrame(surface_data, columns=["Salary Ratio", "Ticket Multiplier", "Debt Ratio", "Valuation"])
print(surface_df.head())
surface_df.to_csv("surface_data_for_3d_surface.csv", index=False)
print("3D surface 数据已保存到 surface_data_for_3d_surface.csv")

# =========================
# 3) ✅ 新增：输出策略轨迹表（year, strategy, W, attend, P, V, D, C）
# =========================
def export_policy_trace(x, strategy_name="Optimal_Control", out_csv="p1_policy_trace.csv"):
    cash = 0.0
    debt = 0.0
    pop = 0.8
    time = 0.0

    rows = []
    for i in range(N):
        f = max(min_frac, min(max_frac, x[3*i]))
        p = max(0.5, min(2.0, x[3*i + 1]))
        u = max(0.0, min(1.0, x[3*i + 2]))

        cap = salary_cap(time)
        r_d = interest_rate(time)

        W = win_rate_from_spending(f)
        attendance = compute_attendance(pop, p)
        annual_revenue = compute_annual_revenue(time, pop, p, attendance)

        salary_cost = f * cap
        annual_cost = salary_cost + overhead

        profit_year = annual_revenue - annual_cost - debt * r_d
        profit = profit_year * dt

        # 现金/债务更新（与你主仿真一致）
        cash += profit
        if cash < 0:
            deficit = -cash
            debt += u * deficit
            cash = 0.0
        else:
            if debt > 0:
                repay = min(debt, cash)
                debt -= repay
                cash -= repay
            if cash > 0:
                cash = cash

        # 人气更新
        pop += (k_W * (W - pop) + k_A * (attendance/capacity - pop)) * dt
        pop = max(0.0, min(1.0, pop))

        # 估值口径：企业价值（倍数法），同时保留债务/现金
        V = multiple * annual_revenue

        rows.append({
            "year(时间点)": round(time + dt, 4),
            "strategy(策略名)": strategy_name,
            "W(胜率)": float(W),
            "attend(上座人数)": float(attendance),
            "P(人气值)": float(pop),
            "V(估值(USD))": float(V),
            "D(债务量)": float(debt),
            "C(现金量)": float(cash),
        })

        time += dt

    df_trace = pd.DataFrame(rows)
    df_trace.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 策略轨迹表已导出：{out_csv}")
    print(df_trace.head(8))
    return df_trace

trace_df = export_policy_trace(x_opt, strategy_name="Optimal_Control", out_csv="p1_policy_trace.csv")

# =========================
# 4)（可选）保留你原来的季度打印
# =========================
pop = 0.8
debt = 0.0
cash = 0.0
time = 0.0

for quarter in range(0, N, 2):
    f = x_opt[3*quarter]
    p = x_opt[3*quarter+1]
    u = x_opt[3*quarter+2]

    W = win_rate_from_spending(f)
    cap = salary_cap(time)
    attendance = compute_attendance(pop, p)

    revenue = (
        attendance * baseline_price * p * 20/6
        + attendance * extra_per_attendee * 20/6
        + (2.0e6 if time < 1 else (2.0e6*0.9 if time < 2 else 2.0e6*1.2))/6
        + (1.0e6*(0.5+0.5*pop))/6
    )
    cost = (f * cap + overhead)/6 + debt * interest_rate(time)/6
    profit = revenue - cost

    cash += profit
    if cash < 0:
        deficit = -cash
        debt += u * deficit
        cash = 0.0
    else:
        if debt > 0:
            repay = min(debt, cash)
            debt -= repay
            cash -= repay
        if cash > 0:
            cash=cash

    pop += (k_W * (W - pop) + k_A * (attendance/capacity - pop)) * (dt*2)
    pop = max(0.0, min(1.0, pop))

    print(f"Year {time+dt:.1f}: f={f:.2f}, W={W:.2f}, p={p:.2f}, attend={attendance:.0f}, Pop={pop:.3f}, Debt={debt:.1f}")
    time += dt*2
