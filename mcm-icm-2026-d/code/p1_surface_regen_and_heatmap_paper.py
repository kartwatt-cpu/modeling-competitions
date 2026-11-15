# -*- coding: utf-8 -*-
"""
Regenerate surface data where debt ratio u truly affects valuation V(T),
then produce paper-ready sensitivity heatmaps (representative figures only).

Inputs (optional, will auto-detect if present):
- 模型参数数据：model_parameters.csv
- 球队估值与运营数据：team_valuations.csv
- 球队上座率数据：attendance_data.csv

Core output:
- surface_data_with_leverage_effect.csv
- figures_sensitivity_paper/Fig_HeatmapA_Stotal_multi_u.png

Run:
    python p1_surface_regen_and_heatmap_paper.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Paths (auto-detect)
# =========================
CAND_PARAM = ["模型参数数据：model_parameters.csv", "model_parameters.csv"]
CAND_VAL   = ["球队估值与运营数据：team_valuations.csv", "team_valuations.csv"]
CAND_ATT   = ["球队上座率数据：attendance_data.csv", "attendance_data.csv"]

TEAM_NAME = "Las Vegas Aces"

OUT_DIR = "figures_sensitivity_paper"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SURFACE = "surface_data_with_leverage_effect.csv"

# =========================
# 1) Load helpers
# =========================
def find_file(cands):
    for f in cands:
        if os.path.exists(f):
            return f
        # also try /mnt/data (for sandbox)
        p = os.path.join("/mnt/data", f)
        if os.path.exists(p):
            return p
    return None

def load_params():
    fp = find_file(CAND_PARAM)
    params = {}
    if fp is None:
        return params
    df = pd.read_csv(fp)
    # expected cols: symbol, suggested_value
    if "symbol" in df.columns and "suggested_value" in df.columns:
        for _, r in df.iterrows():
            params[str(r["symbol"]).strip()] = float(r["suggested_value"])
    return params

def getp(params, key, default):
    return float(params.get(key, default))

def load_aces_baseline():
    # valuation/revenue/capacity
    fpv = find_file(CAND_VAL)
    fpa = find_file(CAND_ATT)

    base = {
        "revenue_2024": 22e6,  # fallback
        "capacity": 8600,
        "avg_att_2024": 8500
    }

    if fpv:
        dv = pd.read_csv(fpv)
        row = dv[dv["team"] == TEAM_NAME].iloc[0]
        base["revenue_2024"] = float(row["revenue_2024_M"]) * 1e6
        base["capacity"] = float(row["arena_capacity"])

    if fpa:
        da = pd.read_csv(fpa)
        row = da[da["team"] == TEAM_NAME].iloc[0]
        base["avg_att_2024"] = float(row["avg_attendance_2024"])
        # attendance file也有capacity时，用更可信的那个
        if "arena_capacity" in da.columns:
            base["capacity"] = float(row["arena_capacity"])

    # clamp attendance to capacity (你设定无多场馆决策)
    base["avg_att_2024"] = min(base["avg_att_2024"], base["capacity"])
    return base

# =========================
# 2) Model (minimal leverage channel)
# =========================
def simulate_policy(f, p, u, params, base, T=3.0, dt=0.05):
    """
    Stylized continuous-time simulation (Euler discretization).
    Valuation V(T) = discounted profit integral (your definition).

    Key: u affects valuation via:
    - interest cost on debt
    - terminal principal penalty (risk haircut)
    - liquidity distress penalty when cash < 0 (optional but recommended)
    """
    # ---- parameters from table ----
    r = getp(params, "r", 0.09)            # discount rate
    eta = getp(params, "eta", -1.0)        # price elasticity (negative)
    beta_win = getp(params, "beta_win", 0.08)  # per 10% WP effect

    S_cap = getp(params, "S_cap", 1507100.0)   # salary cap

    # ---- leverage mechanics (you can tune, but keep stable) ----
    i_debt = 0.08        # annual interest rate (can be sensitivity-tested later)
    gamma  = 0.18        # debt amortization speed (annualized)
    phi_T  = 1.00        # terminal principal penalty multiplier (must have, otherwise "free money")
    k_distress = 6.0     # distress penalty strength when cash < 0

    # ---- business baseline ----
    capacity = float(base["capacity"])
    att0 = float(base["avg_att_2024"])
    rev0 = float(base["revenue_2024"])
    home_games = getp(params, "N_games", 44.0) / 2.0  # home games

    # calibrate base ticket price so that ticket revenue is a reasonable share of total revenue
    # share ~ 40% is a safe placeholder; adjust if你有更准确分解
    ticket_share = 0.40
    ticket_rev0 = rev0 * ticket_share
    price0 = ticket_rev0 / max(att0 * home_games, 1e-9)  # baseline ticket price

    other_rev0 = max(rev0 - ticket_rev0, 0.0)

    # ---- states ----
    W0 = 0.65
    P0 = 1.00
    W = W0
    P = P0
    C = 2.0e6            # initial cash buffer (order-of-magnitude)
    D = 0.0              # initial debt

    # popularity dynamics (simple + stable)
    a_pop = 0.55
    d_pop = 0.85

    # win-rate mapping from salary ratio (bounded)
    w_gain = 0.55

    pv = 0.0
    cash_min = C

    steps = int(np.ceil(T / dt))
    for k in range(steps):
        t = k * dt

        # effective win rate from salary ratio
        W = np.clip(W0 + w_gain * (f - 0.90), 0.35, 0.85)

        # popularity ODE
        dPdt = a_pop * (W - W0) - d_pop * (P - P0)
        P = float(np.clip(P + dt * dPdt, 0.60, 1.60))

        # demand/attendance: depends on popularity and price elasticity, capped by capacity
        demand = att0 * (P ** 0.60) * (p ** eta) * (1.0 + 0.30 * (W - W0))
        attend = float(np.clip(demand, 0.0, capacity))

        # revenue rates (annualized)
        ticket_rev = price0 * p * attend * home_games
        # win-driven uplift (per 10% WP): factor = 1 + beta_win*(ΔW/0.1)
        win_uplift = 1.0 + beta_win * ((W - W0) / 0.10)
        win_uplift = float(np.clip(win_uplift, 0.85, 1.25))
        other_rev = other_rev0 * (P ** 0.35) * win_uplift
        R = ticket_rev + other_rev

        # costs (annualized)
        payroll = f * S_cap
        fixed_cost = 6.0e6   # fixed operating cost (kept constant by your assumption)
        interest = i_debt * D

        profit = R - payroll - fixed_cost - interest  # this is what you discount

        # ---- liquidity & financing rule ----
        # Project cash without new borrowing
        C_next = C + profit * dt

        # If cash goes negative, borrow to partially/fully cover deficit according to u
        if C_next < 0:
            deficit = -C_next
            borrow = u * deficit
            D = D + borrow
            C_next = C_next + borrow  # borrowing restores cash
        else:
            borrow = 0.0

        # distress penalty if still negative (u too small)
        if C_next < 0:
            profit = profit - k_distress * (-C_next) / max(dt, 1e-9)

        # debt amortization (principal repayment proxy)
        D = max(D * (1.0 - gamma * dt), 0.0)

        # update cash
        C = C_next
        cash_min = min(cash_min, C)

        # discounted PV accumulation
        pv += np.exp(-r * t) * profit * dt

    # terminal principal penalty (otherwise leverage becomes unrealistically attractive)
    V = pv - phi_T * D
    return V, D, cash_min


# =========================
# 3) Generate surface data
# =========================
def generate_surface(params, base):
    # You can reuse your grid; below是论文友好的中等密度
    f_grid = np.round(np.linspace(0.75, 1.00, 6), 2)  # 0.75..1.00
    p_grid = np.round(np.linspace(0.50, 2.00, 7), 2)  # 0.50..2.00
    u_grid = [0.0, 0.6, 1.0]  # representative policies only

    rows = []
    for u in u_grid:
        for f in f_grid:
            for p in p_grid:
                V, D_T, C_min = simulate_policy(f, p, u, params, base, T=3.0, dt=0.05)
                rows.append({
                    "f": f, "p": p, "u": u,
                    "Valuation": V,
                    "Debt_T": D_T,
                    "Cash_min": C_min
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_SURFACE, index=False, encoding="utf-8-sig")
    print(f"[OK] Surface regenerated: {OUT_SURFACE}")
    return df

# =========================
# 4) Paper-ready heatmap (representative, 1 figure)
# =========================
def plot_stotal_multi_u(df):
    # build grids and compute Stotal per u
    def build_grid(df_u):
        pv = df_u.pivot_table(index="f", columns="p", values="Valuation", aggfunc="mean").sort_index().sort_index(axis=1)
        fvals = pv.index.to_numpy()
        pvals = pv.columns.to_numpy()
        Vmat = pv.to_numpy()
        dV_df, dV_dp = np.gradient(Vmat, fvals, pvals, edge_order=2)
        denom = np.maximum(np.abs(Vmat), 1e-9)
        F = fvals.reshape(-1, 1)
        P = pvals.reshape(1, -1)
        Sf = np.abs(dV_df) * F / denom
        Sp = np.abs(dV_dp) * P / denom
        St = np.sqrt(Sf**2 + Sp**2)
        return fvals, pvals, St

    u_list = sorted(df["u"].unique().tolist())
    mats = []
    grids = []
    for u in u_list:
        fvals, pvals, St = build_grid(df[df["u"] == u])
        mats.append(St)
        grids.append((fvals, pvals))

    vmin = min(np.min(m) for m in mats)
    vmax = max(np.max(m) for m in mats)

    fig, axes = plt.subplots(1, len(u_list), figsize=(11.0, 3.6), constrained_layout=True)
    if len(u_list) == 1:
        axes = [axes]

    for ax, u, St, (fvals, pvals) in zip(axes, u_list, mats, grids):
        extent = [pvals.min(), pvals.max(), fvals.min(), fvals.max()]
        im = ax.imshow(St, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(f"u = {u:g}")
        ax.set_xlabel("Ticket Multiplier  p")
        ax.set_ylabel("Salary Ratio  f")
        # guide band for f in [0.9,1.0]
        ax.axhline(0.90, linewidth=1)
        ax.axhline(1.00, linewidth=1)
        # guide for p≈2.0
        ax.axvline(2.00, linewidth=1)

    cbar = fig.colorbar(im, ax=axes, shrink=0.90, pad=0.02)
    cbar.set_label(r"Total Sensitivity  $S_{total}$ (relative)")
    fig.suptitle(r"Local Sensitivity Heatmaps of Valuation  $S_{total}(f,p)$  under Different Debt Policies", y=1.03)

    out = os.path.join(OUT_DIR, "Fig_HeatmapA_Stotal_multi_u.png")
    fig.savefig(out, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Heatmap saved: {out}")

# =========================
# 5) Main
# =========================
def main():
    params = load_params()
    base = load_aces_baseline()

    df = generate_surface(params, base)

    # sanity check: u-effect should NOT be zero anymore
    pv = df.pivot_table(index=["f","p"], columns="u", values="Valuation", aggfunc="mean")
    diff_span = pv.max(axis=1) - pv.min(axis=1)
    print("\n[CHECK] u-effect (diff_span) summary:")
    print(diff_span.describe())

    plot_stotal_multi_u(df)

if __name__ == "__main__":
    main()
