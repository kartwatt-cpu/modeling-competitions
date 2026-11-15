# -*- coding: utf-8 -*-
"""
Full Sensitivity Analysis Pipeline for 2026 ICM Problem D
- Model I: discount rate r & horizon T sensitivity -> valuation & cashflow
- Model II: alpha (mixed utility weight) & lambda (risk aversion) -> total utility, salary total (proxy), starter/bench roster shifts
- Chapter 6: ticket demand elasticity & injury probability -> profit, retention proxy, injury response cost & performance

Input:
  - ICM_D_WNBA_ThreeFiles_Merged.xlsx
    sheets: 01_ASSUMPTIONS, 30_MASTER_PLAYER_GAME

Outputs:
  - outputs_sensitivity/
      model1_results.csv, model1_cashflow_examples.csv, figures...
      model2_results.csv, model2_rosters/..., figures...
      ch6_results.csv, figures...

Notes:
  1) Salary per player is not directly present in the merged dataset; we build a reproducible salary proxy from minutes share.
  2) Commercial value proxy is derived from (team attendance context * player usage).
  3) Retention is a proxy computed from attendance trend + price + injury shocks (configurable).
"""

import os
import math
import json
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================
# 0) User-configurable setup
# =========================
XLSX_PATH = "ICM_D_WNBA_ThreeFiles_Merged.xlsx"
CACHE_DIR = "cache_parquet"
OUT_DIR = "outputs_sensitivity"

# Team focus (you can change if the string in your dataset differs)
TEAM_QUERY = "Las Vegas Aces"     # exact match if exists; otherwise fuzzy contains match
TEAM_FALLBACK_CONTAINS = "Aces"   # fallback keyword

# Evaluation season: auto-detect latest season with non-missing attendance (recommended)
EVAL_SEASON = None  # set to an int like 2024 to force

# WNBA season structure (used for annualizing revenue/profit)
DEFAULT_HOME_GAMES = 20  # if cannot infer reliably from data

# ---------- Model I sensitivity grids ----------
R_GRID = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]  # discount rate
T_GRID = list(range(2, 11))                    # horizon years: 2..10
DT = 0.20                                      # years per step (per your paper)

# Strategy controls (constant-control scenarios; can be replaced by time-varying controls)
MODEL1_STRATEGIES = {
    "aggressive": {"f": 1.20, "p": 1.15, "u": 0.60},
    "conservative": {"f": 0.90, "p": 1.00, "u": 0.00},
}

# ---------- Model II sensitivity grids ----------
ALPHA_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
LAMBDA_GRID = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
BASELINE_ALPHA = 0.70
BASELINE_LAMBDA = 0.30

# Roster structure: starters + bench fixed ratio (2G-2F-1C) each
ROSTER_CFG = {"G": 2, "F": 2, "C": 1}
MIN_GAMES_PLAYER_POOL = 10
MIN_MINUTES_PLAYER_POOL = 200
TOPK_PER_POS_POOL = 80  # reduce candidate set for speed/stability

# ---------- Chapter 6 sensitivity grids ----------
ELASTICITY_GRID = [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0]  # ticket demand elasticity epsilon
INJURY_PROB_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Chapter 6 price policy (baseline multiplier). You can link it to Model I strategy if you want.
CH6_PRICE_MULTIPLIER = 1.10

# Injury response plans for Chapter 6 (cost vs performance recovery)
CH6_PLANS = {
    "none":        {"annual_cost": 0.0,     "perf_recover": 0.00, "attendance_recover": 0.00},
    "rehab":       {"annual_cost": 250000., "perf_recover": 0.25, "attendance_recover": 0.10},
    "replacement": {"annual_cost": 450000., "perf_recover": 0.45, "attendance_recover": 0.18},
}
RETENTION_TARGET = 0.70  # optional constraint when picking “best plan region”


# =========================
# 1) Utilities
# =========================
def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "model2_rosters"), exist_ok=True)


def minmax_norm(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min() + eps)


def safe_contains(series: pd.Series, keyword: str) -> pd.Series:
    return series.astype(str).str.contains(keyword, case=False, na=False)


def pick_team_name(game_df: pd.DataFrame) -> str:
    """
    Try exact TEAM_QUERY match; else fuzzy contains TEAM_FALLBACK_CONTAINS; else raise.
    """
    teams = game_df["team"].dropna().astype(str).unique().tolist()
    if TEAM_QUERY in teams:
        return TEAM_QUERY
    # fuzzy
    hits = [t for t in teams if TEAM_FALLBACK_CONTAINS.lower() in t.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # pick the most similar (longest common / shortest length difference)
        hits_sorted = sorted(hits, key=lambda s: (abs(len(s) - len(TEAM_QUERY)), -len(s)))
        return hits_sorted[0]
    raise ValueError(f"Cannot find team by TEAM_QUERY='{TEAM_QUERY}' or contains='{TEAM_FALLBACK_CONTAINS}'. "
                     f"Available sample teams (first 20): {teams[:20]}")


def infer_eval_season(master_df: pd.DataFrame) -> int:
    """
    Latest season with non-missing attendance at game level.
    """
    g = master_df.drop_duplicates(["game_id", "team"])[["season", "attendance"]].copy()
    g = g[pd.notna(g["attendance"])]
    if g.empty:
        # fallback: latest season in data
        return int(master_df["season"].dropna().max())
    return int(g["season"].max())


# =========================
# 2) Data loading (with caching)
# =========================
def load_assumptions(xlsx_path: str) -> Dict[str, float]:
    df = pd.read_excel(xlsx_path, sheet_name="01_ASSUMPTIONS", engine="openpyxl")
    # expected columns: assumption, value
    out = {}
    for _, r in df.iterrows():
        k = str(r["assumption"]).strip()
        out[k] = float(r["value"])
    return out


def load_master_cached(xlsx_path: str) -> pd.DataFrame:
    """
    Cache to parquet for fast re-runs. Read minimal columns used across analyses.
    """
    parquet_path = os.path.join(CACHE_DIR, "master_min.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    # Minimal columns used for all 3 parts
    usecols = [
        "game_id", "season", "date", "team", "team_abbr", "opponent", "opp_abbr",
        "team_winner", "attendance", "ticket_revenue_proxy", "gameday_revenue_proxy",
        "player_id", "player", "pos", "starter", "minutes",
        "points", "rebounds", "assists", "steals", "blocks", "turnovers",
        "plus_minus", "did_not_play", "active"
    ]

    df = pd.read_excel(
        xlsx_path, sheet_name="30_MASTER_PLAYER_GAME",
        usecols=lambda c: c in usecols,  # robust to extra columns
        engine="openpyxl"
    )

    # types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # starter might be bool-ish
    if "starter" in df.columns:
        df["starter"] = df["starter"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    # did_not_play might be 0/1
    if "did_not_play" in df.columns:
        df["did_not_play"] = pd.to_numeric(df["did_not_play"], errors="coerce").fillna(0).astype(int)
    if "active" in df.columns:
        # active could be boolean or 0/1
        df["active"] = df["active"].astype(str).str.lower().isin(["1", "true", "yes", "y"])

    df.to_parquet(parquet_path, index=False)
    return df


def make_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert player-game rows to team-game rows (unique game_id x team).
    """
    gcols = [
        "game_id", "season", "date", "team", "team_abbr", "opponent", "opp_abbr",
        "team_winner", "attendance", "ticket_revenue_proxy", "gameday_revenue_proxy",
    ]
    g = df[gcols].drop_duplicates(["game_id", "team"]).copy()
    return g


def infer_home_games(team_games: pd.DataFrame) -> int:
    """
    Data may not reliably contain home/away per game at team-game level.
    If cannot infer, fallback DEFAULT_HOME_GAMES.
    """
    # If half schedule approx, use total games / 2
    n_games = team_games["game_id"].nunique()
    if n_games >= 30:
        return max(10, int(round(n_games / 2)))
    return DEFAULT_HOME_GAMES


# =========================
# 3) Model I simulation (ODE)
# =========================
@dataclass
class Model1Params:
    alpha_f: float
    delta_w: float
    alpha_w: float
    gamma_p: float
    eta_p: float
    kappa: float
    Cap: float
    Cost0: float
    mu: float
    eta_val: float
    W0: float
    P0: float
    C0: float
    D0: float


def calibrate_model1_params(
    team_games: pd.DataFrame,
    assumptions: Dict[str, float],
    cap_salary: float = 1_500_000.0,
    fixed_cost: float = 2_000_000.0,
    mu: float = 5.0,
) -> Model1Params:
    """
    Calibrate a consistent baseline using team attendance + win-rate proxies.
    - W0: team win rate in eval season
    - P0: popularity index from attendance / capacity
    - kappa: revenue coefficient so that R(P0, p=1) matches observed annual revenue proxy
    Other ODE coefficients are set to satisfy steady-state consistency at baseline.
    """

    # win rate proxy: fraction of team-game rows where team_winner==team
    tg = team_games.copy()
    tg = tg[pd.notna(tg["team_winner"])]
    if tg.empty:
        W0 = 0.55
    else:
        W0 = float((tg["team_winner"].astype(str) == tg["team"].astype(str)).mean())
        W0 = float(np.clip(W0, 0.30, 0.80))

    # popularity from attendance (use capacity proxy as max attendance within season/team)
    att = tg["attendance"].dropna().astype(float)
    if len(att) < 10:
        P0 = 0.60
        base_att = 8000.0
        cap_att = 12000.0
    else:
        base_att = float(att.mean())
        cap_att = float(att.max())
        P0 = float(np.clip(base_att / (cap_att + 1e-9), 0.20, 0.95))

    # annual revenue proxy
    # gameday_revenue_proxy is ticket + concessions + merch (by construction in your assumptions sheet)
    rev_game = tg["gameday_revenue_proxy"].dropna().astype(float)
    if rev_game.empty:
        # rebuild from assumptions if needed
        avg_ticket = assumptions.get("avg_ticket_price", 45.0)
        per_att = assumptions.get("concessions_per_attendee", 18.0) + assumptions.get("merch_per_attendee", 7.0)
        rev_game_mean = base_att * (avg_ticket + per_att)
    else:
        rev_game_mean = float(rev_game.mean())

    home_games = infer_home_games(tg)
    annual_revenue_proxy = rev_game_mean * home_games

    # kappa so that R(P0,1)=kappa*P0 matches annual_revenue_proxy
    kappa = annual_revenue_proxy / (P0 + 1e-9)

    # ODE coefficients
    # Win dynamics: choose delta_w and alpha_f to match steady-state W0 given baseline f=1
    delta_w = 1.0
    alpha_f = (W0 * delta_w) / (1.0 * (1.0 - W0) + 1e-9)

    # Popularity dynamics: choose gamma_p and alpha_w s.t. steady-state P0=alpha_w*W0/gamma_p
    gamma_p = 1.0
    alpha_w = (P0 * gamma_p) / (W0 + 1e-9)

    # price drag on popularity (eta_p): keep moderate; you can override if you have a fitted elasticity mapping
    eta_p = 0.20

    # brand/terminal valuation scaling
    eta_val = mu * annual_revenue_proxy  # so eta_val*P is comparable to mu*R

    return Model1Params(
        alpha_f=alpha_f, delta_w=delta_w,
        alpha_w=alpha_w, gamma_p=gamma_p, eta_p=eta_p,
        kappa=kappa, Cap=cap_salary, Cost0=fixed_cost,
        mu=mu, eta_val=eta_val,
        W0=W0, P0=P0, C0=0.0, D0=0.0
    )


def simulate_model1(params: Model1Params, controls: Dict[str, float], r: float, T: float, dt: float) -> Dict[str, np.ndarray]:
    """
    Forward Euler simulation.
    States: W, P, C, D
    Revenue: R = kappa * P * p
    Profit cashflow: pi = R - f*Cap - Cost0
    Objective PV: sum exp(-r t) * pi * dt
    Terminal valuation: V(T)=mu*R(T) + eta_val*P(T)
    """
    n = int(math.ceil(T / dt)) + 1
    t = np.linspace(0, T, n)

    W = np.zeros(n); P = np.zeros(n); C = np.zeros(n); D = np.zeros(n)
    Rv = np.zeros(n); profit = np.zeros(n); pv_profit = np.zeros(n)

    W[0] = params.W0
    P[0] = params.P0
    C[0] = params.C0
    D[0] = params.D0

    f = float(controls["f"])
    p = float(controls["p"])
    u = float(controls["u"])

    for k in range(n - 1):
        # revenue and operating profit
        Rv[k] = params.kappa * P[k] * p
        profit[k] = Rv[k] - f * params.Cap - params.Cost0
        pv_profit[k] = math.exp(-r * t[k]) * profit[k] * dt

        # ODEs
        dW = params.alpha_f * f * (1.0 - W[k]) - params.delta_w * W[k]
        dP = params.alpha_w * W[k] - params.gamma_p * P[k] - params.eta_p * (p - 1.0)
        dC = Rv[k] - f * params.Cap - params.Cost0 + u * f * params.Cap
        dD = u * f * params.Cap

        W[k + 1] = max(0.0, min(1.0, W[k] + dt * dW))
        P[k + 1] = max(0.0, P[k] + dt * dP)
        C[k + 1] = C[k] + dt * dC
        D[k + 1] = max(0.0, D[k] + dt * dD)

    # last step revenue/profit
    Rv[-1] = params.kappa * P[-1] * p
    profit[-1] = Rv[-1] - f * params.Cap - params.Cost0
    pv_profit[-1] = math.exp(-r * t[-1]) * profit[-1] * dt

    PV = float(pv_profit.sum())
    V_terminal = float(params.mu * Rv[-1] + params.eta_val * P[-1])
    # "long-term valuation": PV of operating profit + discounted terminal proxy
    long_term_value = PV + math.exp(-r * T) * V_terminal

    return {
        "t": t,
        "W": W, "P": P, "C": C, "D": D,
        "R": Rv, "profit": profit,
        "PV_profit": PV,
        "V_terminal": V_terminal,
        "long_term_value": long_term_value
    }


def run_model1_sensitivity(team_games: pd.DataFrame, assumptions: Dict[str, float]):
    params = calibrate_model1_params(team_games, assumptions)
    rows = []
    cashflow_examples = []

    for strat_name, ctrl in MODEL1_STRATEGIES.items():
        for r, T in itertools.product(R_GRID, T_GRID):
            sim = simulate_model1(params, ctrl, r=r, T=float(T), dt=DT)
            rows.append({
                "strategy": strat_name,
                "r": r,
                "T": T,
                "PV_profit": sim["PV_profit"],
                "V_terminal": sim["V_terminal"],
                "long_term_value": sim["long_term_value"],
                "profit_mean": float(np.mean(sim["profit"])),
                "profit_std": float(np.std(sim["profit"])),
                "P_end": float(sim["P"][-1]),
                "W_end": float(sim["W"][-1]),
            })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "model1_results.csv"), index=False)

    # pick a few representative cashflow curves to save (baseline + extremes)
    pick_points = [
        ("baseline", MODEL1_STRATEGIES["aggressive"], 0.10, 5),
        ("low_r_long_T", MODEL1_STRATEGIES["aggressive"], min(R_GRID), max(T_GRID)),
        ("high_r_long_T", MODEL1_STRATEGIES["aggressive"], max(R_GRID), max(T_GRID)),
        ("baseline_cons", MODEL1_STRATEGIES["conservative"], 0.10, 5),
    ]
    for tag, ctrl, r, T in pick_points:
        sim = simulate_model1(params, ctrl, r=r, T=float(T), dt=DT)
        df_cf = pd.DataFrame({"t": sim["t"], "profit": sim["profit"], "R": sim["R"], "P": sim["P"], "W": sim["W"]})
        df_cf["tag"] = tag
        df_cf["r"] = r
        df_cf["T"] = T
        df_cf["strategy"] = "aggressive" if ctrl == MODEL1_STRATEGIES["aggressive"] else "conservative"
        cashflow_examples.append(df_cf)
    cashflow_examples = pd.concat(cashflow_examples, ignore_index=True)
    cashflow_examples.to_csv(os.path.join(OUT_DIR, "model1_cashflow_examples.csv"), index=False)

    # ---- figures ----
    # 3D surface long_term_value for each strategy
    for strat_name in MODEL1_STRATEGIES.keys():
        sub = res[res["strategy"] == strat_name].copy()
        pivot = sub.pivot(index="T", columns="r", values="long_term_value").sort_index()
        Ts = pivot.index.values
        rs = pivot.columns.values
        Rm, Tm = np.meshgrid(rs, Ts)
        Z = pivot.values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Rm, Tm, Z, linewidth=0, antialiased=True)
        ax.set_xlabel("Discount rate r")
        ax.set_ylabel("Horizon T (years)")
        ax.set_zlabel("Long-term valuation")
        ax.set_title(f"Model I: Valuation Surface ({strat_name})")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "figures", f"model1_surface_{strat_name}.png"), dpi=220)
        plt.close(fig)

        # heatmap (2D)
        fig = plt.figure()
        plt.imshow(Z, aspect="auto", origin="lower",
                   extent=[rs.min(), rs.max(), Ts.min(), Ts.max()])
        plt.colorbar(label="Long-term valuation")
        plt.xlabel("Discount rate r")
        plt.ylabel("Horizon T (years)")
        plt.title(f"Model I: Heatmap of Valuation ({strat_name})")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "figures", f"model1_heatmap_{strat_name}.png"), dpi=220)
        plt.close(fig)

    # cashflow comparison plot
    fig = plt.figure()
    for tag in cashflow_examples["tag"].unique():
        sub = cashflow_examples[cashflow_examples["tag"] == tag]
        # plot profit over time (each tag is one curve)
        plt.plot(sub["t"], sub["profit"], label=f"{tag}")
    plt.xlabel("t (years)")
    plt.ylabel("Operating profit cashflow (per year units)")
    plt.title("Model I: Cashflow examples under selected (r,T) cases")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "model1_cashflow_examples.png"), dpi=220)
    plt.close(fig)

    return res


# =========================
# 4) Model II sensitivity (roster selection)
# =========================
def build_player_pool(master_df: pd.DataFrame, season: int, assumptions: Dict[str, float], team_name: str) -> pd.DataFrame:
    """
    Aggregate player metrics within evaluation window (season).
    Outputs per-player:
      - performance_score (competitive)
      - commercial_score (attendance context * usage)
      - injury_prob (DNP rate)
      - salary_proxy (minutes-based)
      - position (G/F/C mapped)
    """
    df = master_df[master_df["season"].astype("Int64") == season].copy()
    # keep only rows with valid player_id & minutes
    df = df[pd.notna(df["player_id"]) & pd.notna(df["player"])]

    # minutes numeric
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    # position mapping: keep only G/F/C
    df["pos"] = df["pos"].astype(str).str.upper().str.strip()
    df["pos_simple"] = df["pos"].str.extract(r'([GFC])', expand=False)
    df = df[pd.notna(df["pos_simple"])]

    # game participation: minutes > 0 and active==True and not DNP
    df["played"] = (df["minutes"] > 0) & (df["did_not_play"].fillna(0).astype(int) == 0) & (df["active"].fillna(False))

    # aggregate to player level
    agg = df.groupby(["player_id", "player", "pos_simple"], as_index=False).agg(
        games_total=("game_id", "nunique"),
        games_played=("played", "sum"),
        minutes_total=("minutes", "sum"),
        points=("points", "sum"),
        rebounds=("rebounds", "sum"),
        assists=("assists", "sum"),
        steals=("steals", "sum"),
        blocks=("blocks", "sum"),
        turnovers=("turnovers", "sum"),
        plus_minus=("plus_minus", "sum"),
    )

    # injury / unavailability probability proxy: DNP rate among games_total
    dnp = df.groupby(["player_id"], as_index=False).agg(
        dnp_count=("did_not_play", "sum"),
        active_rate=("active", "mean"),
    )
    agg = agg.merge(dnp, on="player_id", how="left")
    agg["injury_prob"] = (agg["dnp_count"].fillna(0.0) / (agg["games_total"].replace(0, np.nan))).fillna(0.0)
    agg["injury_prob"] = agg["injury_prob"].clip(0, 1)

    # performance efficiency score (finance-style “return” proxy)
    # You can change weights to match your paper; this is stable and monotonic.
    # Scale by minutes to get efficiency.
    eps = 1e-9
    agg["perf_raw"] = (
        agg["points"].fillna(0) +
        1.2 * agg["rebounds"].fillna(0) +
        1.5 * agg["assists"].fillna(0) +
        3.0 * agg["steals"].fillna(0) +
        3.0 * agg["blocks"].fillna(0) -
        2.0 * agg["turnovers"].fillna(0)
    ) / (agg["minutes_total"].fillna(0) + eps)

    # commercial score proxy: (team attendance context) * (player usage)
    # usage ~ minutes_total / games_played
    game_level = make_game_level(df)
    # team mean attendance (within season) as context
    team_att = game_level.groupby("team", as_index=False)["attendance"].mean().rename(columns={"attendance": "team_att_mean"})
    # join player's primary team context using max games played team
    player_team = df[df["played"]].groupby(["player_id", "team"], as_index=False)["game_id"].nunique()
    player_team = player_team.sort_values(["player_id", "game_id"], ascending=[True, False]).drop_duplicates(["player_id"])
    player_team = player_team.rename(columns={"team": "primary_team", "game_id": "games_on_primary_team"})
    agg = agg.merge(player_team[["player_id", "primary_team"]], on="player_id", how="left")
    agg = agg.merge(team_att.rename(columns={"team": "primary_team"}), on="primary_team", how="left")

    agg["minutes_per_game"] = agg["minutes_total"] / (agg["games_played"].replace(0, np.nan))
    agg["minutes_per_game"] = agg["minutes_per_game"].fillna(0.0)

    # commercial raw: higher if (context attendance high) and (player usage high) and (scoring appealing)
    agg["comm_raw"] = (
        agg["team_att_mean"].fillna(0.0) * (agg["minutes_per_game"].fillna(0.0) / 30.0) +
        300.0 * (agg["points"].fillna(0.0) / (agg["games_played"].replace(0, np.nan))).fillna(0.0)
    )

    # normalization
    agg["P_hat"] = minmax_norm(agg["perf_raw"])
    agg["B_hat"] = minmax_norm(agg["comm_raw"])

    # salary proxy: scale league-average salary by minutes share (reproducible; uses your assumptions sheet)
    base_salary = float(assumptions.get("avg_wnba_salary_league", 75000.0))
    # minutes scale: 0.6..1.8 multiplier (clipped)
    minutes_scale = (agg["minutes_per_game"] / 22.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minutes_scale = minutes_scale.clip(0.6, 1.8)
    agg["salary_proxy"] = base_salary * minutes_scale

    # incumbent flag (Aces players) for reporting only
    agg["is_aces_context"] = agg["primary_team"].astype(str).str.lower().eq(str(team_name).lower())

    return agg


def select_roster_greedy(player_pool: pd.DataFrame, alpha: float, lam: float) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    Greedy selection with fixed (2G-2F-1C) for starters and bench, no duplicates.
    Score = U_eff = (1 - lam * injury_prob) * (alpha*P_hat + (1-alpha)*B_hat)
    Transaction cost penalties intentionally excluded per your request.

    Returns:
      roster: {"starters":[...], "bench":[...]}
      metrics: {total_utility, salary_total, starter_ids, bench_ids, ...}
    """
    df = player_pool.copy()

    # eligibility filters
    df = df[(df["games_total"] >= MIN_GAMES_PLAYER_POOL) & (df["minutes_total"] >= MIN_MINUTES_PLAYER_POOL)].copy()

    # reduce candidate set per position for stability
    out_frames = []
    for pos in ["G", "F", "C"]:
        sub = df[df["pos_simple"] == pos].copy()
        # pre-rank by neutral score (alpha=0.5, lam=0.3)
        neutral = (0.5 * sub["P_hat"] + 0.5 * sub["B_hat"]) * (1 - 0.3 * sub["injury_prob"])
        sub = sub.assign(_neutral=neutral).sort_values("_neutral", ascending=False).head(TOPK_PER_POS_POOL)
        out_frames.append(sub.drop(columns=["_neutral"]))
    df = pd.concat(out_frames, ignore_index=True)

    # compute scores under (alpha, lam)
    U = alpha * df["P_hat"] + (1 - alpha) * df["B_hat"]
    U_eff = (1 - lam * df["injury_prob"]).clip(lower=0.0) * U
    df["score"] = U_eff

    # choose starters by position counts
    starters = []
    used_ids = set()
    for pos, need in ROSTER_CFG.items():
        cand = df[df["pos_simple"] == pos].sort_values("score", ascending=False)
        for _, r in cand.iterrows():
            if len([x for x in starters if x[2] == pos]) >= need:
                break
            pid = r["player_id"]
            if pid in used_ids:
                continue
            starters.append((pid, r["player"], pos))
            used_ids.add(pid)

    # choose bench similarly from remaining
    bench = []
    for pos, need in ROSTER_CFG.items():
        cand = df[df["pos_simple"] == pos].sort_values("score", ascending=False)
        for _, r in cand.iterrows():
            if len([x for x in bench if x[2] == pos]) >= need:
                break
            pid = r["player_id"]
            if pid in used_ids:
                continue
            bench.append((pid, r["player"], pos))
            used_ids.add(pid)

    # metrics
    sel_ids = [x[0] for x in starters + bench]
    sel = df[df["player_id"].isin(sel_ids)].copy()
    total_utility = float(sel["score"].sum())
    salary_total = float(sel["salary_proxy"].sum())

    roster = {
        "starters": [x[1] for x in starters],
        "bench": [x[1] for x in bench],
    }
    metrics = {
        "total_utility": total_utility,
        "salary_total_proxy": salary_total,
        "n_selected": int(len(sel_ids)),
    }
    return roster, metrics


def run_model2_sensitivity(player_pool: pd.DataFrame):
    # baseline roster for measuring changes
    base_roster, base_metrics = select_roster_greedy(player_pool, BASELINE_ALPHA, BASELINE_LAMBDA)
    base_set = set(base_roster["starters"] + base_roster["bench"])
    base_starters_set = set(base_roster["starters"])
    base_bench_set = set(base_roster["bench"])

    rows = []
    roster_dump = []

    for alpha, lam in itertools.product(ALPHA_GRID, LAMBDA_GRID):
        roster, metrics = select_roster_greedy(player_pool, alpha, lam)
        cur_set = set(roster["starters"] + roster["bench"])
        cur_starters_set = set(roster["starters"])
        cur_bench_set = set(roster["bench"])

        # change metrics
        roster_change = len(base_set.symmetric_difference(cur_set))
        starter_change = len(base_starters_set.symmetric_difference(cur_starters_set))
        bench_change = len(base_bench_set.symmetric_difference(cur_bench_set))

        rows.append({
            "alpha": alpha,
            "lambda": lam,
            "total_utility": metrics["total_utility"],
            "salary_total_proxy": metrics["salary_total_proxy"],
            "roster_change_vs_baseline": roster_change,
            "starter_change_vs_baseline": starter_change,
            "bench_change_vs_baseline": bench_change,
        })

        # save roster text for traceability
        roster_dump.append({
            "alpha": alpha, "lambda": lam,
            "starters": roster["starters"],
            "bench": roster["bench"]
        })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "model2_results.csv"), index=False)

    # dump rosters as json lines
    with open(os.path.join(OUT_DIR, "model2_rosters", "rosters.jsonl"), "w", encoding="utf-8") as f:
        for x in roster_dump:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- figures: heatmaps ----
    def heatmap(value_col: str, fname: str, title: str):
        pivot = res.pivot(index="lambda", columns="alpha", values=value_col).sort_index()
        Z = pivot.values
        xs = pivot.columns.values
        ys = pivot.index.values

        fig = plt.figure()
        plt.imshow(Z, aspect="auto", origin="lower",
                   extent=[xs.min(), xs.max(), ys.min(), ys.max()])
        plt.colorbar(label=value_col)
        plt.xlabel("alpha (utility weight)")
        plt.ylabel("lambda (risk aversion)")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "figures", fname), dpi=220)
        plt.close(fig)

    heatmap("total_utility", "model2_heatmap_utility.png", "Model II: Total Utility across (alpha, lambda)")
    heatmap("salary_total_proxy", "model2_heatmap_salary.png", "Model II: Salary Total (Proxy) across (alpha, lambda)")
    heatmap("roster_change_vs_baseline", "model2_heatmap_roster_change.png", "Model II: Roster Change vs Baseline")

    return res, base_roster, base_metrics


# =========================
# 5) Chapter 6 sensitivity (ticket elasticity & injury probability)
# =========================
def compute_retention_proxy(
    base_ret: float,
    win_rate: float,
    win_rate_base: float,
    price_mult: float,
    injury_prob: float,
    beta_win: float = 0.25,
    beta_price: float = 0.08,
    beta_injury: float = 0.30,
) -> float:
    """
    A stable retention proxy:
      retention = clip(base_ret + beta_win*(win_rate-win_base) - beta_price*(p-1) - beta_injury*injury_prob, 0, 1)
    """
    ret = base_ret + beta_win * (win_rate - win_rate_base) - beta_price * (price_mult - 1.0) - beta_injury * injury_prob
    return float(np.clip(ret, 0.0, 1.0))


def run_ch6_sensitivity(
    assumptions: Dict[str, float],
    team_games: pd.DataFrame,
    model1_params: Model1Params,
    price_mult: float = CH6_PRICE_MULTIPLIER
):
    """
    For each (elasticity, injury_prob) and each injury plan:
      - attendance = base_attendance * price_mult^elasticity * injury_factor + recovery_factor(plan)
      - profit = annual_revenue - payroll - fixed_cost - plan_cost
      - retention = proxy from win_rate, price, injury
      - performance = recovered win rate proxy
    """
    avg_ticket = float(assumptions.get("avg_ticket_price", 45.0))
    per_att_extra = float(assumptions.get("concessions_per_attendee", 18.0) + assumptions.get("merch_per_attendee", 7.0))

    tg = team_games.copy()
    att = tg["attendance"].dropna().astype(float)
    if len(att) < 10:
        base_att = 8000.0
        cap_att = 12000.0
    else:
        base_att = float(att.mean())
        cap_att = float(att.max())

    home_games = infer_home_games(tg)

    # base win rate from Model I calibration
    win_base = model1_params.W0
    # base retention proxy from attendance trend across seasons (if possible); else 0.75
    base_ret = 0.75

    # payroll and fixed cost: from Model I params
    payroll = model1_params.Cap * 1.0  # baseline f=1
    fixed_cost = model1_params.Cost0

    rows = []
    for eps, q in itertools.product(ELASTICITY_GRID, INJURY_PROB_GRID):
        # baseline demand adjustment from elasticity
        demand_mult = float((price_mult ** eps) if price_mult > 0 else 1.0)
        # injury reduces attendance and performance
        injury_att_loss = 0.25  # attendance loss factor at q=1
        injury_perf_loss = 0.40 # win rate loss factor at q=1

        for plan_name, plan in CH6_PLANS.items():
            # recovery factors
            att_recover = float(plan["attendance_recover"])
            perf_recover = float(plan["perf_recover"])

            # expected attendance
            att_mult = demand_mult * (1.0 - injury_att_loss * q) * (1.0 + att_recover * q)
            attendance = float(np.clip(base_att * att_mult, 0.0, cap_att * 1.20))  # allow slight over-cap proxy

            # win rate proxy under injury + mitigation
            win = win_base * (1.0 - injury_perf_loss * q) * (1.0 + perf_recover * q)
            win = float(np.clip(win, 0.2, 0.9))

            # annual revenue proxy
            price = avg_ticket * price_mult
            rev_game = attendance * (price + per_att_extra)
            annual_revenue = rev_game * home_games

            # plan cost
            plan_cost = float(plan["annual_cost"])

            profit = annual_revenue - payroll - fixed_cost - plan_cost

            retention = compute_retention_proxy(base_ret, win, win_base, price_mult, q)

            rows.append({
                "elasticity": eps,
                "injury_prob": q,
                "plan": plan_name,
                "attendance": attendance,
                "win_rate_proxy": win,
                "profit": profit,
                "retention_proxy": retention,
                "plan_cost": plan_cost,
                "plan_perf_recover": perf_recover,
            })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "ch6_results.csv"), index=False)

    # ---- Figures ----
    def heatmap_for_plan(value_col: str, plan: str, fname: str, title: str):
        sub = res[res["plan"] == plan].copy()
        pivot = sub.pivot(index="injury_prob", columns="elasticity", values=value_col).sort_index()
        Z = pivot.values
        xs = pivot.columns.values
        ys = pivot.index.values

        fig = plt.figure()
        plt.imshow(Z, aspect="auto", origin="lower",
                   extent=[xs.min(), xs.max(), ys.min(), ys.max()])
        plt.colorbar(label=value_col)
        plt.xlabel("ticket elasticity (epsilon)")
        plt.ylabel("injury probability")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "figures", fname), dpi=220)
        plt.close(fig)

    for plan_name in CH6_PLANS.keys():
        heatmap_for_plan("profit", plan_name,
                         f"ch6_profit_{plan_name}.png",
                         f"Chapter 6: Profit heatmap ({plan_name})")
        heatmap_for_plan("retention_proxy", plan_name,
                         f"ch6_retention_{plan_name}.png",
                         f"Chapter 6: Retention proxy heatmap ({plan_name})")

    # Best plan map: maximize profit subject to retention >= target (else unconstrained best)
    best_rows = []
    for eps, q in itertools.product(ELASTICITY_GRID, INJURY_PROB_GRID):
        sub = res[(res["elasticity"] == eps) & (res["injury_prob"] == q)].copy()
        feasible = sub[sub["retention_proxy"] >= RETENTION_TARGET]
        if not feasible.empty:
            pick = feasible.sort_values("profit", ascending=False).iloc[0]
            constrained = True
        else:
            pick = sub.sort_values("profit", ascending=False).iloc[0]
            constrained = False
        best_rows.append({
            "elasticity": eps, "injury_prob": q,
            "best_plan": pick["plan"],
            "best_profit": pick["profit"],
            "best_retention": pick["retention_proxy"],
            "constrained_used": constrained
        })
    best = pd.DataFrame(best_rows)
    best.to_csv(os.path.join(OUT_DIR, "ch6_best_plan_map.csv"), index=False)

    # Plot best plan as categorical heatmap (encode plans)
    plan_to_code = {p: i for i, p in enumerate(CH6_PLANS.keys())}
    pivot = best.pivot(index="injury_prob", columns="elasticity", values="best_plan").sort_index()
    Z = pivot.replace(plan_to_code).values.astype(float)
    xs = pivot.columns.values
    ys = pivot.index.values

    fig = plt.figure()
    plt.imshow(Z, aspect="auto", origin="lower",
               extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    cbar = plt.colorbar()
    cbar.set_label("best plan code")
    # label ticks
    cbar.set_ticks(list(plan_to_code.values()))
    cbar.set_ticklabels(list(plan_to_code.keys()))
    plt.xlabel("ticket elasticity (epsilon)")
    plt.ylabel("injury probability")
    plt.title(f"Chapter 6: Best plan region (profit-max, retention>= {RETENTION_TARGET})")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "ch6_best_plan_region.png"), dpi=220)
    plt.close(fig)

    return res, best


# =========================
# 6) Main
# =========================
def main():
    ensure_dirs()

    if not os.path.exists(XLSX_PATH):
        raise FileNotFoundError(f"Cannot find '{XLSX_PATH}'. Put it in the same folder as this script or update XLSX_PATH.")

    assumptions = load_assumptions(XLSX_PATH)
    master = load_master_cached(XLSX_PATH)
    game = make_game_level(master)

    # pick evaluation season
    season = int(EVAL_SEASON) if EVAL_SEASON is not None else infer_eval_season(master)
    # filter game-level to season
    game_season = game[game["season"].astype("Int64") == season].copy()
    team_name = pick_team_name(game_season)
    team_games = game_season[game_season["team"].astype(str).str.lower() == str(team_name).lower()].copy()

    if team_games.empty:
        raise ValueError(f"Team games empty for team='{team_name}', season={season}. "
                         f"Try changing TEAM_QUERY / TEAM_FALLBACK_CONTAINS or set EVAL_SEASON manually.")

    print(f"[INFO] Using season={season}, team='{team_name}', team_games={len(team_games)}")

    # ---- Model I ----
    model1_res = run_model1_sensitivity(team_games, assumptions)
    print("[OK] Model I sensitivity done -> outputs_sensitivity/model1_results.csv and figures/")

    # ---- Model II ----
    player_pool = build_player_pool(master, season, assumptions, team_name)
    model2_res, base_roster, base_metrics = run_model2_sensitivity(player_pool)
    print("[OK] Model II sensitivity done -> outputs_sensitivity/model2_results.csv, model2_rosters/")

    # ---- Chapter 6 ----
    m1_params = calibrate_model1_params(team_games, assumptions)
    ch6_res, best_map = run_ch6_sensitivity(assumptions, team_games, m1_params, price_mult=CH6_PRICE_MULTIPLIER)
    print("[OK] Chapter 6 sensitivity done -> outputs_sensitivity/ch6_results.csv, ch6_best_plan_map.csv")

    # summary print
    print("\n========== BASELINE (Model II) ==========")
    print(f"Baseline alpha={BASELINE_ALPHA}, lambda={BASELINE_LAMBDA}")
    print("Starters:", base_roster["starters"])
    print("Bench   :", base_roster["bench"])
    print("Metrics :", base_metrics)

    print("\nAll outputs saved under:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
