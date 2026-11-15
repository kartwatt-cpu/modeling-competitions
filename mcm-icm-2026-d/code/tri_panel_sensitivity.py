# tri_panel_sensitivity.py
# -*- coding: utf-8 -*-
"""
3-panel sensitivity triptych (paper-ready):

(a) Model I: valuation over (r, T)   -> heatmap + contour
(b) Model II: utility over (alpha, lambda) -> heatmap + roster-change contours (optional)
(c) Chapter 6: profit over (epsilon, q) -> heatmap + overlay best plan regions (none/rehab/replacement),
    under retention constraint (retention_proxy >= threshold). Infeasible cells are masked.

This version is adapted to your actual CSV naming/columns:
- Model I: strategy, r, T, long_term_value (and/or valuation synonyms)
- Model II: alpha, lambda, total_utility (+ optional roster_change_vs_baseline)
- Ch6: elasticity, injury_prob, plan, profit, retention_proxy  (long format)

Usage (recommended):
  python tri_panel_sensitivity.py

Or with explicit paths:
  python tri_panel_sensitivity.py --model1_csv ... --model2_csv ... --ch6_csv ... --out ...
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


# -----------------------------
# Generic helpers
# -----------------------------
def _safe_mkdir_for_file(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and (not os.path.exists(folder)):
        os.makedirs(folder, exist_ok=True)


def _assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Tip: pass absolute paths via --model1_csv/--model2_csv/--ch6_csv, "
            f"or place the CSVs next to this script and run without args."
        )


def _pick_col(df: pd.DataFrame, candidates: List[str], must: bool = True) -> Optional[str]:
    """Case-insensitive exact match with candidate list."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    if must:
        raise ValueError(
            f"Missing required column. Tried (case-insensitive): {candidates}\n"
            f"Existing columns: {list(df.columns)}"
        )
    return None


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _to_grid(df: pd.DataFrame, x: str, y: str, z: str, agg: str = "mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a long table into grid using pivot_table (robust to duplicates).
    Rows: y, Cols: x, Values: z.
    """
    df2 = df[[x, y, z]].copy()
    df2 = _ensure_numeric(df2, [x, y, z]).dropna(subset=[x, y, z])

    xvals = np.sort(df2[x].unique())
    yvals = np.sort(df2[y].unique())

    piv = df2.pivot_table(index=y, columns=x, values=z, aggfunc=agg)
    piv = piv.reindex(index=yvals, columns=xvals)

    Z = piv.values.astype(float)
    return xvals, yvals, Z


# -----------------------------
# Panel A: Model I
# -----------------------------
@dataclass
class Model1Data:
    r_col: str
    T_col: str
    V_col: str
    df: pd.DataFrame


def load_model1(csv_path: str) -> Model1Data:
    _assert_exists(csv_path)
    df = pd.read_csv(csv_path)

    r_col = _pick_col(df, ["r", "discount_rate", "rate"])
    T_col = _pick_col(df, ["T", "horizon", "years", "planning_horizon"])
    V_col = _pick_col(df, ["V", "valuation", "long_term_valuation", "long_term_value", "V_T", "V_final"])

    # If multiple strategies exist, select a single strategy deterministically for clean (r,T)->V grid
    strat_col = _pick_col(df, ["strategy", "policy", "scenario"], must=False)
    if strat_col is not None and df[strat_col].nunique() > 1:
        unique_strats = df[strat_col].astype(str).str.lower().unique().tolist()
        pick = "aggressive" if "aggressive" in unique_strats else df[strat_col].iloc[0]
        df = df[df[strat_col].astype(str).str.lower() == str(pick).lower()].copy()
        print(f"[Model I] multiple strategies detected -> using strategy='{pick}'")

    return Model1Data(r_col=r_col, T_col=T_col, V_col=V_col, df=df)


def plot_model1(ax: plt.Axes, m1: Model1Data, title: str) -> None:
    xvals, yvals, Z = _to_grid(m1.df, x=m1.r_col, y=m1.T_col, z=m1.V_col)

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
    )

    # Contours to show slope structure
    try:
        levels = 6
        cs = ax.contour(xvals, yvals, Z, levels=levels, linewidths=0.9, colors="white", alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2g")
    except Exception:
        pass

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Discount rate r")
    ax.set_ylabel("Horizon T (years)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Long-term valuation", rotation=90)


# -----------------------------
# Panel B: Model II
# -----------------------------
@dataclass
class Model2Data:
    a_col: str
    l_col: str
    U_col: str
    R_col: Optional[str]
    df: pd.DataFrame


def load_model2(csv_path: str) -> Model2Data:
    _assert_exists(csv_path)
    df = pd.read_csv(csv_path)

    a_col = _pick_col(df, ["alpha", "a"])
    l_col = _pick_col(df, ["lambda", "risk_aversion", "lam", "lmbd"])
    U_col = _pick_col(df, ["total_utility", "utility", "U", "U_total"])

    R_col = _pick_col(
        df,
        ["roster_change_vs_baseline", "roster_change", "roster_delta", "roster_change_count"],
        must=False
    )

    return Model2Data(a_col=a_col, l_col=l_col, U_col=U_col, R_col=R_col, df=df)


def plot_model2(
    ax: plt.Axes,
    m2: Model2Data,
    alpha0: float,
    lambda0: float,
    title: str
) -> None:
    xvals, yvals, U = _to_grid(m2.df, x=m2.a_col, y=m2.l_col, z=m2.U_col)

    im = ax.imshow(
        U,
        origin="lower",
        aspect="auto",
        extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
    )

    # Optional overlay: roster-change contour lines
    if m2.R_col is not None:
        _, _, R = _to_grid(m2.df, x=m2.a_col, y=m2.l_col, z=m2.R_col)
        # Choose a few informative contour levels (auto clipped)
        raw_levels = np.array([0, 1, 2, 4, 6, 8], dtype=float)
        lo, hi = np.nanmin(R), np.nanmax(R)
        levels = np.unique(np.clip(raw_levels, lo, hi))
        try:
            cs = ax.contour(xvals, yvals, R, levels=levels, linewidths=0.9, colors="black", alpha=0.85)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
        except Exception:
            pass

    # Baseline point marker
    ax.scatter([alpha0], [lambda0], s=32, marker="o", edgecolors="black", facecolors="white", zorder=5)
    ax.annotate("baseline", (alpha0, lambda0), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("alpha (utility weight)")
    ax.set_ylabel("lambda (risk aversion)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Total utility", rotation=90)


# -----------------------------
# Panel C: Chapter 6 (epsilon, q)
# -----------------------------
@dataclass
class Ch6Data:
    e_col: str
    q_col: str
    plan_col: str
    profit_col: str
    ret_col: str
    df: pd.DataFrame


def load_ch6(csv_path: str) -> Ch6Data:
    _assert_exists(csv_path)
    df = pd.read_csv(csv_path)

    e_col = _pick_col(df, ["epsilon", "ticket_elasticity", "elasticity", "eps"])
    q_col = _pick_col(df, ["q", "injury_prob", "injury_probability", "p_injury", "injury_p"])

    plan_col = _pick_col(df, ["plan"])
    profit_col = _pick_col(df, ["profit", "profit_value", "total_profit"])
    ret_col = _pick_col(df, ["retention", "retention_proxy", "retention_rate", "retain"])

    # Normalize plan strings
    df[plan_col] = df[plan_col].astype(str).str.lower().str.strip()
    df = df[df[plan_col].isin(["none", "rehab", "replacement"])].copy()

    return Ch6Data(
        e_col=e_col, q_col=q_col,
        plan_col=plan_col, profit_col=profit_col, ret_col=ret_col,
        df=df
    )


def _ch6_best_plan_grid(ch6: Ch6Data, retention_min: float) -> pd.DataFrame:
    """
    From long-format df: (epsilon, q, plan, profit, retention_proxy)
    compute best feasible plan and best profit per (epsilon, q) under retention constraint.
    """
    df = ch6.df.copy()
    df = _ensure_numeric(df, [ch6.e_col, ch6.q_col, ch6.profit_col, ch6.ret_col])
    df = df.dropna(subset=[ch6.e_col, ch6.q_col, ch6.profit_col, ch6.ret_col])

    # Feasibility mask
    df["feasible"] = df[ch6.ret_col] >= retention_min

    # For each (q, e), select max profit among feasible; if none feasible -> infeasible
    grouped = []
    for (qv, ev), g in df.groupby([ch6.q_col, ch6.e_col], sort=False):
        g_feas = g[g["feasible"]]
        if len(g_feas) == 0:
            grouped.append({
                ch6.q_col: qv, ch6.e_col: ev,
                "best_profit": np.nan,
                "best_plan": "infeasible",
                "plan_code": -1
            })
        else:
            best_row = g_feas.loc[g_feas[ch6.profit_col].idxmax()]
            plan = str(best_row[ch6.plan_col]).lower()
            grouped.append({
                ch6.q_col: qv, ch6.e_col: ev,
                "best_profit": float(best_row[ch6.profit_col]),
                "best_plan": plan,
                "plan_code": {"none": 0, "rehab": 1, "replacement": 2}.get(plan, -1)
            })

    out = pd.DataFrame(grouped)
    return out


def plot_ch6(
    ax: plt.Axes,
    ch6: Ch6Data,
    retention_min: float,
    title: str
) -> None:
    best = _ch6_best_plan_grid(ch6, retention_min=retention_min)

    # Profit grid
    ex, qy, P = _to_grid(best, x=ch6.e_col, y=ch6.q_col, z="best_profit")
    _, _, C = _to_grid(best, x=ch6.e_col, y=ch6.q_col, z="plan_code")

    # Background: profit heatmap
    im = ax.imshow(
        P,
        origin="lower",
        aspect="auto",
        extent=[ex.min(), ex.max(), qy.min(), qy.max()],
    )

    # Mask infeasible (retention constraint fails for all plans)
    infeasible = (C < 0) | np.isnan(P)
    if np.any(infeasible):
        mask = np.zeros_like(P, dtype=float)
        mask[infeasible] = 1.0
        ax.imshow(
            mask,
            origin="lower",
            aspect="auto",
            extent=[ex.min(), ex.max(), qy.min(), qy.max()],
            alpha=0.25,
        )
        ax.text(
            0.02, 0.02,
            f"masked: retention < {retention_min}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    # Overlay: best-plan regions (semi-transparent discrete colors)
    # codes: 0 none, 1 rehab, 2 replacement
    plan_cmap = ListedColormap([
        (0.15, 0.15, 0.15, 1.0),  # none (dark)
        (0.20, 0.60, 0.80, 1.0),  # rehab (blue-ish)
        (0.95, 0.85, 0.10, 1.0),  # replacement (yellow)
    ])

    # Create an RGBA overlay only where feasible (C>=0)
    overlay = np.ma.masked_where(C < 0, C)
    ax.imshow(
        overlay,
        origin="lower",
        aspect="auto",
        extent=[ex.min(), ex.max(), qy.min(), qy.max()],
        cmap=plan_cmap,
        alpha=0.28,  # transparency so profit is still readable
        vmin=0, vmax=2
    )

    # Draw boundaries between plans for crisp decision regions
    try:
        ax.contour(
            ex, qy, C,
            levels=[0.5, 1.5],
            linewidths=1.2,
            colors="white",
            alpha=0.95
        )
    except Exception:
        pass

    # Legend for plan overlay
    handles = [
        Patch(facecolor=plan_cmap(0), edgecolor="black", label="none"),
        Patch(facecolor=plan_cmap(0.5), edgecolor="black", label="rehab"),
        Patch(facecolor=plan_cmap(1.0), edgecolor="black", label="replacement"),
        Patch(facecolor="none", edgecolor="none", label=f"constraint: retention ≥ {retention_min}"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True, loc="upper left")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("ticket elasticity (epsilon)")
    ax.set_ylabel("injury probability (q)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Profit (best feasible plan)", rotation=90)


# -----------------------------
# Build the triptych
# -----------------------------
def build_triptych(
    model1_csv: str,
    model2_csv: str,
    ch6_csv: str,
    out_path: str,
    alpha0: float = 0.65,
    lambda0: float = 0.60,
    retention_min: float = 0.70,
    dpi: int = 300,
) -> None:
    m1 = load_model1(model1_csv)
    m2 = load_model2(model2_csv)
    c6 = load_ch6(ch6_csv)

    plt.close("all")
    fig = plt.figure(figsize=(15.8, 4.9))
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    plot_model1(ax1, m1, title="(a) Model I: Valuation sensitivity over (r, T)")
    plot_model2(ax2, m2, alpha0=alpha0, lambda0=lambda0,
                title="(b) Model II: Utility heatmap + roster-change contours")
    plot_ch6(ax3, c6, retention_min=retention_min,
             title="(c) Chapter 6: Profit heatmap + best-plan regions")

    fig.suptitle("Sensitivity Triptych: Model I / Model II / Injury-Response (Chapter 6)", fontsize=13, y=1.02)
    fig.tight_layout()

    _safe_mkdir_for_file(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_csv", default="", help="Model I CSV (default: ./model1_results.csv)")
    parser.add_argument("--model2_csv", default="", help="Model II CSV (default: ./model2_results.csv)")
    parser.add_argument("--ch6_csv", default="", help="Chapter 6 CSV (default: ./ch6_results.csv)")
    parser.add_argument("--out", default="", help="Output path (default: ./figures/sensitivity_triptych.png)")
    parser.add_argument("--alpha0", type=float, default=0.65, help="Baseline alpha marker")
    parser.add_argument("--lambda0", type=float, default=0.60, help="Baseline lambda marker")
    parser.add_argument("--retention_min", type=float, default=0.70, help="Retention constraint threshold")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Defaults: same folder as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model1 = os.path.join(script_dir, "model1_results.csv")
    default_model2 = os.path.join(script_dir, "model2_results.csv")
    default_ch6 = os.path.join(script_dir, "ch6_results.csv")
    default_out = os.path.join(script_dir, "figures", "sensitivity_triptych.png")

    model1_csv = args.model1_csv or default_model1
    model2_csv = args.model2_csv or default_model2
    ch6_csv = args.ch6_csv or default_ch6
    out_path = args.out or default_out

    build_triptych(
        model1_csv=model1_csv,
        model2_csv=model2_csv,
        ch6_csv=ch6_csv,
        out_path=out_path,
        alpha0=args.alpha0,
        lambda0=args.lambda0,
        retention_min=args.retention_min,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
