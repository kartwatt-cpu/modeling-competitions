# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ===================== #
#  文件路径              #
# ===================== #
VAL_FILE = Path("valuation_trajectories.csv")
CF_FILE  = Path("cashflow_structure.csv")
OUT_FIG1 = Path("fig1_valuation_trajectories.png")
OUT_FIG2 = Path("fig2_cashflow_structure.png")


# ===================== #
#  全局审美（论文级）     #
# ===================== #
def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12.3,
        "xtick.labelsize": 10.3,
        "ytick.labelsize": 10.3,
        "legend.fontsize": 10.2,
        "axes.titlepad": 10,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "lines.linewidth": 2.7,
        "lines.markersize": 7.0,
    })

setup_style()

PALETTE = {
    "Aggressive": "#1f77b4",
    "Balanced": "#ff7f0e",
    "Conservative": "#2ca02c",
    "drawdown": "#7f7f7f",
    "ticket": "#4C78A8",
    "media": "#F58518",
    "sponsor": "#54A24B",
    "salary": "#B279A2",
    "operation": "#E45756",
}

pref_order = ["Aggressive", "Balanced", "Conservative"]


# ===================== #
#  数轴格式化            #
# ===================== #
def fmt_money_m(x, _):
    return f"${x/1e6:.0f}M"
moneyM = FuncFormatter(fmt_money_m)

def fmt_pct(x, _):
    return f"{x*100:.0f}%"
pctfmt = FuncFormatter(fmt_pct)


# ===================== #
#  读取数据              #
# ===================== #
df_val = pd.read_csv(VAL_FILE)
req_val = {"time", "scenario_id", "strategy_id", "valuation"}
missing = req_val - set(df_val.columns)
if missing:
    raise ValueError(f"估值轨迹数据缺少列: {missing}")
has_drawdown = "drawdown" in df_val.columns

df_cf = pd.read_csv(CF_FILE)
req_cf = {"scenario_id","strategy_id","revenue_ticket","revenue_media","revenue_sponsor","cost_salary","cost_operation"}
missing2 = req_cf - set(df_cf.columns)
if missing2:
    raise ValueError(f"现金流结构数据缺少列: {missing2}")

df_val = df_val.sort_values(["scenario_id","strategy_id","time"])
scenarios = list(pd.unique(df_val["scenario_id"]))
strategies = list(pd.unique(df_val["strategy_id"]))
strategies = [s for s in pref_order if s in strategies] + [s for s in strategies if s not in pref_order]

time_vals = sorted(df_val["time"].unique())


# ===================== #
#  小工具：末端标注避让    #
# ===================== #
def annotate_line_ends(ax, end_points, x_offset=0.03, y_pad_frac=0.02):
    """
    end_points: list of dict {x, y, label, color}
    自动按 y 排序并施加轻微避让，避免文字重叠
    """
    if not end_points:
        return
    end_points = sorted(end_points, key=lambda d: d["y"])
    ys = [d["y"] for d in end_points]
    y_min, y_max = min(ys), max(ys)
    y_span = (y_max - y_min) if y_max > y_min else 1.0
    min_gap = y_pad_frac * y_span

    adj_y = []
    for i, d in enumerate(end_points):
        y = d["y"]
        if i > 0 and y - adj_y[-1] < min_gap:
            y = adj_y[-1] + min_gap
        adj_y.append(y)

    xlim = ax.get_xlim()
    dx = (xlim[1] - xlim[0]) * x_offset

    for d, y_new in zip(end_points, adj_y):
        ax.text(
            d["x"] + dx, y_new,
            d["label"],
            color=d["color"],
            fontsize=10.2,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75)
        )


# ===================== #
#  图1：估值轨迹（增强版）  #
# ===================== #
def plot_fig1():
    n = len(scenarios)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10.5, 3.45*n), sharex=True)
    if n == 1:
        axes = [axes]

    y_min = df_val["valuation"].min()
    y_max = df_val["valuation"].max()
    pad = 0.07 * (y_max - y_min)
    y_lim = (y_min - pad, y_max + pad)

    for ax, scen in zip(axes, scenarios):
        sub = df_val[df_val["scenario_id"] == scen].copy()

        # 主线
        end_points = []
        for strat in strategies:
            ssub = sub[sub["strategy_id"] == strat].sort_values("time")
            if ssub.empty:
                continue
            ax.plot(
                ssub["time"], ssub["valuation"],
                marker="o",
                color=PALETTE.get(strat, None),
                label=strat
            )
            # 末端标注点
            last = ssub.iloc[-1]
            end_points.append({
                "x": float(last["time"]),
                "y": float(last["valuation"]),
                "label": f"{strat}: {last['valuation']/1e6:.0f}M",
                "color": PALETTE.get(strat, "black")
            })

        ax.set_title(f"Scenario: {scen}", fontsize=13.6, weight="semibold")
        ax.set_ylabel("Valuation (USD)")
        ax.set_ylim(*y_lim)
        ax.yaxis.set_major_formatter(moneyM)
        ax.set_xticks(time_vals)
        ax.grid(True, alpha=0.16)
        ax.set_axisbelow(True)

        # drawdown：轻量叠加（只画 Balanced）
        peak_dd = None
        if has_drawdown:
            base_strat = "Balanced" if "Balanced" in strategies else strategies[0]
            base = sub[sub["strategy_id"] == base_strat].sort_values("time")
            if not base.empty:
                ax2 = ax.twinx()
                ax2.plot(base["time"], base["drawdown"], linestyle="--",
                         linewidth=2.0, color=PALETTE["drawdown"], alpha=0.62)
                ax2.fill_between(base["time"], 0, base["drawdown"].values,
                                 color=PALETTE["drawdown"], alpha=0.10)
                ax2.set_ylabel("Drawdown", fontsize=11)
                ax2.yaxis.set_major_formatter(pctfmt)
                ax2.tick_params(axis="y", labelsize=9.8)
                peak_dd = float(base["drawdown"].max())

        # 末端标注（自动避让）
        annotate_line_ends(ax, end_points, x_offset=0.02, y_pad_frac=0.035)

        # 小结框：本子图最佳策略 + 最大回撤
        # “最佳”定义：期末估值最高
        end_vals = {d["label"].split(":")[0]: d["y"] for d in end_points}
        if end_vals:
            best_str = max(end_vals, key=end_vals.get)
            best_val = end_vals[best_str] / 1e6
            box_lines = [f"Best (2028): {best_str}  |  {best_val:.0f}M"]
            if peak_dd is not None:
                box_lines.append(f"Peak drawdown: {peak_dd*100:.1f}%")
            ax.text(
                0.02, 0.06,
                "\n".join(box_lines),
                transform=ax.transAxes,
                fontsize=10.2,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd", alpha=0.92)
            )

    axes[-1].set_xlabel("Time (Season Index)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(strategies),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.55,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle("Valuation Trajectories under Scenarios and Strategies",
                 fontsize=18.5, weight="bold", y=0.995)

    fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.965])
    fig.savefig(OUT_FIG1, dpi=300, bbox_inches="tight", pad_inches=0.25)
    print(f"[INFO] 图1已保存到: {OUT_FIG1.resolve()}")
    plt.close(fig)


# ===================== #
#  图2：现金流结构（增强版） #
#  100%堆叠展示结构 + 黑点展示净现金流规模
# ===================== #
def plot_fig2():
    df = df_cf.copy()
    scenarios_cf = sorted(df["scenario_id"].unique())
    strategies_cf = strategies[:]  # 与图1一致
    df["strategy_id"] = pd.Categorical(df["strategy_id"], categories=strategies_cf, ordered=True)

    n = len(scenarios_cf)
    fig, axes = plt.subplots(1, n, figsize=(5.4*n, 5.3), sharey=True)
    if n == 1:
        axes = [axes]

    # 计算净现金流（绝对值）
    df["net_cashflow"] = (
        df["revenue_ticket"] + df["revenue_media"] + df["revenue_sponsor"]
        - df["cost_salary"] - df["cost_operation"]
    )

    # 右轴（Net CF）统一范围：扩一点避免刻度挤
    net_min = float(df["net_cashflow"].min())
    net_max = float(df["net_cashflow"].max())
    span = net_max - net_min if net_max > net_min else 1.0
    net_lo = net_min - 0.15 * span
    net_hi = net_max + 0.15 * span

    # 结构占比：收入与成本内部归一
    for j, (ax, scen) in enumerate(zip(axes, scenarios_cf)):
        sub = df[df["scenario_id"] == scen].sort_values("strategy_id").reset_index(drop=True)
        x = np.arange(len(strategies_cf))
        width = 0.70

        # 收入 share
        rev = sub[["revenue_ticket", "revenue_media", "revenue_sponsor"]].values
        rev_sum = rev.sum(axis=1, keepdims=True)
        rev_share = np.divide(rev, rev_sum, out=np.zeros_like(rev), where=rev_sum != 0)

        # 成本 share
        cost = sub[["cost_salary", "cost_operation"]].values
        cost_sum = cost.sum(axis=1, keepdims=True)
        cost_share = np.divide(cost, cost_sum, out=np.zeros_like(cost), where=cost_sum != 0)

        # ---- 收入（正向 0->1）----
        bottom = np.zeros(len(sub))
        labels = ["Ticket", "Media", "Sponsor"]
        colors = [PALETTE["ticket"], PALETTE["media"], PALETTE["sponsor"]]
        for i in range(3):
            ax.bar(
                x, rev_share[:, i], width=width, bottom=bottom,
                color=colors[i], edgecolor="white", linewidth=0.8, alpha=0.96,
                label=labels[i] if j == 0 else None
            )
            bottom += rev_share[:, i]

        # ---- 成本（负向 0->-1）----
        bottom_c = np.zeros(len(sub))
        labels_c = ["Salary", "Operation"]
        colors_c = [PALETTE["salary"], PALETTE["operation"]]
        for i in range(2):
            ax.bar(
                x, -cost_share[:, i], width=width, bottom=bottom_c,
                color=colors_c[i], edgecolor="white", linewidth=0.8,
                alpha=0.88, hatch="///",
                label=labels_c[i] if j == 0 else None
            )
            bottom_c -= cost_share[:, i]

        # 0轴强调
        ax.axhline(0, color="black", linewidth=1.1, alpha=0.85)

        # ---- 右轴：Net CF（只在最右侧子图显示刻度与标题）----
        ax2 = ax.twinx()
        ax2.set_ylim(net_lo, net_hi)

        ax2.scatter(
            x, sub["net_cashflow"].values,
            s=62, color="black", zorder=6,
            label="Net CF ($)" if j == 0 else None
        )

        # 控制右轴刻度密度 + padding，避免挤压
        ax2.yaxis.set_major_formatter(moneyM)
        ax2.set_yticks(np.linspace(net_lo, net_hi, 4))  # 只留 4 个刻度
        ax2.tick_params(axis="y", labelsize=9.8, pad=6)

        # 关键：除了最右侧，其它子图隐藏右轴刻度/标签（解决重叠）
        if j != n - 1:
            ax2.set_yticklabels([])
            ax2.set_ylabel("")
            ax2.tick_params(right=False)
            ax2.spines["right"].set_visible(False)
        else:
            ax2.set_ylabel("Net Cash Flow (USD)", fontsize=11)

        # 左轴（占比轴）
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["-100%", "-50%", "0", "50%", "100%"])
        ax.set_title(f"Scenario: {scen}", fontsize=13.6, weight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_cf)
        ax.set_xlabel("Strategy")
        ax.grid(True, axis="y", alpha=0.16)
        ax.set_axisbelow(True)

        # 小结框：净现金流最高策略（避免挡住柱子/轴）
        best_idx = int(np.argmax(sub["net_cashflow"].values))
        best_str = str(sub.loc[best_idx, "strategy_id"])
        best_net = sub.loc[best_idx, "net_cashflow"] / 1e6
        ax.text(
            0.02, 0.06,
            f"Best net CF: {best_str}\n{best_net:.1f}M",
            transform=ax.transAxes,
            fontsize=10.0,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd", alpha=0.92)
        )

    axes[0].set_ylabel("Structure Share (Revenue vs Cost)")

    # Legend：统一放底部（更干净，且不抢标题）
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=PALETTE["ticket"], edgecolor="white", label="Ticket"),
        Patch(facecolor=PALETTE["media"], edgecolor="white", label="Media"),
        Patch(facecolor=PALETTE["sponsor"], edgecolor="white", label="Sponsor"),
        Patch(facecolor=PALETTE["salary"], edgecolor="white", hatch="///", label="Salary"),
        Patch(facecolor=PALETTE["operation"], edgecolor="white", hatch="///", label="Operation"),
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=7, label="Net CF ($)")
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=6,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.55,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle("Cash Flow Structure (100% Stacked) + Net Cash Flow",
                 fontsize=18.5, weight="bold", y=0.985)

    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.93])
    fig.savefig(OUT_FIG2, dpi=300, bbox_inches="tight", pad_inches=0.25)
    print(f"[INFO] 图2已保存到: {OUT_FIG2.resolve()}")
    plt.close(fig)



if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
