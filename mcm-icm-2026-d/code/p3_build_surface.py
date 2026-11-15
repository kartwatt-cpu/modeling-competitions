import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

CSV_PATH = "outputs/frontier_scan.csv"
OUT_FIG  = "figures/valuation_surface_peak_only.png"

df = pd.read_csv(CSV_PATH)
df["alpha"] = df["alpha"].astype(float)
df["leverage"] = df["leverage"].astype(float)
df["V_final"] = df["V_final"].astype(float)

# Peak
peak = df.loc[df["V_final"].idxmax()].copy()

# Grid
pivot = (df.pivot_table(index="leverage", columns="alpha", values="V_final", aggfunc="mean")
           .sort_index(axis=0).sort_index(axis=1))
X = pivot.columns.values.astype(float)
Y = pivot.index.values.astype(float)
Z = pivot.values.astype(float)
Xg, Yg = np.meshgrid(X, Y)

# Premium low-saturation gradient (magma mid-range)
base = cm.get_cmap("magma")
colors = base(np.linspace(0.18, 0.88, 256))
premium_cmap = LinearSegmentedColormap.from_list("magma_premium", colors)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig = plt.figure(figsize=(9.6, 6.0), dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=22, azim=232)

# Z scaling
Z_SCALE = 1e8
ax.zaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / Z_SCALE:.2f}"))

# Surface
surf = ax.plot_surface(
    Xg, Yg, Z,
    cmap=premium_cmap,
    linewidth=0,
    antialiased=True,
    shade=True
)

# Clean panes
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor((0.90, 0.90, 0.90, 1.0))

ax.set_title("Valuation Surface under Joint Policy Variations", pad=16)
ax.set_xlabel(r"Strategic Preference $\alpha$", labelpad=12)
ax.set_ylabel(r"Leverage / Investment $\kappa$", labelpad=12)
ax.set_zlabel(r"Valuation $V(T)$ ($\times 10^8$ USD)", labelpad=14)

ax.tick_params(axis="x", pad=2)
ax.tick_params(axis="y", pad=2)
ax.tick_params(axis="z", pad=6)

# Peak marker only (no baseline)
z_span = float(np.nanmax(Z) - np.nanmin(Z))
lift = 0.06 * z_span  # moderat_*
