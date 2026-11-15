import pandas as pd
import numpy as np

fp = "surface_data_for_3d_surface.csv"  # <- 改成你的实际路径
df = pd.read_csv(fp)

# 兼容列名
rename_map = {}
for c in df.columns:
    k = c.strip().lower().replace(" ", "_")
    if k in ["salary_ratio", "f"]: rename_map[c] = "f"
    if k in ["ticket_multiplier", "p", "price_multiplier"]: rename_map[c] = "p"
    if k in ["debt_ratio", "u", "leverage_ratio"]: rename_map[c] = "u"
    if k in ["valuation", "v", "value"]: rename_map[c] = "V"
df = df.rename(columns=rename_map)

print("Columns:", df.columns.tolist())
print("\n=== V(T) summary by u ===")
print(df.groupby("u")["V"].agg(["count","min","max","mean","std"]).reset_index())

print("\n=== Check if V(T) identical across u for each (f,p) ===")
pv = df.pivot_table(index=["f","p"], columns="u", values="V", aggfunc="mean")
diff_span = (pv.max(axis=1) - pv.min(axis=1))
print(diff_span.describe())

# 如果你想看“有多少点几乎不变”
tol = 1e-6
print("\nFraction of (f,p) with almost no u-effect:",
      (diff_span <= tol).mean())
