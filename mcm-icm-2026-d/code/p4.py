import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

# -------------------------------
# 0. 超参数设置（你可以按需要调整）
# -------------------------------
capacity        = 12000   # 球场容量
elasticity      = 0.8     # 价格弹性（越大价格变动对上座率影响越大）

conversion_rate = 0.5     # 新观众 -> 长期粉丝 转化率
clv_per_fan     = 300     # 每个长期粉丝的生命周期价值
lambda_clv      = 0.1     # CLV 在目标中的权重 0~1

disc_early      = 0.50    # 早鸟折扣 15% off
markup_premium  = 0.50    # 溢价加价 20%

np.random.seed(42)

# -------------------------------
# 1. 构建合成比赛数据（Las Vegas Aces 主场）
# -------------------------------
n_games = 100
data = pd.DataFrame({
    "opponent_quality": np.random.randint(1, 6, size=n_games),    # 1~5
    "day_of_week":      np.random.randint(1, 8, size=n_games),    # 1=Mon,...,7=Sun
    "is_hot_game":      np.random.binomial(1, 0.2, size=n_games), # 宿敌/焦点战 20%
    "weather_good":     np.random.binomial(1, 0.8, size=n_games), # 天气好 80%
    "playoff_imp":      np.random.binomial(1, 0.4, size=n_games), # 关系季后赛 40%
})
data["is_weekend"] = (data["day_of_week"] >= 6).astype(int)

# -------------------------------
# 2. 合成“真实”上座率（占比）作为学习目标
# -------------------------------
def compute_true_occ(row):
    opp     = row["opponent_quality"]
    weekend = row["is_weekend"]
    hot     = row["is_hot_game"]
    playoff = row["playoff_imp"]
    weather = row["weather_good"]
    # 简单线性打分 + Sigmoid，数值只是用于生成模拟数据
    score = 0.2*opp + 0.1*weekend + 0.15*hot + 0.15*playoff + 0.2*weather - 0.6
    return 1 / (1 + np.exp(-score))

data["occupancy"] = data.apply(compute_true_occ, axis=1)

# -------------------------------
# 3. 用 XGBoost 拟合需求函数 occ = f(features)
# -------------------------------
features = ["opponent_quality", "is_weekend",
            "is_hot_game", "weather_good", "playoff_imp"]
X = data[features]
y = data["occupancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=80,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

data["predicted_occ"] = model.predict(data[features]).clip(0, 1)

# -------------------------------
# 4. 为每场比赛生成“现实合理”的三档票价
#    base_price 随比赛重要性微调，再做 ±15% / +20%
# -------------------------------
def compute_base_price(row):
    # 设一个中枢价格，比如 50 美元
    base = 50.0
    # 对手越强，价格略高（每个等级+2）
    base += 2.0 * (row["opponent_quality"] - 3)
    # 周末附加 +3
    base += 3.0 * row["is_weekend"]
    # 宿敌/热门对阵再 +3
    base += 3.0 * row["is_hot_game"]
    # 关系季后赛再 +2
    base += 2.0 * row["playoff_imp"]
    # 限制在一个现实区间，比如 [40, 80]
    return float(np.clip(base, 40.0, 80.0))

data["base_price"] = data.apply(compute_base_price, axis=1)

# 三档价格：早鸟/正常/溢价
data["price_early"]   = data["base_price"] * (1.0 - disc_early)      # 约 15% 折扣
data["price_normal"]  = data["base_price"]
data["price_premium"] = data["base_price"] * (1.0 + markup_premium)  # 约 20% 加价

# -------------------------------
# 5. 不同价位下的上座率 & 票房
#    以 price_normal 为参考点做弹性调整
# -------------------------------
# occ(price) = occ_baseline * (p_normal / p)^elasticity
data["occ_normal"]  = data["predicted_occ"].clip(0, 1)
data["occ_early"]   = (data["predicted_occ"]
                       * (data["price_normal"] / data["price_early"])**elasticity).clip(0, 1)
data["occ_premium"] = (data["predicted_occ"]
                       * (data["price_normal"] / data["price_premium"])**elasticity).clip(0, 1)

# 各档位观众数
data["attend_early"]   = data["occ_early"]   * capacity
data["attend_normal"]  = data["occ_normal"]  * capacity
data["attend_premium"] = data["occ_premium"] * capacity

# 票房收入
data["rev_early"]   = data["attend_early"]   * data["price_early"]
data["rev_normal"]  = data["attend_normal"]  * data["price_normal"]
data["rev_premium"] = data["attend_premium"] * data["price_premium"]

# -------------------------------
# 6. 新粉丝 & LTV（未加权）
# -------------------------------
data["new_ratio_early"]   = 0.20
data["new_ratio_normal"]  = 0.10
data["new_ratio_premium"] = 0.05

data["new_fans_early"]   = data["attend_early"]   * data["new_ratio_early"]   * conversion_rate
data["new_fans_normal"]  = data["attend_normal"]  * data["new_ratio_normal"]  * conversion_rate
data["new_fans_premium"] = data["attend_premium"] * data["new_ratio_premium"] * conversion_rate

data["ltv_early"]   = data["new_fans_early"]   * clv_per_fan
data["ltv_normal"]  = data["new_fans_normal"]  * clv_per_fan
data["ltv_premium"] = data["new_fans_premium"] * clv_per_fan

# -------------------------------
# 7. 引入 CLV 加权系数 λ：总价值 = 票房 + λ·LTV
# -------------------------------
data["value_early"]   = data["rev_early"]   + lambda_clv * data["ltv_early"]
data["value_normal"]  = data["rev_normal"]  + lambda_clv * data["ltv_normal"]
data["value_premium"] = data["rev_premium"] + lambda_clv * data["ltv_premium"]

# -------------------------------
# 8. 逐场选择最优票档
# -------------------------------
def choose_best(row):
    v_dict = {
        "early":   row["value_early"],
        "normal":  row["value_normal"],
        "premium": row["value_premium"],
    }
    best_tier = max(v_dict, key=v_dict.get)
    price_map = {
        "early":   row["price_early"],
        "normal":  row["price_normal"],
        "premium": row["price_premium"],
    }
    return pd.Series({"best_tier": best_tier, "best_price": price_map[best_tier]})

best_df = data.apply(choose_best, axis=1)
data = pd.concat([data, best_df], axis=1)

# -------------------------------
# 9. 输出关键变量（方便你后续画图）
# -------------------------------
output_cols = [
    "opponent_quality", "day_of_week", "is_weekend",
    "is_hot_game", "weather_good", "playoff_imp",
    "base_price", "price_early", "price_normal", "price_premium",
    "predicted_occ",
    "occ_early", "occ_normal", "occ_premium",
    "attend_early", "attend_normal", "attend_premium",
    "rev_early", "rev_normal", "rev_premium",
    "new_fans_early", "new_fans_normal", "new_fans_premium",
    "ltv_early", "ltv_normal", "ltv_premium",
    "value_early", "value_normal", "value_premium",
    "best_tier", "best_price"
]
output = data[output_cols].copy()

print(f"\nλ_clv = {lambda_clv:.2f}, disc_early = {disc_early:.2f}, markup_premium = {markup_premium:.2f}")
print(output.head(10))

output.to_csv("ticket_pricing_recommendations_diff_price.csv", index=False, encoding="utf-8-sig")
print("\n[INFO] 已导出: ticket_pricing_recommendations_diff_price.csv")

# -------------------------------
# 10. 可视化示例：第1场比赛 三档票价比较
# -------------------------------
idx = 0
tiers = ["early", "normal", "premium"]
revenues = [
    data.at[idx, "rev_early"],
    data.at[idx, "rev_normal"],
    data.at[idx, "rev_premium"],
]
values = [
    data.at[idx, "value_early"],
    data.at[idx, "value_normal"],
    data.at[idx, "value_premium"],
]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(tiers, revenues, color=["skyblue","orange","green"])
plt.title(f"Game {idx+1}: Ticket Tier vs Revenue")
plt.ylabel("Box Office Revenue")

plt.subplot(1,2,2)
plt.bar(tiers, values, color=["skyblue","orange","green"])
plt.title(f"Game {idx+1}: Ticket Tier vs Total Value\n(Revenue + λ·LTV)")
plt.ylabel("Total Value")

plt.tight_layout()
plt.show()
