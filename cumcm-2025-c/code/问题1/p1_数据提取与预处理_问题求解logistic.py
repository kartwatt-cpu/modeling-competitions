import pandas as pd
import re


df = pd.read_excel('附件.xlsx', sheet_name='男胎检测数据')


df = df[['检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()


def gestational_weeks_to_float(s: str):
    if pd.isna(s):
        return None
    s = str(s).strip().lower() 

    match = re.match(r'(\d+)(?:w|周)\+?(\d*)', s)
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return weeks + days / 7.0

    match2 = re.match(r'(\d+)\+(\d+)', s)
    if match2:
        weeks = int(match2.group(1)); days = int(match2.group(2))
        return weeks + days / 7.0

    match3 = re.match(r'(\d+)(?:w|周)', s)
    if match3:
        return float(match3.group(1))
    return None
##############################################################################################################################################

df['孕周(周)'] = df['检测孕周'].apply(gestational_weeks_to_float)

print(df[['检测孕周', '孕周(周)', '孕妇BMI', 'Y染色体浓度']].head(5))



# 筛选：BMI < 40 且 Y染色体浓度 <= 0.18
clean_df = df.dropna().copy()
clean_df = clean_df[(clean_df['孕妇BMI'] < 40) & (clean_df['Y染色体浓度'] <= 0.18)]


print(f"清洗前总样本数: {len(df)}")
print(f"清洗后总样本数: {len(clean_df)}")
print("BMI范围: {:.3f} - {:.3f}".format(clean_df['孕妇BMI'].min(), clean_df['孕妇BMI'].max()))
print("Y浓度范围: {:.3f} - {:.3f}".format(clean_df['Y染色体浓度'].min(), clean_df['Y染色体浓度'].max()))

# df.to_excel("数据清洗.xlsx")




##############################################################################################################################################

#建立分组Logistic增长模型
import numpy as np
from scipy.optimize import curve_fit

#Logistic增长函数
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))


def logistic_safe(t, L, k, t0):
    #避免exp过大溢出
    exp_val = -k * (t - t0)
    exp_val = np.clip(exp_val, -700, 700) 
    return L / (1 + np.exp(exp_val))

clean_df['BMI_group'] = pd.cut(clean_df['孕妇BMI'], bins=[0, 30, 35, 40], labels=['<30', '30-35', '35-40'])
group_data = {grp: sub for grp, sub in clean_df.groupby('BMI_group')}

initial_params = {
    '<30': [0.15, 0.5, 16],
    '30-35': [0.15, 0.5, 18],
    '35-40': [0.12, 0.5, 20]
}

#拟合每个BMI组的Logistic模型
fit_results = {}
for grp, subdf in group_data.items():
    x = subdf['孕周(周)'].values
    y = subdf['Y染色体浓度'].values
    #参数取值范围约束：L在0~0.3之间，k在>0，t0在0~60周之间
    bounds = ([0, 0, 0], [0.3, 5, 60])
    p0 = initial_params.get(grp, [0.1, 0.5, 18])
    popt, pcov = curve_fit(logistic_safe, x, y, p0=p0, bounds=bounds, maxfev=1000000)
    fit_results[grp] = (popt, pcov)
    print(f"{grp}组拟合参数: L={popt[0]:.4f}, k={popt[1]:.5f}, t0={popt[2]:.2f}")
