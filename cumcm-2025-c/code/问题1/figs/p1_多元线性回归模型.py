import pandas as pd
import numpy as np

# 1. 读取数据并提取所需列
data = pd.read_excel('附件.xlsx', sheet_name='男胎检测数据')  # 读取附件中的男胎数据表
df = data[['检测孕周', '孕妇BMI', '年龄', 'Y染色体浓度']].copy()
df.columns = ['GA_raw', 'BMI', 'Age', 'Y_conc']  # 重命名列方便引用

# 2. 将孕周“周+天”字符串转换为小数周
# 提取数字部分（周数和天数）
ga_parts = df['GA_raw'].astype(str).str.lower().str.extract(r'(\d+)w(?:\+(\d+))?')
ga_parts = ga_parts.fillna(0).astype(int)  # NaN天数视作0天
df['GA_weeks'] = ga_parts[0]  # 周数
df['GA_days'] = ga_parts[1]   # 天数
df['GA_decimal'] = df['GA_weeks'] + df['GA_days'] / 7  # 转换为小数周

# 3. 数据清洗：剔除缺失和异常值
df.dropna(subset=['GA_decimal', 'BMI', 'Age', 'Y_conc'], inplace=True)       # 去除缺失值
df = df[df['BMI'] < 40]                                                     # 去除BMI>=40的异常值
df = df[(df['Y_conc'] >= 0) & (df['Y_conc'] <= 0.18)]                       # 去除Y浓度<0或>18%的异常值

# 4. 添加Y浓度对数列
df['log_Y'] = np.log(df['Y_conc'])

# 查看清洗后数据集概况
print(f"清洗后样本数: {df.shape[0]}，列: {list(df.columns)}")
print(df[['GA_decimal','BMI','Age','Y_conc','log_Y']].head(5))
df.to_excel("数据.xlsx")
############################################################################################################################################################
import statsmodels.formula.api as smf

# 5. 构建含交互项和二次项的线性模型
model = smf.ols('log_Y ~ GA_decimal * BMI + Age + I(BMI**2)', data=df).fit()

# 输出回归系数、标准误和显著性
coef_table = pd.DataFrame({
    'Coefficient': model.params,
    'Std_Error': model.bse,
    't_value': model.tvalues,
    'p_value': model.pvalues
})
coef_table[['Coefficient','Std_Error','t_value','p_value']] = coef_table[['Coefficient','Std_Error','t_value','p_value']].round(4)
print(coef_table)
coef_table.to_excel("coef_table.xlsx")
# 计算模型R平方和调整R平方
print(f"R平方 = {model.rsquared:.3f},  调整R平方 = {model.rsquared_adj:.3f}")
