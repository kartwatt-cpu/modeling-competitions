import pandas as pd
import numpy as np


data = pd.read_excel('附件.xlsx', sheet_name='男胎检测数据')
df = data[['检测孕周', '孕妇BMI', '年龄', 'Y染色体浓度']].copy()
df.columns = ['GA_raw', 'BMI', 'Age', 'Y_conc'] 


ga_parts = df['GA_raw'].astype(str).str.lower().str.extract(r'(\d+)w(?:\+(\d+))?')
ga_parts = ga_parts.fillna(0).astype(int)  
df['GA_weeks'] = ga_parts[0] 
df['GA_days'] = ga_parts[1]  
df['GA_decimal'] = df['GA_weeks'] + df['GA_days'] / 7  

#数据清洗
df.dropna(subset=['GA_decimal', 'BMI', 'Age', 'Y_conc'], inplace=True)      
df = df[df['BMI'] < 40]                                                     #去除BMI>=40的异常值
df = df[(df['Y_conc'] >= 0) & (df['Y_conc'] <= 0.18)]                       #去除Y浓度<0或>18%的异常值


df['log_Y'] = np.log(df['Y_conc'])


print(f"清洗后样本数: {df.shape[0]}，列: {list(df.columns)}")
print(df[['GA_decimal','BMI','Age','Y_conc','log_Y']].head(5))
df.to_excel("数据.xlsx")
############################################################################################################################################################
import statsmodels.formula.api as smf

#构建含交互项和二次项的线性模型
model = smf.ols('log_Y ~ GA_decimal * BMI + Age + I(BMI**2)', data=df).fit()


coef_table = pd.DataFrame({
    'Coefficient': model.params,
    'Std_Error': model.bse,
    't_value': model.tvalues,
    'p_value': model.pvalues
})
coef_table[['Coefficient','Std_Error','t_value','p_value']] = coef_table[['Coefficient','Std_Error','t_value','p_value']].round(4)
print(coef_table)
coef_table.to_excel("coef_table.xlsx")

print(f"R平方 = {model.rsquared:.3f},  调整R平方 = {model.rsquared_adj:.3f}")
