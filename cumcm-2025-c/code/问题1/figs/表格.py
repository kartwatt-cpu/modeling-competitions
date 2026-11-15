# -*- coding: utf-8 -*-
"""
多元线性回归结果表导出 Word
"""

import os
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from docx import Document
from docx.shared import Inches

# ===================== 数据读取 & 预处理（与之前一致） =====================
EXCEL_PATH = '附件.xlsx'
SHEET_NAME = None

def read_sheet_auto(path, sheet_name=None):
    xls = pd.ExcelFile(path)
    if sheet_name:
        return pd.read_excel(path, sheet_name=sheet_name)
    cand = None
    for s in xls.sheet_names:
        s_low = s.lower()
        if ('男' in s) or ('胎' in s_low) or ('male' in s_low):
            cand = s; break
    if cand is None:
        cand = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=cand)

def gestational_weeks_to_float(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower()
    m = re.match(r'(\d+)\s*(?:w|周)\+?(\d+)?', s)
    if m:
        weeks = int(m.group(1)); days = int(m.group(2)) if m.group(2) else 0
        return weeks + days/7.0
    m = re.match(r'(\d+)\+(\d+)', s)
    if m:
        return int(m.group(1)) + int(m.group(2))/7.0
    m = re.match(r'(\d+)', s)
    if m:
        return float(m.group(1))
    return np.nan

# 读取
df_raw = read_sheet_auto(EXCEL_PATH, SHEET_NAME)
# 自动匹配列
GA_col  = [c for c in df_raw.columns if '孕周' in str(c)][0]
BMI_col = [c for c in df_raw.columns if 'BMI' in str(c)][0]
Age_col = [c for c in df_raw.columns if '年龄' in str(c) or 'age' in str(c).lower()][0]
Y_col   = [c for c in df_raw.columns if 'Y' in str(c)][0]

df = df_raw[[GA_col,BMI_col,Age_col,Y_col]].copy()
df.columns = ['GA_raw','BMI','Age','Y_conc']

df['GA'] = df['GA_raw'].apply(gestational_weeks_to_float)
y_raw = pd.to_numeric(df['Y_conc'], errors='coerce')
df['Y'] = y_raw/100.0 if (y_raw>1.0).mean()>0.5 else y_raw

df = df.dropna(subset=['GA','BMI','Age','Y']).copy()
df = df[(df['BMI']<40)&(df['Y']>=0)&(df['Y']<=0.18)]
df = df[(df['GA']>=10)&(df['GA']<=30)]
df['logY'] = np.log(df['Y'])

# ===================== 模型拟合 =====================
model = smf.ols('logY ~ GA * BMI + Age + I(BMI**2)', data=df).fit()
summary = model.summary()
print(summary)

# 系数表
coef_table = pd.DataFrame({
    'Coefficient': model.params,
    'Std_Error': model.bse,
    't_value': model.tvalues,
    'p_value': model.pvalues,
    'CI_lower': model.conf_int()[0],
    'CI_upper': model.conf_int()[1]
}).round(4)

# ===================== 导出 Word =====================
os.makedirs("figs", exist_ok=True)
doc = Document()
doc.add_heading('多元线性回归结果表 | Regression Results', level=1)

# 表格
table = doc.add_table(rows=1, cols=6, style='Light Grid')
hdr_cells = table.rows[0].cells
hdr_cells[0].text = '变量 (Variable)'
hdr_cells[1].text = '系数估计 (Coef.)'
hdr_cells[2].text = '标准误 (Std.Err)'
hdr_cells[3].text = 't 值 (t)'
hdr_cells[4].text = 'p 值 (p)'
hdr_cells[5].text = '95%置信区间 (95% CI)'

for idx,row in coef_table.iterrows():
    row_cells = table.add_row().cells
    row_cells[0].text = str(idx)
    row_cells[1].text = str(row['Coefficient'])
    row_cells[2].text = str(row['Std_Error'])
    row_cells[3].text = str(row['t_value'])
    row_cells[4].text = str(row['p_value'])
    row_cells[5].text = f"[{row['CI_lower']}, {row['CI_upper']}]"

doc.add_paragraph(f"模型判定系数 R² = {model.rsquared:.3f}, 调整后 R² = {model.rsquared_adj:.3f}")
doc.add_paragraph("注：显著性水平 * p<0.05, ** p<0.01, *** p<0.001")

out_path = os.path.join("figs","Regression_Result.docx")
doc.save(out_path)
print(f"✅ 已导出回归结果表到 {out_path}")
