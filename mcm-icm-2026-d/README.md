# MCM/ICM 2026 · Problem D

**题目**：职业体育联盟（WNBA）扩军周期下的球队经营决策与风险分析
**结果**：Honorable Mention（H 奖），Team Control Number 2622388
**我的角色**：编程手（三人队，另有建模与写作分工）

## 问题与思路

球队老板要同时兼顾两件事：短期竞争力与长期企业价值。题目要求在扩军、媒体转播增长、薪资帽约束下给出经营建议。

我们的处理分四层：

1. **财务动态仿真**：把竞争力、球迷影响力、现金流、债务写成状态变量，按赛季推进，输出多情景（悲观 / 基准 / 乐观）路径。
2. **阵容优化**：在薪资帽与人数约束下，用混合整数规划求 10 人最优阵容，目标函数为"竞争力—成本"的加权组合。
3. **需求预测**：XGBoost 预测上座率与周边收入，SHAP 给出特征贡献，识别哪些因素真正驱动收入。
4. **博弈与风险**：扩军选秀的暴露风险、票价与伤病响应的两个博弈层模块。

## 文件

| 文件 | 说明 |
|---|---|
| `ICM_D.pdf` | 论文终稿，26 页 |
| `ICM_D.tex` | LaTeX 源码（Overleaf 编译） |
| `references.bib` | 参考文献 |
| `code/` | Python 脚本：仿真、优化、预测、绘图 |

## 代码入口

- `code/P1_最优控制方法构建球队利润-估值优化模型.py` —— 财务控制与估值主体
- `code/p1_compare_3_strategies.py` —— 三种经营策略的情景对比
- `code/salary_catch.py` —— 薪资帽约束处理
- `code/p2_fa_trade_stability_v3_FIXED_DIRECT_REPLACE.py` —— 阵容优化的稳定性与交易方案
- `code/p3.py` `code/p4.py` `code/p5.py` —— 扩军选秀、票价、伤病响应模块

数据文件（赛题附件与处理后的中间数据）未入库，脚本中的路径需按本地目录调整。

## 复现

```bash
pip install numpy pandas matplotlib scikit-learn xgboost shap scipy
python code/p1_compare_3_strategies.py
```

编译器：XeLaTeX（`ICM_D.tex` 依赖 `mcmthesis` 模板与 `references.bib`）。
