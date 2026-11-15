# -*- coding: utf-8 -*-
"""
问题一——建模思路流程图（统计学与生物学双视角）
特点：中文标签、鲜艳配色、圆角卡片、清晰层级、自动导出 PNG/SVG
依赖：matplotlib（自带），无需安装 graphviz 等外部组件
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -------- 画布与中文字体设置 --------
plt.rcParams['figure.dpi'] = 150
# 这里列出多种常见中文字体名，环境存在哪一种就会用哪一种；找不到也能正常显示（会有告警但不影响图像）
plt.rcParams['font.sans-serif'] = [
    'SimHei','Microsoft YaHei','Noto Sans CJK SC',
    'Source Han Sans CN','Arial Unicode MS','DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

# -------- 小工具：节点与箭头 --------
def node(x, y, w, h, text, *,
         fc="#4FC3F7", ec="#0288D1", txt="#0D47A1",
         fontsize=11, lw=2, style="round,pad=0.02"):
    """绘制圆角矩形节点，并返回几何中心坐标以便连线"""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=style, linewidth=lw,
                          edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, color=txt, wrap=True)
    return (x + w/2, y + h/2)

def link(p1, p2, color="#37474F", lw=1.8,
         style="Simple,head_width=8,head_length=10"):
    """从p1指向p2画一根箭头"""
    ax.annotate("",
                xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# -------- 布局：上→下，左/右分支 --------
title = node(3.2, 8.0, 5.6, 0.9,
             "问题一目标：研究胎儿 Y 染色体浓度与孕周、BMI 的关系，\n建立并检验相应模型",
             fc="#FFD54F", ec="#F57F17", txt="#4E342E", fontsize=12)

inp = node(0.6, 6.9, 4.4, 0.9,
           "输入数据：\n孕周(J)、BMI(K)、Y浓度(V)、测序质量指标(L–P、AA)\n（含多次检测/可能失败样本）",
           fc="#A5D6A7", ec="#2E7D32", txt="#1B5E20")

prep = node(6.0, 6.9, 5.4, 0.9,
            "数据清洗与预处理：\n缺失/异常处理；同人多次检测整合；构造孕周(周)；\n筛选有效样本，标准化/变换（必要时）",
            fc="#80CBC4", ec="#00695C", txt="#004D40")

eda = node(0.6, 5.3, 10.8, 0.9,
           "探索性分析（EDA）：\n分布直方图、箱线图；相关系数热力图；BMI 分组对 Y 浓度的差异",
           fc="#FFAB91", ec="#D84315", txt="#BF360C")

branch = node(0.6, 4.1, 10.8, 0.5,
              "模型构建（双视角）",
              fc="#CE93D8", ec="#6A1B9A", txt="#4A148C", fontsize=12)

stat = node(0.6, 3.0, 5.1, 1.5,
            "统计学角度：多元线性回归\n"
            "Y浓度 = β₀ + β₁·孕周 + β₂·BMI "
            "(+ 交互/高阶项/对数变换)\n"
            "输出：系数、置信区间、p值、R²、残差诊断",
            fc="#64B5F6", ec="#1565C0", txt="#0D47A1")

bio = node(6.3, 3.0, 5.1, 1.5,
           "生物学角度：Logistic 增长模型\n"
           "刻画浓度随孕周的非线性增长（BMI 作为调制/分组）\n"
           "输出：增长速率/拐点、拟合优度与不确定性",
           fc="#81C784", ec="#2E7D32", txt="#1B5E20")

fuse = node(0.6, 1.7, 10.8, 0.9,
            "模型比较与融合：\n"
            "交叉验证/信息准则；误差传播与敏感性分析；\n"
            "结合临床阈值（Y≥4%）评估达标时间",
            fc="#F06292", ec="#AD1457", txt="#880E4F")

outp = node(2.5, 0.5, 7.0, 1.0,
            "输出：\n① 合理的 BMI 分组与最佳 NIPT 时点建议\n"
            "② 检测误差影响评估与鲁棒性结论\n③ 可视化与论文报告素材",
            fc="#FFE082", ec="#FF8F00", txt="#E65100", fontsize=12)

# -------- 连线 --------
for a, b in [(title, inp), (title, prep),
             (inp, eda), (prep, eda),
             (eda, branch), (branch, stat), (branch, bio),
             (stat, fuse), (bio, fuse), (fuse, outp)]:
    link(a, b, color="#6D4C41")

ax.set_title("问题一——建模思路流程图（统计学与生物学双视角）",
             fontsize=14, color="#263238", pad=12)

plt.tight_layout()

# 导出高清文件
plt.savefig("问题一流程图.png", bbox_inches="tight", dpi=300)
plt.savefig("问题一流程图.svg", bbox_inches="tight")
plt.show()
