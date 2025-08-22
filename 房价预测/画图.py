import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
majors = ['车辆工程', '装甲车辆', '智能制造', '机械工程', '能源与动力', '工业工程']
values = [73, 68, 51, 46, 21, 16]

# 计算百分比
total = sum(values)
percentages = [f'{v/total*100:.1f}%' for v in values]
labels = [f'{major}\n{value} ({pct})' for major, value, pct in zip(majors, values, percentages)]

# 设置鲜活的颜色方案 - 采用明亮且对比明显的色彩
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# 绘制实心饼图
wedges, texts, autotexts = ax.pie(
    values,
    labels=majors,
    colors=colors,
    autopct='%1.1f%%',  # 显示百分比
    startangle=140,     # 起始角度
    pctdistance=0.8,    # 百分比标签距离中心的距离
    wedgeprops=dict(edgecolor='white', linewidth=2),  # 饼块边缘样式
    textprops=dict(fontsize=12)
)

# 美化文本
plt.setp(autotexts, size=11, weight="bold", color="black")
plt.setp(texts, size=12, weight='medium')

# 标题
ax.set_title('精工书院专业数据分布', fontsize=18, pad=20, weight='bold', color='#333333')

# 添加图例，放在右侧
ax.legend(labels, title="专业及数据", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
          title_fontsize=14, fontsize=11)

# 确保布局紧凑
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存图片
# plt.savefig('vehicle_majors_solid_pie.png', dpi=300, bbox_inches='tight')