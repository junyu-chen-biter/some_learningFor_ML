import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 新数据
majors = [
    '电子信息', '自动化', '计算机', '网安',
    '光电', '集电', '人工智能', '机器人工程',
    '软件工程', '大数据科学', '电气工程'
]
values = [98, 82, 82, 66, 46, 44, 32, 31, 31, 15, 13]

# 计算百分比
total = sum(values)
percentages = [f'{v/total*100:.1f}%' for v in values]
labels = [f'{major}\n{value} ({pct})' for major, value, pct in zip(majors, values, percentages)]

# 扩展鲜活的颜色方案（增加了几个颜色以匹配更多类别）
colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#D6A2E8', '#FF8E72',
    '#78E08F', '#E55039', '#60A3BC'
]

# 创建图形和轴（适当增大尺寸以容纳更多类别）
fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

# 绘制实心饼图
wedges, texts, autotexts = ax.pie(
    values,
    labels=majors,
    colors=colors,
    autopct='%1.1f%%',  # 显示百分比
    startangle=160,     # 调整起始角度，使数据分布更均衡
    pctdistance=0.82,   # 百分比标签距离中心的距离
    wedgeprops=dict(edgecolor='white', linewidth=2),  # 饼块边缘样式
    textprops=dict(fontsize=11)
)

# 美化文本
plt.setp(autotexts, size=10, weight="bold", color="black")
plt.setp(texts, size=11, weight='medium')

# 标题
ax.set_title('睿信书院专业数据分布', fontsize=18, pad=20, weight='bold', color='#333333')

# 添加图例，放在右侧（调整字体大小以适应更多类别）
ax.legend(labels, title="专业及数据", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
          title_fontsize=14, fontsize=10)

# 确保布局紧凑
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存图片
# plt.savefig('electronic_majors_pie.png', dpi=300, bbox_inches='tight')