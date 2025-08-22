import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from sklearn.linear_model import Ridge  # 岭回归模型
from sklearn.model_selection import cross_val_score  # 交叉验证函数

""""
先确实一个alpha的范围，画出图之后可以观察到这个参数取何值是效果最好
之后设置好这个参数，就可以有最终的模型了
alpha是正则化系数，控制正则化强度
alpha为0就为简单线性回归
alpha越大对过拟合的惩罚也就越大
"""


# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 合并训练集和测试集的特征列（不包括ID和目标变量）
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理
# 筛选数值特征并标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))  # 标准化
all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 填充缺失值

# 类别特征独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 划分训练集和测试集特征
n_train = train_data.shape[0]
train_features = all_features.iloc[:n_train].values
test_features = all_features.iloc[n_train:].values
train_labels = train_data['SalePrice'].values  # 提取训练集标签

# 超参数调优（验证alpha=20的合理性）
alphas = np.logspace(-3, 2, 50)
test_scores = []

for alpha in alphas:
    clf = Ridge(alpha=alpha)
    # 修正：使用训练集标签进行交叉验证（关键错误修复）
    test_score = np.sqrt(-cross_val_score(
        clf, train_features, train_labels,  # 此处y应为训练标签而非测试特征
        cv=10, scoring='neg_mean_squared_error'
    ))
    test_scores.append(np.mean(test_score))

# 可视化超参数与RMSE的关系
plt.plot(alphas, test_scores)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.xscale('log')
plt.title('Alpha vs RMSE (Cross Validation)')
plt.axvline(x=20, color='r', linestyle='--', label='alpha=20')  # 标记最优alpha
plt.legend()
plt.show()

# 使用最优alpha=20训练最终模型
best_alpha = 20
final_model = Ridge(alpha=best_alpha)
final_model.fit(train_features, train_labels)

# 预测测试集
test_preds = final_model.predict(test_features)

# 生成提交文件
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_preds
})
submission.to_csv('ridge_submission_alpha20.csv', index=False)
print(f"预测结果已保存，使用的alpha值为：{best_alpha}")