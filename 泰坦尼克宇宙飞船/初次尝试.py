import pandas as pd
import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


"""
这是一个分类问题，那么和房价预测的回归问题有啥差别呢，差别体现在哪里呢
"""

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#因为数据的特殊性，这里要做一个分成3份，投了别人的函数
def CabinSplit(df):
    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["Num"] = list(map(float, df["Cabin"].str.split("/").str[1]))
    df["Side"] = df["Cabin"].str.split("/").str[2]
    df = df.drop(["Cabin", "PassengerId", "Destination", "Name"], axis=1)
    return df


# 处理训练集：保留除最后两列外的所有列
train_features = CabinSplit(train)
# 处理测试集：保留除最后一列外的所有列
test_features = CabinSplit(test)
# 合并特征
all_features = pd.concat([train_features, test_features])

#数据预处理
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features=pd.get_dummies(all_features,dummy_na=True)

#转化为pytorch张量
all_features = all_features.astype(np.float32)
n_train=train.shape[0]#确实是包含了目标变量的
train_features=torch.tensor(all_features.iloc[:n_train].values,dtype=torch.float32)
test_features=torch.tensor(all_features.iloc[n_train:].values,dtype=torch.float32)
train_labels = torch.tensor(
    train.Transported.astype(int).values.reshape(-1, 1), dtype=torch.float32
)

#定义损失函数
loss=nn.BCELoss()
#输入的大小
in_features=train_features.shape[1]

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32,1),
        nn.Sigmoid()

    )
    return net

train_ls,test_ls=[],[]
net=get_net()
train_dataset = TensorDataset(train_features, train_labels)
train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)

optimizer=torch.optim.Adam(net.parameters(),lr=0.001,weight_decay=0)

for epoch in range(100):
    for x,y in train_iter:
        optimizer.zero_grad()
        l=loss(net(x),y)
        l.backward()
        optimizer.step()

#切换评估模式和禁用梯度
net.eval()
with torch.no_grad():
    preds = net(test_features).detach().numpy()
    preds_bool = preds >= 0.5

test['Transported'] = pd.Series(preds_bool.reshape(1, -1)[0])
submission = pd.concat([test['PassengerId'], test['Transported']], axis=1)
submission.to_csv('submission.csv', index=False)


