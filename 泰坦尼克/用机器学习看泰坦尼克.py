import hashlib
import os
import tarfile
import zipfile
from sklearn.model_selection import KFold
import requests
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l


# 加载数据集
# 读取训练集数据，路径为 "train.csv"
train = pd.read_csv("train.csv")
# 读取测试集数据，路径为 "test.csv"
test = pd.read_csv("test.csv")
test_passenger_ids = test['PassengerId']
# 合并训练集和测试集，以便进行统一的特征工程
data = pd.concat([train, test], sort=False)


# 特征工程部分
# 从姓名中提取头衔信息
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace(['Mme'], 'Mrs')
rare_titles = data['Title'].value_counts()[data['Title'].value_counts() < 10].index
data['Title'] = data['Title'].replace(rare_titles, 'Rare')
# 计算家庭规模
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
# 判断是否独自一人
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
# 填充缺失的年龄值
data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
# 填充缺失的票价信息
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
# 填充缺失的登船港口信息
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 处理Cabin列
# 提取Cabin的首字母作为甲板信息
data['Cabin'] = data['Cabin'].str[0]
data['Cabin'] = data['Cabin'].fillna('Unknown')  # 填充缺失值

# 从Ticket中提取前缀
data['Ticket'] = data['Ticket'].apply(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'None')
data['Ticket'] = data['Ticket'].str.replace('[^a-zA-Z]', '', regex=True)
data['Ticket'] = data['Ticket'].replace('', 'None')

"""
投降了，还是ai好用，这个数据处理由ai完成
"""
#将数据转化为热独编码，这个东西数据才会在后面转化为tensoe类型的数据，所以要小心
categorical_cols = ['Sex', 'Embarked', 'Title', 'Pclass', 'Cabin', 'Ticket']
data = pd.get_dummies(data, columns=categorical_cols, dummy_na=True)

# 删除不需要的列，或许不用删除啊，所以我先注释了，但是这个四个地方确实没有数据处理，不如要先处理一下数据什么的
#之前的代码要删除这四个数据，所以是直接没有去进行数据处理，于是我在上面有这些东西的增加
#data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

#划分训练于测试部分
train=data[:len(train)]
test=data[len(train):]

test=test.drop('Survived',axis=1)

#定义特征和目标列
features = [col for col in train.columns if col != 'Survived']
X = train[features]
y = train['Survived']
X_test = test

# 删除非数值列，防止 tensor 转换失败
non_numeric_cols = X.select_dtypes(include=['object']).columns
print("以下列是非数值列，将被删除：", list(non_numeric_cols))
X = X.drop(columns=non_numeric_cols)
X_test = X_test.drop(columns=non_numeric_cols)



# 转换为 float 类型（保险一点）
X = X.astype(np.float32)
X_test = X_test.astype(np.float32)

#将数据转化为pytorch张量
"""
大问题，只有数值类的值才可以转化为这个tensor的数据类型，那这里岂不是就直接炸了，我的数据处理一定还需要大改动
"""
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

"""
这后面的和李沐的几乎一摸一样，我其实是不套喜欢的，一点点改动把
"""


# 定义损失函数
loss = nn.BCEWithLogitsLoss()

# 明确输入层神经元数量
in_features = X.shape[1]


#昨天项目做到这里是弄不懂了，还有几个问题没有解决
# 首先这是模型能不能优化一下
# k折验证可否简单一点，直接调用
# train和pred这个东西能不能简化一点呢，这个老哥的代码就要比我这个好啊

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(0.3),  # 添加Dropout正则化
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )
    return net


# 计算二元交叉熵损失
def binary_loss(net, features, labels):
    preds = net(features)
    l = loss(preds, labels)
    return l.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(binary_loss(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(binary_loss(net, test_features, test_labels))
    return train_ls, test_ls


def k_fold_cv(X, y, num_epochs, learning_rate, weight_decay, batch_size, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_losses = []
    valid_losses = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        net = get_net()
        train_ls, valid_ls = train(net, X_train, y_train, X_valid, y_valid,
                                   num_epochs, learning_rate, weight_decay, batch_size)

        print(f"折 {fold+1}，训练loss: {train_ls[-1]:.4f}，验证loss: {valid_ls[-1]:.4f}")
        train_losses.append(train_ls[-1])
        valid_losses.append(valid_ls[-1])

        if fold == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch',
                     ylabel='loss', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')

    print(f"\n{k}-折交叉验证结果:")
    print(f"平均训练loss: {np.mean(train_losses):.4f}")
    print(f"平均验证loss: {np.mean(valid_losses):.4f}")
    return np.mean(train_losses), np.mean(valid_losses)



def train_and_pred(train_features, test_features, train_labels, test_passenger_ids,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='loss', xlim=[1, num_epochs], yscale='log')
    print(f'训练loss：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = torch.sigmoid(net(test_features)).detach().numpy()
    preds = (preds > 0.5).astype(int).reshape(-1)
    # 将其重新格式化以导出到Kaggle
    submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': preds})
    submission.to_csv('submission.csv', index=False)



k, num_epochs, lr, weight_decay, batch_size = 7, 100, 0.0005,0.00001, 64
train_l, valid_l = k_fold_cv(X, y, num_epochs, lr, weight_decay, batch_size, k=7)
print(f'{k}-折预测：平均训练loss：{float(train_l):f},' f'平均验证loss：{float(valid_l):f}')
plt.show()
train_and_pred(X, X_test, y, test_passenger_ids,
               num_epochs, lr, weight_decay, batch_size)

