import pandas as pd
import numpy as np
import torch
from torch import nn

# 数据预处理部分保持不变...
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def CabinSplit(df):
    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["Num"] = list(map(float, df["Cabin"].str.split("/").str[1]))
    df["Side"] = df["Cabin"].str.split("/").str[2]
    df = df.drop(["Cabin", "PassengerId", "Destination", "Name"], axis=1)
    return df

train_features = CabinSplit(train)
test_features = CabinSplit(test)
all_features = pd.concat([train_features, test_features])

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

all_features = all_features.astype(np.float32)
n_train = train.shape[0]
train_features = torch.tensor(all_features.iloc[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features.iloc[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train.Transported.astype(int).values.reshape(-1, 1), dtype=torch.float32
)

# 定义损失函数
loss = nn.BCELoss()
in_features = train_features.shape[1]

# 使用 nn.Sequential 定义模型
model = nn.Sequential(
    nn.Linear(in_features, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_features)
    l = loss(outputs, train_labels)

    # 反向传播和优化
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {l.item():.4f}')

# 进行预测
#将模型切换到评估模式
model.eval()
#梯度临时禁用
with torch.no_grad():
    test_outputs = model(test_features)
    predictions = (test_outputs >= 0.5).float()

# 保存预测结果
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions.numpy().flatten().astype(bool)})
submission.to_csv('submission.csv', index=False)