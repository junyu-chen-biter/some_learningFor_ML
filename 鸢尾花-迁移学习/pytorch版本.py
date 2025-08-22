import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


"""
注意，这里的数据集是内置在sklearn里面的，而且虽然叫花，但是其实这不是图像
数据集里面是只给了一些数据，不是图像识别
"""
# ===================== 数据处理 =====================
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 归一化（让训练更稳定）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ===================== 定义模型 =====================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 线性层

    def forward(self, x):
        return self.linear(x)  # 输出 raw logits，交给 CrossEntropyLoss 处理 softmax

model = LogisticRegressionModel(input_dim=4, output_dim=3)

# ===================== 定义损失 & 优化器 =====================
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降

# ===================== 训练模型 =====================
epochs = 200
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ===================== 预测 & 评估 =====================
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)  # 取最大概率类别
    acc = accuracy_score(y_test.numpy(), predicted.numpy())
    print("准确率: %.2f" % acc)
