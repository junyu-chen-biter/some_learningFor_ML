from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建一个逻辑回归对象
lr = LogisticRegression()

# 使用训练集训练模型
lr.fit(X_train, y_train)

# 对测试集进行预测
y_pred = lr.predict(X_test)

# 打印模型的准确率
print("准确率: %.2f" % accuracy_score(y_test, y_pred))