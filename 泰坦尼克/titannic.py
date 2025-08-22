# 此代码参考了一位老哥提交的代码，由于参考 CSDN 上的内容未能实现，
# 相关参考内容保存在 jupyter 笔记本中
import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import numpy as np  # 导入 numpy 库，用于数值计算
from sklearn.model_selection import train_test_split, cross_val_score  # 导入 sklearn 库中的 train_test_split 用于划分数据集，cross_val_score 用于交叉验证
from xgboost import XGBClassifier  # 导入 xgboost 库中的 XGBClassifier 用于构建分类模型
from sklearn.preprocessing import LabelEncoder  # 导入 sklearn 库中的 LabelEncoder 用于对分类特征进行编码

"""
这个代码用了大量的已经好的函数，比如绝大部分都是特征工程，而实际的模型群都是用是已经内置好的函数，
比如XGBC这些，包括爱他的K折交叉验证模型
首先是特征工程
先合并test和train数据集，之后对齐进行一些修改和缺失值处理，这个用多个方法都是直接用中位数来填充
但是在一些大佬的代码里面，处理的方式更加的复杂
然后在处理完成之后，再将这个数据重新分成train和test，然后在去套模型

！！！
有一个种重要的数据梳理的要点，就是最后将数据转化为什么，编码，标签编码，热度编码等等，这些编码用于处理不同的模型

"""

# 加载数据集
# 读取训练集数据，路径为 "train.csv"
train = pd.read_csv("train.csv")
# 读取测试集数据，路径为 "test.csv"
test = pd.read_csv("test.csv")
# 提取测试集中的乘客 ID 列，用于后续结果输出
test_passenger_ids = test['PassengerId']

# 合并训练集和测试集，以便进行统一的特征工程
data = pd.concat([train, test], sort=False)

# 特征工程部分

# 从姓名中提取头衔信息
# 使用正则表达式从姓名中提取头衔（例如 Mr., Mrs., Miss 等）
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# 将 'Mlle' 和 'Ms' 统一替换为 'Miss'
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
# 将 'Mme' 替换为 'Mrs'
data['Title'] = data['Title'].replace(['Mme'], 'Mrs')
# 找出出现次数少于 10 次的头衔，将其归为 'Rare' 类别
rare_titles = data['Title'].value_counts()[data['Title'].value_counts() < 10].index
data['Title'] = data['Title'].replace(rare_titles, 'Rare')

# 计算家庭规模
# 家庭规模等于兄弟姐妹数量（SibSp）加上父母子女数量（Parch）再加 1（本人）
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# 判断是否独自一人
# 如果家庭规模为 1，则标记为 1，表示独自一人；否则标记为 0
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# 填充缺失的年龄值
# 根据乘客的舱位（Pclass）和性别（Sex）分组，用每组的年龄中位数填充缺失的年龄值
"""
这段代码的含义是：在名为data的数据集中，对Age这一列的数据进行处理。
具体做法是，先按照Pclass（可能是乘客等级之类的分类变量）和Sex（性别）这两个列进行分组
然后对于每个分组内的Age列数据，通过transform方法应用一个匿名函数（lambda函数）。
这个匿名函数的作用是，对于每个分组内Age列中的缺失值（NaN），用该分组内Age列的中位数进行填充。
最终，将处理后的数据重新赋值给data数据集的Age列。
然后就是X是直接提取出来了一大推可以用到的特征，把y设置成我们最终要预测的survival
之后就是直接套用那个模型，又设置了一个5折交叉验证去发现这是合理的，此时此刻，那就是训练好了一个模型
之后preds = model.predict(X_test)，这既是结果
"""
data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# 填充缺失的票价信息
# 用票价的中位数填充缺失的票价
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# 填充缺失的登船港口信息
# 用出现次数最多的登船港口填充缺失值
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])



# 对分类特征进行编码
# 定义需要编码的列
label_cols = ['Sex', 'Embarked', 'Title']
# 遍历需要编码的列
for col in label_cols:
    le = LabelEncoder()  # 创建 LabelEncoder 对象
    data[col] = le.fit_transform(data[col])  # 对列进行编码

# 删除不需要的列
# 删除 'Cabin', 'Name', 'Ticket' 列，因为这些列在模型训练中可能用处不大
data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# 模型训练部分

# 重新划分训练集和测试集
# 从合并的数据集中提取训练集部分
train = data[:len(train)]
# 从合并的数据集中提取测试集部分
test = data[len(train):]

# 定义特征列
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
# 提取训练集的特征
X = train[features]
# 提取训练集的标签
y = train['Survived']
# 提取测试集的特征
X_test = test[features]

# 训练模型
# 创建 XGBClassifier 模型，设置参数
# n_estimators 表示树的数量，max_depth 表示树的最大深度，learning_rate 表示学习率
# 注：use_label_encoder 和 eval_metric 在新版本 xgboost 中可能会引发警告，可考虑移除
model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, use_label_encoder=False, eval_metric='logloss')
# 使用训练集数据对模型进行训练
model.fit(X, y)

# 进行交叉验证并计算准确率
# 使用 5 折交叉验证计算模型的准确率，这是sklearn用的k折交叉验证，
scores = cross_val_score(model, X, y, cv=5)
# 打印交叉验证的平均准确率和标准差
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# 进行预测
# 使用训练好的模型对测试集进行预测
preds = model.predict(X_test)

# 输出结果
# 创建一个包含乘客 ID 和预测结果的 DataFrame
submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': preds})
# 将结果保存为 CSV 文件，不包含索引列
submission.to_csv('submission.csv', index=False)