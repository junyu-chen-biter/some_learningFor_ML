import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# 读取训练集数据，路径为 "train.csv"
train = pd.read_csv("train.csv")
# 读取测试集数据，路径为 "test.csv"
test = pd.read_csv("test.csv")
all_features=pd.concat((train.iloc[:,1:-1],test.iloc[:,1:]))

#数据预处理
#筛选出类型不是物体的数据
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#那现在是没有进行任何的分组的，用泰坦尼克号那个模型说明白了就是要分类，但这里我还没有分类
#其实这里的类型太多了，有好几百个，所以在事实上不太适用这个分类标签啊

