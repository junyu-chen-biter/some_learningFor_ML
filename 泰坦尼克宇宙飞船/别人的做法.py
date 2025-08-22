import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#这个sklrean我会改一下的
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np



pd.set_option('display.max_columns', None)
df = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
y = df.pop("Transported")
df.head(5)



def CabinSplit(df):
    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["Num"] = list(map(float, df["Cabin"].str.split("/").str[1]))
    df["Side"] = df["Cabin"].str.split("/").str[2]
    df = df.drop(["Cabin", "PassengerId", "Destination", "Name"], axis=1)
    return df

df = CabinSplit(df)
df.head(5)


plt.figure(figsize=(22,1))
heatmap(pd.DataFrame(np.asarray(df.isna().sum()).reshape(1, df.shape[1]),
                     columns=list(df.columns)),cmap='Spectral', annot=True, fmt=".0f")



def FillNaN(df):
    mf = ["HomePlanet", "CryoSleep", "VIP", "Side", "Deck"]
    mean = ["Age"]
    zero = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Num"]
    df[mf] = df[mf].fillna(df.mode().iloc[0])
    df[mean] = df[mean].fillna(df[mean].mean())
    df[zero] = df[zero].fillna(0)
    return df
df = FillNaN(df)
df.head(5)



def FeatureCreation(df):
    df["MoneySpent"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df.insert(loc = 3, column="AgeCategories",value=0)
    df.loc[df["Age"] <= 14, "AgeCategories"] = 1
    df.loc[(df["Age"] > 14) & (df["Age"] <= 24), "AgeCategories"] = 2
    df.loc[(df["Age"] > 24) & (df["Age"] <= 64), "AgeCategories"] = 3
    df.loc[(df["Age"] > 64), "AgeCategories"] = 4
    return df
df = FeatureCreation(df)



def Encode(df):
    df = pd.concat([df, pd.get_dummies(df[["Deck", "Side", "HomePlanet"]])], axis=1)
    df = pd.concat([df, pd.get_dummies(df["AgeCategories"])], axis=1)
    df[["CryoSleep", "VIP"]] = df[["CryoSleep", "VIP"]].apply(LabelEncoder().fit_transform)
    df = df.drop(["Deck", "Side", "HomePlanet", "AgeCategories"], axis=1)
    return df
df = Encode(df)
y = LabelEncoder().fit_transform(y)
df.head(5)


def Norm(df):
    for col in df:
        if df[col].dtypes == "float64":
            df[col] = StandardScaler().fit_transform(np.array(df[col]).reshape(-1, 1))
    return df
df = Norm(df)
df.head(5)



x_train, x_validation, y_train, y_validation = train_test_split(df, y, test_size=0.2, shuffle=True, random_state=5)
class Dataset(Dataset):
    def __init__(self, x, y):
        self.df = np.array(x)
        self.df_labels = np.array(y)
        self.dataset = torch.tensor(self.df)
        self.labels = torch.tensor(self.df_labels)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]
batch_size = 64
train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(Dataset(x_validation, y_validation), batch_size=batch_size, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.model(x)

input_size = x_train.shape[1]
hidden_size = int(input_size * 2)
model = NeuralNetwork(input_size, hidden_size).to(device)
print(model)


epochs = 200
lr = 0.003
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def progress_bar(progress, total, lenght):
    percent = lenght * (progress / total)
    bar = "❚" * int(percent) + " " * (lenght - int(percent))
    return bar


train_plot = []
val_plot = []
train_accuracy_plot = []
val_accuracy_plot = []
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    train_correct = 0
    val_correct = 0
    for x, y in train_dataloader:
        optimizer.zero_grad()
        x, y = x.to(device).float(), y.to(device).float().unsqueeze(1)
        output = model(x)
        loss = criterion(output, y)
        train_loss += criterion(output, y).item()
        train_correct += (y == torch.round(torch.sigmoid(model(x)))).float().sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            x_val, y_val = x_val.to(device).float(), y_val.to(device).float().unsqueeze(1)
            pred = model(x_val)
            loss = criterion(pred, y_val)
            val_loss += criterion(pred, y_val).item()
            val_correct += (y_val == torch.round(torch.sigmoid(model(x_val)))).float().sum()
    train_plot.append((train_loss/len(train_dataloader)))
    val_plot.append((val_loss/len(val_dataloader)))
    train_accuracy_plot.append((train_correct / len(y_train)).item())
    val_accuracy_plot.append((val_correct / len(y_validation)).item())
    print(fr"|{progress_bar(epoch + 1, epochs, 50)}| {epoch + 1} / {epochs}, train_loss = {(train_loss/len(train_dataloader)):.5f}, val_loss = {(val_loss/len(val_dataloader)):.5f}, train_accuracy = {(train_correct / len(y_train)):.5f}, val_accuracy = {(val_correct / len(y_validation)):.5f}", end="\r")




fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot([i for i in range(1, epochs + 1)], train_plot, label="train")
ax1.plot([i for i in range(1, epochs + 1)], val_plot, label="val")
ax1.legend()
ax2.plot([i for i in range(1, epochs + 1)], train_accuracy_plot, label="train")
ax2.plot([i for i in range(1, epochs + 1)], val_accuracy_plot, label="val")
ax2.legend()
plt.show()


val = torch.tensor(np.array(x_validation)).float().to(device)
val_output = torch.round(torch.sigmoid(model(val)))
val_output = val_output.reshape(-1).type(torch.bool).cpu().numpy()
error_analysis_df = pd.DataFrame({"Prediction" : val_output, "TrueValue" : y_validation.astype(bool)})
TrueTrue = len(error_analysis_df.loc[(error_analysis_df["TrueValue"] == True) & (error_analysis_df["Prediction"] == True)])
FalseTrue = len(error_analysis_df.loc[(error_analysis_df["TrueValue"] == False) & (error_analysis_df["Prediction"] == True)])
TrueFalse = len(error_analysis_df.loc[(error_analysis_df["TrueValue"] == False) & (error_analysis_df["Prediction"] == False)])
FalseFalse = len(error_analysis_df.loc[(error_analysis_df["TrueValue"] == True) & (error_analysis_df["Prediction"] == False)])
plt.figure()
heatmap(pd.DataFrame({"True" : [FalseTrue, TrueTrue], "False" : [TrueFalse, FalseFalse]}, index=["False", "True"]),cmap='Spectral', annot=True, fmt=".0f")
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()

torch.save(model.state_dict(), 'model_weights.pth')




test_df = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
def Preprocessing(data):
    data = CabinSplit(data)
    data = FillNaN(data)
    data = FeatureCreation(data)
    data = Encode(data)
    data = Norm(data)
    return data
test_data = Preprocessing(test_df)
test_data.head(5)

model.load_state_dict(torch.load('model_weights.pth'))
data = torch.tensor(np.array(test_data)).float().to(device)
pred = torch.round(torch.sigmoid(model(data)))
pred = pred.reshape(-1).type(torch.bool).cpu().numpy()
submission = pd.DataFrame({"PassengerId" : test_df["PassengerId"], "Transported" : pred})
submission.to_csv("submission.csv", index=False)