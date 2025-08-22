import sys
import os
from torch.utils.data import dataloader
import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

torch.cuda.set_device(0) #设置使用的GPU是第0号GPU
batch_size = 1

#定义字符类别，包括数字、字母和中文省份缩写
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
                'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
                'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
                'zh_zang', 'zh_zhe']

#定义CNN
class char_cnn_net(nn.Module):
    def __init__(self):
        super().__init__()

#定义卷积层和全连接层
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.PReLU(),
            nn.Conv2d(64,16,3,1,1),
            nn.PReLU(),
            nn.Conv2d(16,4,3,1,1),
            nn.PReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1600, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256,67)
        )

#前向传播方法
    # 这里写的过于简化了，先进入卷积，变了形状之后在进入线性层
    def forward(self, x):
        y = self.conv(x).reshape(batch_size, -1,)
        return self.fc(y)

# 定义自定义数据集类，继承自PyTorch的Dataset类，用于处理字符图片数据
class CharPic(data.Dataset):  # 继承data.Dataset类，使其具备数据集的基本功能

    # 定义一个递归方法，用于获取指定目录下的所有文件路径
    def list_all_files(self, root):
        files = []  # 初始化一个空列表，用于存储文件路径
        list = os.listdir(root)  # 获取root目录下的所有文件和文件夹名称
        for i in range(len(list)):  # 遍历所有文件和文件夹
            element = os.path.join(root, list[i])  # 将root路径与子文件/文件夹名拼接
            if os.path.isdir(element):  # 判断当前元素是否为文件夹
                # 如果是文件夹，递归调用本方法，并将结果合并到files列表
                files.extend(self.list_all_files(element))
            elif os.path.isfile(element):  # 判断当前元素是否为文件
                files.append(element)  # 如果是文件，将路径添加到files列表
        return files  # 返回所有文件的路径列表。是一个字符串列表

    # 类的初始化方法，用于加载和预处理数据
    def __init__(self, root):
        super().__init__()  # 调用父类的初始化方法
        if not os.path.exists(root):  # 检查root路径是否存在
            raise ValueError('没有找到文件夹')  # 如果不存在，抛出异常
        files = self.list_all_files(root)  # 获取root目录下的所有文件路径

        self.X = []  # 初始化列表，用于存储图像数据
        self.y = []  # 初始化列表，用于存储标签数据
        # 假设numbers, alphbets, chinese已定义，存储所有可能的字符类别
        self.dataset = numbers + alphbets + chinese

        for file in files:  # 遍历所有文件
            # 读取图像，并转换为灰度图（注意：这里参数应该是cv2.IMREAD_GRAYSCALE）
            # 利用内置函数直接转化为灰度图，读的时候直接转了
            """
            这里是唯一一句处理图片数据的，就是用cv2直接转化为灰度像素矩阵，就可以直接用来后续处理了
            """
            src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

            if src_img.ndim == 3:  # 检查是否为灰度图（灰度图维度为2，彩色图为3）
                continue  # 如果不是灰度图，跳过该文件

            resize_img = cv2.resize(src_img, (20, 20))  # 将图像调整为20x20大小
            self.X.append(resize_img)  # 将处理后的图像添加到X列表

            dir = os.path.dirname(file)  # 获取文件所在的目录路径
            dir_name = os.path.split(dir)[-1]  # 获取目录的名称（作为标签）


            #这一段有点意思，算是手搓热独编码，就是dir_name其实是这个字符的具体名字
            #那么后面这个index_y的值就是这个值在dataset数组里面的索引
            # 找到当前标签在dataset中的索引，但会的是一个索引

            index_y = self.dataset.index(dir_name)
            self.y.append([index_y])  # 将索引作为标签添加到y列表

        self.X = np.array(self.X)  # 将图像列表转换为numpy数组
        self.y = np.array(self.y)  # 将标签列表转换为numpy数组


    # 定义获取数据集中单个样本的方法，PyTorch会自动调用该方法
    def __getitem__(self, index):
        tf = transforms.ToTensor()  # 创建一个将numpy数组转换为Tensor的转换器
        # 把图像转换为Tensor，并将标签转换为LongTensor格式返回
        return tf(self.X[index]), torch.LongTensor(self.y[index])

    # 定义返回数据集大小的方法，返回样本的数量
    def __len__(self) -> int:
        return len(self.X)  # 返回X列表的长度，即样本数量

#定义权重初始化函数
"""
这是对卷积层的初始化方法，其中用到了一个提取class name的方法
是卷积层的话就设置
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.1)
        m.bias.data.fill_(0)

#定义训练函数
def train(epoch, lr):
    model.train()

    #设置loss函数
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    #开始遍历dataloader中的数据。batch_idx是批次的索引，input是批次中的输入数据，target是对应的标签
    # 这里的enumerate方法就是直接给train_loader分成了train和test了
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        input, target = input, target.reshape(batch_size, )

        #创建Adam优化器，用于后续的参数更新
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()

        output = model(input)

        #执行前向传播，计算模型对当前批次数据的输出。算出输出和目标真值的损失值
        loss = criterion(output, target)

        # 执行反向传播，计算损失值相对于模型参数的梯度
        loss.backward()

        #如果损失大于历史损失，学习率乘以0.95，衰减学习率
        if loss_history and loss_history[-1] < loss.data:
            lr *= 0.95
        loss_history.append(loss.data)

        #计算出的梯度更新模型的参数
        optimizer.step()

        #监控训练过程
        if batch_idx % 12000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

#定义评估模型准确率的函数（计算模型在训练集上的准确率，并在必要时保存模型）
def get_accuracy(model, train_model_path):
    tot = len(train_loader.dataset)
    right = 0

    with torch.no_grad():
        for (input, target) in train_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)

            for idx in range(len(output)):
                # 这个方法是返回最大值的索引，因为这是一个多分类问题，所以最后一层的输出是67个
                # 现在就是在着67个里面，找到了一个值最大的作为预测结果
                _, predict = torch.topk(output[idx], 1)
                if predict == target[idx]:
                    right += 1

        acc = right / tot
        print('accuracy : %.3f' % acc)

        # 保存最好的那一个模型
        global best_acc
        if acc > best_acc:
            best_acc = acc
            torch.save(model, train_model_path)


if __name__ == '__main__':

    #设置数据目录和模型保存路径
    data_dir = '../images/cnn_char_train'
    train_model_path = 'char.pth'

    #构建定义的CNN模型，并利用GPU和权重进行训练
    model = char_cnn_net()
    #model = torch.load(train_model_path)
    model = model.cuda()
    model.apply(weights_init)

    print("Generate Model.")


    # 每次训练一张，这就叫在线学习，如果样本的大小很大就是可以用
    batch_size = 1

    #初始化数据集和DataLoader
    dataset = CharPic(data_dir)
    train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size,
                    num_workers=14, pin_memory=True, drop_last=True)

    global best_acc
    best_acc = 0.0

    #开始训练
    for epoch in range(100):
        lr = 0.001
        train(epoch, lr)
        get_accuracy(model, train_model_path)

    #保存最终训练的模型
    torch.save(model, train_model_path)

    print("Finish Training")
