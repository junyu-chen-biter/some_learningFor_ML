import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from jinja2.nodes import FromImport
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms,models,datasets
import imageio
import time
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

train_dir='flower_data/train'
test_dir='flower_data/test'

#这里是数据增强操作
data_transforms={
    'train':
        transforms.Compose([
            transforms.Resize([96,96]),
            transforms.RandomRotation(45),#随即旋转，在45度之内
            transforms.CenterCrop(64),#裁剪成64*64
            transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转，概率0.5
            transforms.RandomVerticalFlip(p=0.5),#随机竖直翻转，概率0.5
            transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),#这是改饱和度亮度的
            transforms.RandomGrayscale(p=0.025),#改灰度图片
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#idk
        ]),
    'valid':
    transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}


batch_size = 64
# 这个数据刚好是文件夹形式的，所以就是可以用这样的形式来做
image_datasets = {x: datasets.ImageFolder(os.path.join('flower_data/', x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes #calsses是一个属性，专门用来看名称的

#这里其实我也不会呢，为什么要两个名字呢？
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#可惜用了别人练好的模型唉唉
"""
这个东西叫做迁移学习，我们用的这个模型不是花朵的
但是我们直接用他的网络，把他的权重看成我们的初始权重
在这样的条件下去学习，是非常好的
"""
model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
"""
如果是true的话，相当于所有特征都冻住
"""
feature_extract = True



# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#这里调用我之前的models模型，使用152层的resnet模型
model_ft = models.resnet152()

#这里就是保留这个模型的参数，设置成了False
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        """
        我们发现这个模型最后一层是做了1000多个分类，但是我们的要求只做102个分类，于是我要修改最后一层
        """
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                   nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model('resnet', 102, feature_extract, use_pretrained=True)

#GPU计算
model_ft = model_ft.to(device)

# 模型保存
filename='checkpoint.pth'

# 是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True: #需要梯度的层，才改变，通常就是只改最后几层
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


# 优化器设置
#这里传进来了我们要更新的参数，就是最后的一层罢了
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
"""
这里是学习率的衰减，因为学习率一直不变是不好的，我们这个工具就是可以在学习的时候降低学习率
每七次训练降低一次学习率，降低为原来的1/10
"""
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss() #损失函数

# 导入需要的库
import time
import copy
import torch
filename='best.pt'


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename='best.pt'):
    """
    训练模型的主函数，包含训练、验证、模型保存和性能跟踪等功能

    参数说明：
    - model: 要训练的神经网络模型（如ResNet、Inception等）
    - dataloaders: 数据加载器字典，包含'train'（训练集）和'valid'（验证集）两个键
    - criterion: 损失函数（如交叉熵损失）
    - optimizer: 优化器（如SGD、Adam等）
    - num_epochs: 训练的总轮数，默认25轮
    - is_inception: 是否使用Inception模型，因为Inception有特殊的辅助输出层，默认False
    - filename: 保存最佳模型参数的文件名
    """

    # 记录训练开始时间，用于计算总训练时间
    since = time.time()

    # 初始化最佳准确率为0，用于跟踪训练过程中最好的模型性能
    """
    不一定是最后的一批训练效果最好，所以需要用这个来看取什么
    """
    best_acc = 0.0

    """
    以下代码块用于加载已保存的模型 checkpoint，实现断点续训
    取消注释后可以从上次训练中断的地方继续训练

    checkpoint = torch.load(filename)  # 加载保存的模型参数
    best_acc = checkpoint['best_acc']  # 恢复最佳准确率
    model.load_state_dict(checkpoint['state_dict'])  # 加载模型权重
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
    model.class_to_idx = checkpoint['mapping']  # 恢复类别映射（如果有的话）
    """

    # 将模型移动到指定的计算设备（GPU或CPU）
    model.to(device)

    # 初始化用于记录训练过程的列表
    val_acc_history = []  # 记录每轮验证集的准确率
    train_acc_history = []  # 记录每轮训练集的准确率
    train_losses = []  # 记录每轮训练集的损失值
    valid_losses = []  # 记录每轮验证集的损失值
    LRs = [optimizer.param_groups[0]['lr']]  # 记录学习率变化，初始学习率加入列表

    # 深拷贝当前模型的权重作为最佳模型权重的初始值
    best_model_wts = copy.deepcopy(model.state_dict())

    # 开始训练循环，遍历每一个epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # 打印当前轮次/总轮次
        print('-' * 10)  # 打印分隔线

        # 每个epoch都包含训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式：启用dropout和batch normalization更新
            else:
                model.eval()  # 设置模型为评估模式：关闭dropout，固定batch normalization参数

            # 初始化本轮的累计损失和正确预测数
            running_loss = 0.0
            running_corrects = 0

            # 遍历该阶段（训练或验证）的所有批次数据
            for inputs, labels in dataloaders[phase]:
                # 将输入数据和标签移动到指定的计算设备（GPU/CPU）
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零优化器的梯度，防止梯度累积
                optimizer.zero_grad()

                # 设置梯度计算的上下文：只有训练阶段需要计算梯度
                with torch.set_grad_enabled(phase == 'train'):
                    # Inception模型在训练时有两个输出（主输出和辅助输出），需要特殊处理
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)  # 获取两个输出
                        loss1 = criterion(outputs, labels)  # 主输出的损失
                        loss2 = criterion(aux_outputs, labels)  # 辅助输出的损失
                        loss = loss1 + 0.4 * loss2  # 总损失：主损失 + 0.4*辅助损失（Inception论文推荐）
                    else:
                        # 普通模型（如ResNet）只有一个输出
                        """
                        这里用的就是resnet模型
                        """
                        outputs = model(inputs)  # 前向传播：输入数据通过模型得到输出
                        loss = criterion(outputs, labels)  # 计算损失：输出与真实标签的差距

                    # 从输出中获取预测结果：取每个样本输出概率最大的类别索引
                    _, preds = torch.max(outputs, 1)  # torch.max返回最大值和索引，这里只需要索引(_忽略值)

                    # 只有训练阶段才进行反向传播和参数更新
                    if phase == 'train':
                        loss.backward()  # 反向传播：计算损失对各参数的梯度
                        optimizer.step()  # 优化器更新：根据梯度调整模型参数

                # 累计损失：将当前批次的损失乘以批次大小（得到总损失）后加入累计
                running_loss += loss.item() * inputs.size(0)
                # 累计正确预测数：将当前批次中预测正确的样本数加入累计
                running_corrects += torch.sum(preds == labels.data)

            # 计算本轮的平均损失：累计损失 / 该阶段的总样本数
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # 计算本轮的准确率：正确预测数 / 该阶段的总样本数（转换为double类型避免精度问题）
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 计算从训练开始到现在的时间
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # 打印当前阶段（训练/验证）的损失和准确率
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 验证阶段：如果当前准确率高于历史最佳，更新最佳准确率和最佳模型权重
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc  # 更新最佳准确率
                best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝当前模型权重作为最佳权重

                # 保存当前最佳模型的状态
                state = {
                    'state_dict': model.state_dict(),  # 模型权重
                    'best_acc': best_acc,  # 当前最佳准确率
                    'optimizer': optimizer.state_dict(),  # 优化器状态（用于断点续训）
                }
                torch.save(state, filename)  # 将状态保存到文件

            # 记录验证阶段的准确率和损失
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # 学习率调度器：根据验证损失调整学习率（假设已定义scheduler）
                scheduler.step(epoch_loss)

            # 记录训练阶段的准确率和损失
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        # 打印当前优化器的学习率
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        # 记录当前学习率
        LRs.append(optimizer.param_groups[0]['lr'])
        print()  # 空行分隔不同轮次

    # 计算总训练时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))  # 打印最佳验证准确率

    # 加载训练过程中得到的最佳模型权重
    model.load_state_dict(best_model_wts)

    # 返回训练好的模型和各种性能指标记录
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


# 开始训练
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = (
    train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=20, is_inception=(model_name=="inception")))

""""
一定要知道，这里的训练的是我们的resnet里面的最后一个全连接层，之前的全部参数我有点都没有改，但是这样是不行的
实际效果一般，于是我就要解冻之前每一层的参数，再去训练整个模型，这样的效果就很好
"""


for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.NLLLoss()

# Load the checkpoint

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
#model_ft.class_to_idx = checkpoint['mapping']

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = (
    train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10, is_inception=(model_name=="inception")))

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename='seriouscheckpoint.pth'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])


def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax

image_path = 'image_06621.jpg'
img = process_image(image_path)
imshow(img)

# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)


_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

fig=plt.figure(figsize=(20, 20))
columns =4
rows = 2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])]==cat_to_name[str(labels[idx].item())] else "red"))
plt.show()