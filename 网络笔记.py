from torch import nn

"""
简单的全链接网络
"""
# 适用于结构化数据（如 Titanic 数据集）
input_size = 50  # 输入特征维度
hidden_size = 64
num_classes = 2  # 二分类问题

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.2),  # 防止过拟合
    nn.Linear(hidden_size, hidden_size//2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_size//2, num_classes),
    nn.Softmax(dim=1)  # 多分类用 Softmax，二分类可用 Sigmoid
)


"""
复杂的全连接网络
"""
# 更深的网络，适用于高维数据
input_size = 100
hidden_size = 128
num_classes = 2

model2 = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.BatchNorm1d(hidden_size),  # 批量归一化
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(hidden_size, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(hidden_size, hidden_size // 2),
    nn.BatchNorm1d(hidden_size // 2),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(hidden_size // 2, num_classes),
    nn.Softmax(dim=1)
)


"""
处理图像的
"""
# 简化版 CNN（如 MNIST 手写数字分类）
input_channels = 1  # 单通道图像
num_classes = 10

model3 = nn.Sequential(
    # 第一个卷积块
    nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # 第二个卷积块
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # 全连接分类器
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 128),  # 根据输入尺寸调整
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes),
    nn.Softmax(dim=1)
)



# 基于 LSTM 的文本分类
input_size = 300  # 词向量维度
hidden_size = 128
num_layers = 2
num_classes = 2
seq_length = 50  # 序列长度

model4 = nn.Sequential(
    nn.LSTM(
        input_size,
        hidden_size,
        num_layers,
        batch_first=True,
        bidirectional=True,  # 双向 LSTM
        dropout=0.2
    ),
    nn.Linear(hidden_size * 2, hidden_size),  # *2 因为双向
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, num_classes),
    nn.Softmax(dim=1)
)


"""
处理图像的restnet网络
"""
# 简化版 ResNet18 块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 使用 nn.Sequential 组装残差块
input_channels = 3
num_classes = 10

model5 = nn.Sequential(
    nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),

    ResidualBlock(64, 64),
    ResidualBlock(64, 128, stride=2),
    ResidualBlock(128, 256, stride=2),

    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(256, num_classes),
    nn.Softmax(dim=1)
)




# 简化版 Transformer 编码器
input_dim = 512
num_heads = 8
num_layers = 2
hidden_dim = 2048
num_classes = 2

encoder_layer = nn.TransformerEncoderLayer(
    d_model=input_dim,
    nhead=num_heads,
    dim_feedforward=hidden_dim,
    dropout=0.1
)

model6 = nn.Sequential(
    nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, num_classes),
    nn.Softmax(dim=1)
)