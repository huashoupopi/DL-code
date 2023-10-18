import paddle
import numpy as np
from paddle.nn import Conv2D,MaxPool2D,Linear
import paddle.nn.functional as F

class LeNet(paddle.nn.Layer):
    def __init__(self,num_class=1):
        super().__init__()
        #创建卷积核池化层
        #创建第一个卷积层
        self.conv1 = Conv2D(in_channels=1,out_channels=6,kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2,stride=2)
        #尺寸的逻辑：池化层未改变通道数，当前通道数为6
        #创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6,out_channels=16,kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2,stride=2)
        #创建第三个卷积层
        self.conv3 = Conv2D(in_channels=16,out_channels=120,kernel_size=4)
        #尺寸的逻辑；输入层将数据拉平[B,C,H,W]->[B,C*H*W]
        #输入size是[28*28],经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120,out_features=64)
        #创建全连接层，第一个全连接层的输出神经元为64，第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64,out_features=num_class)
    #网络的前向计算过程
    def forward(self,x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        #尺寸的逻辑：输入层将数据拉平[B,C,W,H]->[B,C*H*W]
        x = paddle.reshape(x,[x.shape[0],-1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x