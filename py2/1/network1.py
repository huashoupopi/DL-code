import paddle.nn.functional as F
from paddle.nn import Linear,Conv2D,MaxPool2D
import paddle

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        #设置全局的参数初始化方式。它接受多个初始化方式作为参数，并将它们应用于模型的所有参数。
        # paddle.nn.initializer.set_global_initializer(paddle.nn.initializer.Uniform(), paddle.nn.initializer.Constant())
        #定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride为1，padding=2
        self.conv1 = Conv2D(in_channels=1,out_channels=20,kernel_size=5,stride=1,padding=2)
        #定义池化层，池化核的大小kernel_size为2，stride为2
        self.max_pool1 = MaxPool2D(kernel_size=2,stride=2)
        #定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1,padding=2
        self.conv2 = Conv2D(in_channels=20,out_channels=20,kernel_size=5,stride=1,padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2,stride=2)
        #定义一层全连接层，输出维度为10
        self.fc = Linear(in_features=980,out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层不使用激活函数
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x,[x.shape[0],980])
        #x = paddle.reshape(x,[x.shape[0],-1])
        x = self.fc(x)
        # if label is not None:
            # acc = paddle.metric.accuracy(input=F.softmax(x),label=label)
            # acc = paddle.metric.accuracy(x,label=label)
        #     return x,acc
        # else:
        return x

