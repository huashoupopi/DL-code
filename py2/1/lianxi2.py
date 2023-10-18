import paddle
import paddle.nn.functional as F
from paddle.nn import Linear,Conv2D,MaxPool2D

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=1,out_channels=20,kernel_size=5,stride=1,padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2,stride=2)
        self.conv2 = Conv2D(in_channels=20,out_channels=20,kernel_size=5,stride=1,padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2,stride=2)
        self.fc = Linear(in_features=980,out_features=10)

    def forward(self,inputs,label):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x,[x.shape[0],980])
        x = self.fc(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x,label=label)
            return x,acc
        else:
            return x