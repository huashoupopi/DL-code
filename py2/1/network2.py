import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D,Linear,MaxPool2D
import numpy as np

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(in_channels=1,out_channels=20,kernel_size=5,stride=1,padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2,stride=2)
        self.conv2 = Conv2D(in_channels=20,out_channels=20,kernel_size=5,stride=1,padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2,stride=2)
        self.fc = Linear(in_features=980,out_features=10)
    #卷积层激活函数使用Relu,全连接层激活函数使用softmax
    def forward(self,inputs,label=None,check_shape=False,check_content=False):
        outputs1 = self.conv1(inputs)
        outputs2 = F.relu(outputs1)
        outputs3 = self.max_pool1(outputs2)
        outputs4 = self.conv2(outputs3)
        outputs5 = F.relu(outputs4)
        outputs6 = self.max_pool2(outputs5)
        outputs6 = paddle.reshape(outputs6,[outputs6.shape[0],-1])
        outputs7 = self.fc(outputs6)
        # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
        if check_shape:
            #打印每层网络设置的超参数——卷积核尺寸，卷积步长，卷积padding，池化核尺寸
            print("\n################ print network layer's superparams ##########")
            print(f"conv1-- kernel_size {self.conv1.weight.shape} padding:{self.conv1._padding} stride: {self.conv1._stride}")
            print(f"conv1-- kernel_size {self.conv2.weight.shape} padding:{self.conv2._padding} stride: {self.conv2._stride}")
            #print(f"max_pool1-- kernel_size: {self.max_pool1.pool_size} ,padding: {self.max_pool1.pool_padding} stride: {self.max_pool1.pool_stride}")
            #print(f"max_pool1-- kernel_size: {self.max_pool2.weight.shape} ,padding: {self.max_pool2.pool_padding} stride: {self.max_pool2.pool_stride}")
            print(f"fc-- weight_size:{self.fc.weight.shape} bias_size {self.fc.bias.shape}")

            #打印每层的输出尺寸
            print("\n############# print shape of features of every layer ############")
            print(f"outputs1_shape {outputs1}")
            print(f"outputs2_shape {outputs2}")
            print(f"outputs3_shape {outputs3}")
            print(f"outputs4_shape {outputs4}")
            print(f"outputs5_shape {outputs5}")
            print(f"outputs6_shape {outputs6}")
            print(f"outputs7_shape {outputs7}")
        #选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
            #打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n############### print convolution layer's kernel ############")
            print("conv1 params -- kernel weights:",self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:",self.conv2.weight[0][0])
            #创建随机数，随即打印某一个通道的输出值
            idx1 = np.random.randint(0,outputs1.shape[1])
            idx2 = np.random.randint(0,outputs4.shape[1])
            #打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print(f"\n The {idx1}th channel of conv1 layer:",outputs1[0][idx1])
            print(f"The {idx2}th channel of conv2 layer:",outputs1[0][idx2])
            print("The output of last layer:",outputs7[0],"\n")

        if label is not None:
            acc = paddle.metric.accuracy(input=F.softmax(outputs7),label=label)
            return outputs7,acc
        else:
            return outputs7