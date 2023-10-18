# import paddle
# import paddle.nn as nn
#
# # 卷积模板 每次卷积与relu都会一起使用，打包一下
# class BasicConv2D(paddle.nn.Layer):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2D, self).__init__()
#         self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, **kwargs)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x
#
#
# #  incepetion 结构
# class Inception(paddle.nn.Layer):
#     #  ch1x1 ch3x3reduce ..... 都是卷积核的个数
#     def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
#         super(Inception, self).__init__()
#
#         # 1x1 卷积
#         self.branch1 = BasicConv2D(in_channels, ch1x1, kernel_size=1)
#
#         # 1x1卷积 + 3x3卷积
#         self.branch2 = nn.Sequential(
#             BasicConv2D(in_channels, ch3x3reduce, kernel_size=1),
#             BasicConv2D(ch3x3reduce, ch3x3, kernel_size=3, padding=1)  # 保证每个分支得到的特征矩阵的高度和宽度是相同的 ， 输出大小等于输入大小
#             #  output_size = (input_size - 3 + 2*1)/1 + 1 = input_size
#         )
#
#         # 1x1 卷积 + 5x5卷积
#         self.branch3 = nn.Sequential(
#             BasicConv2D(in_channels, ch5x5reduce, kernel_size=1),
#             BasicConv2D(ch5x5reduce, ch5x5, kernel_size=5, padding=2)  # 保证  输出大小等于输入大小
#             #  output_size = (input_size - 5 + 2*2)/1 + 1 = input_size
#         )
#
#         # 3x3 MaxPooling + 1x1 卷积
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
#             BasicConv2D(in_channels, pool_proj, kernel_size=1)
#         )
#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#
#         outputs = [branch1, branch2, branch3, branch4]
#         # 拼接输出
#         # Tensor的排列顺序是[batch, channel, h, w] , 所以要在深度上进行拼接时，就要输入1
#         return paddle.concat(outputs, 1)
#
#
# # 辅助分类器结构
# class InceptionAux(paddle.nn.Layer):
#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#
#         self.averagePool = nn.AvgPool2D(kernel_size=5, stride=3)
#         # 卷积大小1x1 ，步距1  卷积核：128
#         self.conv = BasicConv2D(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
#
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         # aux1 ；N x 512 x 14 x 14  aux2: N x 528 x 14 x 14
#         x = self.averagePool(x)
#         # aux1 : N x 512 x 4 x 4   aux2 : N x 528 x 4 x 4
#         x = self.conv(x)
#         #  N x 128 x 4 x 4
#         x = paddle.flatten(x, 1)
#         # 原论文0.7 -> 0.5
#         # x = paddle.nn.functional.dropout(x, 0.5, training=self.training)
#         # N x 2048
#         x = paddle.nn.functional.relu(self.fc1(x))
#         x = paddle.nn.functional.dropout(x, 0.7, training=self.training)
#         # N x 1024
#         x = paddle.nn.functional.relu(self.fc2(x))
#         # N x num_classes
#         return x
#
#
# # 定义 GoogleNet 网络
# class GoogleNet(paddle.nn.Layer):
#     #  aux_logits : 是否使用辅助分类器
#     def __init__(self, num_classes=2, aux_logits=True):
#         super(GoogleNet, self).__init__()
#         # name_scope = self.full_name()
#         self.aux_logits = aux_logits
#
#         # 根据参数表格搭建网络
#         self.conv1 = BasicConv2D(3, 64, kernel_size=7, stride=2,
#                                  padding=3)  # 特征矩阵的高和宽缩减为原来的一般，padding=3 （224-7+2*3）/2 + 1 = 112.5 向下取整112
#         # 计算结果为小数时：  ceil_mode=True : 小数向上取整 ， false：向下取整
#         self.maxpool1 = nn.MaxPool2D(3, stride=2, ceil_mode=True)
#
#         # LocalResponseNormal 作用不大，没有实现
#         # nn.LocalResponseNorm()
#
#         self.conv2 = BasicConv2D(64, 64, kernel_size=1)
#         self.conv3 = BasicConv2D(64, 192, kernel_size=3, padding=1)
#         # nn.LocalResponseNorm()
#         self.maxpool2 = nn.MaxPool2D(3, stride=2, ceil_mode=True)
#
#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2D(3, stride=2, ceil_mode=True)
#
#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2D(3, stride=2, ceil_mode=True)
#
#         self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
#
#         if self.aux_logits:
#             self.aux1 = InceptionAux(512, num_classes)
#             self.aux2 = InceptionAux(528, num_classes)
#
#         # 自适应平均池化下采样，无论你输入的特征矩阵的高和宽是多大，都能得到指定的高和宽 （1，1）
#         self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.conv1(x)
#         # N x 64 x 112 x 112
#         x = self.maxpool1(x)
#         # N x 64 x 56 x 56
#         x = self.conv2(x)
#         # N x 64 x 56 x 56
#         x = self.conv3(x)
#         # N x 192 x 56 x 56
#         x = self.maxpool2(x)
#
#         # N x 192 x 28 x 28
#         x = self.inception3a(x)
#         # N x 256 x 28 x 28
#         x = self.inception3b(x)
#         # N x 480 x 28 x 28
#         x = self.maxpool3(x)
#         # N x 480 x 14 x 14
#         x = self.inception4a(x)
#         # N x 512 x 14 x 14
#         if self.training and self.aux_logits:  # eval model lose this layer
#             aux1 = self.aux1(x)
#
#         x = self.inception4b(x)
#         # N x 512 x 14 x 14
#         x = self.inception4c(x)
#         # N x 512 x 14 x 14
#         x = self.inception4d(x)
#         # N x 528 x 14 x 14
#         if self.training and self.aux_logits:  # eval model lose this layer
#             aux2 = self.aux2(x)
#
#         x = self.inception4e(x)
#         # N x 832 x 14 x 14
#         x = self.maxpool4(x)
#         # N x 832 x 7 x 7
#         x = self.inception5a(x)
#         # N x 832 x 7 x 7
#         x = self.inception5b(x)
#         # N x 1024 x 7 x 7
#
#         x = self.avgpool(x)
#         # N x 1024 x 1 x 1
#         x = paddle.flatten(x, 1)
#         # N x 1024
#         x = self.dropout(x)
#         x = self.fc(x)
#         # N x 1000 (num_classes)
#         if self.training and self.aux_logits:  # eval model lose this layer
#             return x, aux2, aux1
#         return x

import paddle
import paddle.nn as nn

# 卷积模板 每次卷积与relu都会一起使用，打包一下
class BasicConv(paddle.nn.Layer):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,**kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(paddle.nn.Layer):
    def __init__(self,in_channels,ch1x1,ch3x3,ch3x3reduce,ch5x5,ch5x5reduce,pool_proj):
        super().__init__()
        # 1x1 卷积
        self.conv1 = BasicConv(in_channels,ch1x1,kernel_size=1)
        # 1x1卷积 + 3x3卷积
        self.conv2 = nn.Sequential(
            BasicConv(in_channels,ch3x3reduce,kernel_size=1),
            BasicConv(ch3x3reduce,ch3x3,kernel_size=3,padding=1)    # 保证每个分支得到的特征矩阵的高度和宽度是相同的 ， 输出大小等于输入大小
            #  output_size = (input_size - 3 + 2*1)/1 + 1 = input_size
        )
        # 1x1 卷积 + 5x5卷积
        self.conv3 = nn.Sequential(
            BasicConv(in_channels,ch5x5reduce,kernel_size=1),
            BasicConv(ch5x5reduce,ch5x5,kernel_size=5,padding=2)       # 保证  输出大小等于输入大小
            #  output_size = (input_size - 5 + 2*2)/1 + 1 = input_size
        )
        # 3x3 MaxPooling + 1x1 卷积
        self.conv4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3,stride=1,padding=1),
            BasicConv(in_channels,pool_proj,kernel_size=1)
        )

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        outputs = [out1,out2,out3,out4]
        # 拼接输出
        # Tensor的排列顺序是[batch, channel, h, w] , 所以要在深度上进行拼接时，就要输入1
        return paddle.concat(outputs,1)

# 辅助分类器结构
class InceptionAux(paddle.nn.Layer):
    def __init__(self,in_channels,num_class):
        super().__init__()
        self.averagePool = nn.AvgPool2D(kernel_size=5,stride=3)
        # 卷积大小1x1 ，步距1  卷积核：128
        self.conv = BasicConv(in_channels,128,kernel_size=1)           # output[batch, 128, 4, 4]
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,num_class)

    def forward(self,x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)

        return x

class GoogLeNet(paddle.nn.Layer):
    def __init__(self,num_class,aux_logits=True):
        super().__init__()
        self.aux = aux_logits

        self.bn64 = nn.BatchNorm2D(64)
        self.bn192 = nn.BatchNorm2D(192)
        self.bn256 = nn.BatchNorm2D(256)
        self.bn480 = nn.BatchNorm2D(480)
        self.bn512 = nn.BatchNorm2D(512)
        self.bn528 = nn.BatchNorm2D(528)
        self.bn832 = nn.BatchNorm2D(832)
        self.bn1024 = nn.BatchNorm2D(1024)

        self.conv1 = BasicConv(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)  # 特征矩阵的高和宽缩减为原来的一般，padding=3 （224-7+2*3）/2 + 1 = 112.5 向下取整112
        # 计算结果为小数时：  ceil_mode=True : 小数向上取整 ， false：向下取整
        self.max_pool1 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        # LocalResponseNormal 作用不大，没有实现
        # nn.LocalResponseNorm()
        self.conv2 = BasicConv(in_channels=64,out_channels=64,kernel_size=1)
        self.conv3 = BasicConv(in_channels=64,out_channels=192,kernel_size=3,padding=1)
        # nn.LocalResponseNorm()
        self.max_pool2 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.Inception3a = Inception(192,64,128,96,32,16,32)
        self.Inception3b = Inception(256,128,192,128,96,32,64)
        self.max_pool3 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.Inception4a = Inception(480,192,208,96,48,16,64)
        self.Inception4b = Inception(512,160,224,112,64,24,64)
        if aux_logits:
            self.aux_1 = InceptionAux(512,num_class)
            self.aux_2 = InceptionAux(528,num_class)
        self.Inception4c = Inception(512, 128, 256, 128, 64, 24, 64)
        self.Inception4d = Inception(512, 112, 288, 144, 64, 32, 64)
        self.Inception4e = Inception(528, 256, 320, 160, 128, 32, 128)
        self.max_pool4 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.Inception5a = Inception(832,256,320,160,128,32,128)
        self.Inception5b = Inception(832,384,384,192,128,48,128)
        self.avg_pool1 = nn.AvgPool2D(kernel_size=7,stride=1)
        # 自适应平均池化下采样，无论你输入的特征矩阵的高和宽是多大，都能得到指定的高和宽 （1，1）
        # self.avg_pool1 = nn.AdaptiveAvgPool2D((1,1))
        self.dropout1 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024,num_class)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn64(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn192(x)
        x = self.max_pool2(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.bn480(x)
        x = self.max_pool3(x)
        x = self.Inception4a(x)
        if self.training and self.aux:      # eval model lose this layer
            aux1 = self.aux_1(x)
        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        if self.training and self.aux:      # eval model lose this layer
            aux2 = self.aux_2(x)
        x = self.Inception4e(x)
        x = self.bn832(x)
        x = self.max_pool4(x)
        x = self.Inception5a(x)
        x = self.Inception5b(x)
        x = self.bn1024(x)
        x = self.avg_pool1(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc(x)
        if self.training and self.aux:  # eval model lose this layer
            return x, aux1, aux2
        else:
            return x
if __name__ == "__main__":
    model = GoogLeNet(2)
    paddle.summary(model,(None,3,224,224))