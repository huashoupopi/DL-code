# import paddle
# import paddle.nn as nn
#
# class Conv(nn.Layer):
#     def __init__(self, in_c, out_c, kernel_size, stride, groups, bn=False, act=False):
#         super().__init__()
#         self.conv = nn.Conv2D(in_c, out_c, kernel_size, stride, kernel_size // 2, groups=groups)  # 调用一个卷积层
#         self.bn = nn.BatchNorm2D(out_c) if bn else None  # BN层
#         self.act = act if act else None  # 激活函数层
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x) if self.bn else x
#         x = self.act(x) if self.act else x
#         return x
#
# class ShuffleBlock(nn.Layer):
#     def __init__(self, in_c, out_c, groups):
#         """ShuffleNet的基础模块
#
#         Args:
#             in_c (int): 输入的通道维度
#             out_c (int): 输出的通道维度
#             groups (int): 网络的分组数
#         """
#         super().__init__()
#         self.groups = groups  # 在channels_shuffle的时候分组的情况
#         self.split = (in_c != out_c)  # 判断是否需要对特征split
#         self.left_branch = self.make_left_brach(in_c, out_c) if self.split else None
#         self.right_branch = self.make_right_brach(in_c, out_c)
#
#     def forward(self, x):
#         (left, right) = (x, x) if self.split else paddle.split(x, num_or_sections=2, axis=1)
#         left = self.left_branch(left) if self.left_branch else left
#         right = self.right_branch(right)
#         x = paddle.concat([left, right], axis=1)
#         x = self.channel_shuffle(x)
#         return x
#
#     def make_left_brach(self, in_c, out_c):
#         """生成模型的左分支，即下采样模块
#
#         Args:
#             in_c (int): 输入的通道维度
#             out_c (int): 输出的通道维度
#
#         Returns:
#             (nn.Layer): 一个组网
#         """
#         return nn.Sequential(
#             Conv(in_c, in_c, kernel_size=3, stride=2, groups=in_c, bn=True),
#             Conv(in_c, out_c // 2, kernel_size=1, bn=True, act=nn.ReLU()),
#         )
#
#     def make_right_brach(self, in_c, out_c):
#         in_c = in_c if self.split else in_c // 2
#         out_c = out_c // 2
#         return nn.Sequential(
#             Conv(in_c, out_c, kernel_size=1, bn=True, act=nn.ReLU()),
#             Conv(out_c, out_c, kernel_size=3, stride=2 if self.split else 1, groups=out_c, bn=True),
#             Conv(out_c, out_c, kernel_size=1, bn=True, act=nn.ReLU())
#         )
#
#     def channel_shuffle(self, x):
#         """对网络通道打乱
#
#         Args:
#             x (Paddle.Tensor): 特征向量，一般来说通道为[B, C, H, W];即Batch数，C通道数， H特征高，W特征宽
#
#         Returns:
#             (Paddle.Tensor): 打乱通道的特征
#         """
#         batch_size, channels, height, width = x.shape
#         x = paddle.reshape(x, (batch_size, self.groups, channels // self.groups, height, width))
#         x = paddle.transpose(x, (0, 2, 1, 3, 4))
#         x = paddle.reshape(x, (batch_size, channels, height, width))
#         return x
#
# class ShuffleNetV2(nn.Layer):
#     def __init__(self, struct=1, groups=4, num_class=1000):
#         super(ShuffleNetV2, self).__init__()
#         cfg = {0.5: 48, 1: 116, 1.5: 176, 2: 244}                    # 不同的配置参数
#         self.groups = groups                                         # 在channels_shuffle时的组数
#         # Stage1
#         self.conv1 = Conv(3, 24, kernel_size=3, stride=2, bn=True, act=nn.ReLU())
#         self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
#         self.stage2 = self.make_stage(4, 24, cfg[struct])
#         self.stage3 = self.make_stage(8, cfg[struct], cfg[struct]*2)
#         self.stage4 = self.make_stage(4, cfg[struct]*2, cfg[struct]*4)
#         hidden_node = 1024 if struct < 2 else 2048
#         self.conv5 = Conv(cfg[struct]*4, hidden_node, kernel_size=1, bn=True, act=nn.ReLU())
#         self.globalpool = nn.AdaptiveAvgPool2D(1)
#         self.flatten = nn.Flatten()                                  # 将特征降维
#         self.linear = nn.Linear(hidden_node, num_class)
#
#     def forward(self, x):
#         x = self.maxpool(self.conv1(x))
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = self.conv5(x)
#         x = self.globalpool(x)
#         x = self.linear(self.flatten(x))
#         return x
#
#     def make_stage(self, n, in_c, out_c):
#         blocks = []
#         for i in range(n):
#             blocks.append(ShuffleBlock(in_c, out_c, self.groups))
#             in_c = out_c
#         return nn.Sequential(*blocks)
#
# if __name__ == "__main__":
#     model = ShuffleNetV2(1, 4)
#     paddle.summary(model, (None, 3, 224, 224))
import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,groups=1,bn=False,act=None):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,groups=groups)
        self.bn = nn.BatchNorm2D(out_channels) if bn else None
        self.act = act if act else None

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x

class Shuffle(nn.Layer):
    def __init__(self,groups):
        super().__init__()
        self.groups = groups

    def forward(self,x):
        num, channels, height, width = x.shape
        x = x.reshape([num,self.groups,channels//self.groups,height,width])
        x = x.transpose([0,2,1,3,4])
        x = x.reshape([num,channels,height,width])
        return x

class ShuffleBlock(nn.Layer):
    def __init__(self,in_c,out_c,groups):
        super().__init__()
        self.groups = groups
        self.spilt = (in_c == out_c)
        self.left_brach = self.make_left_brach(in_c,out_c) if self.spilt is False else None
        self.right_brach = self.make_right_brach(in_c,out_c)
        self.shuffle = Shuffle(self.groups)

    def forward(self,x):
        (left, right) = paddle.split(x,num_or_sections=2,axis=1) if self.spilt else (x, x)
        left = self.left_brach(left) if self.left_brach else left
        right = self.right_brach(right)
        x = paddle.concat([left,right],axis=1)
        x = self.shuffle(x)
        return x

    def make_left_brach(self,in_c,out_c):
        return nn.Sequential(
            Conv(in_c,in_c,3,2,in_c,bn=True),
            Conv(in_c,out_c//2,1,1,1,bn=True,act=nn.ReLU())
        )

    def make_right_brach(self,in_c,out_c):
        in_c = in_c // 2 if self.spilt else in_c
        out_c = out_c // 2
        return nn.Sequential(
            Conv(in_c,out_c,1,1,1,bn=True,act=nn.ReLU()),
            Conv(out_c,out_c,3,1 if self.spilt else 2,out_c,bn=True),
            Conv(out_c,out_c,1,1,1,bn=True,act=nn.ReLU())
        )

class ShuffleNetV2(nn.Layer):
    def __init__(self,num_class,structure=1,groups=2):
        super().__init__()
        self.num_class = num_class
        cfg = {0.5: 48, 1: 116, 1.5: 176, 2: 244}
        self.groups = groups
        hidden_nodes = 1024 if structure < 2 else 2048
        self.stage1 = nn.Sequential(
            Conv(3,24,3,2,1,bn=True,act=nn.ReLU()),
            nn.MaxPool2D(3,2,1)
        )
        self.stage2 = self._make_layer(4,24,cfg[structure],self.groups)
        self.stage3 = self._make_layer(8,cfg[structure],cfg[structure]*2,self.groups)
        self.stage4 = self._make_layer(4,cfg[structure]*2,cfg[structure]*4,self.groups)
        self.conv5 = Conv(cfg[structure]*4,hidden_nodes,1,1,1,bn=True,act=nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_nodes,num_class)

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_layer(self,n,in_c,out_c,groups):
        layers = []
        for i in range(n):
            layers.append(ShuffleBlock(in_c,out_c,groups))
            in_c = out_c
        return nn.Sequential(*layers)

if __name__ == "__main__":
    model = ShuffleNetV2(102,2,4)
    paddle.flops(model,[1,3,224,224],print_detail=True)