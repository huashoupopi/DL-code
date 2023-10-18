import paddle.nn as nn
import paddle
from functools import reduce

class SKConv(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        '''
                :param in_channels:  输入通道维度
                :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
                :param stride:  步长，默认为1
                :param M:  分支数
                :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
                :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.LayerList()
        for i in range(M):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32),
                    nn.BatchNorm2D(out_channels),
                    nn.ReLU()
                )
                # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积 G=32
            )
        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Sequential(
            nn.Conv2D(out_channels, d, 1,1),
            nn.BatchNorm2D(d),
            nn.ReLU()
        )
        self.fc2 = nn.Conv2D(d, out_channels * M, 1,1)
        self.softmax = nn.Softmax(axis=1)     # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size = input.shape[0]
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = paddle.reshape(a_b, [batch_size, self.M, self.out_channels, -1])
        a_b = self.softmax(a_b)
        a_b = paddle.chunk(a_b, self.M, axis=1)   #chunk方法，将tensor按照指定维度切分成 几个tensor
        a_b = [paddle.reshape(x, [batch_size, self.out_channels, 1, 1]) for x in a_b]
        V = [output[i] * a_b[i] for i in range(self.M)]
        V = reduce(lambda x, y: x + y, V)
        return V

class SKBlock(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SKBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(inplanes, planes, 1),
            nn.BatchNorm2D(planes),
            nn.ReLU()
        )
        self.conv2 = SKConv(planes, planes, stride)
        self.conv3 = nn.Sequential(
            nn.Conv2D(planes, planes * self.expansion, 1),
            nn.BatchNorm2D(planes * self.expansion)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, input):
        shortcut = input
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.downsample is not None:
            shortcut = self.downsample(input)
        output += shortcut
        return self.relu(output)

class SKNet(nn.Layer):
    def __init__(self, nums_class=1000, block=SKBlock, nums_block_list=[3, 4, 6, 3]):
        super(SKNet, self).__init__()
        self.inplanes = 64
        self.conv = nn.Sequential(
            nn.Conv2D(3, 64, 7, 2, 3, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2D(3, 2, 1)
        self.layer1 = self._make_layer(block, 128, nums_block_list[0], stride=1)
        self.layer2 = self._make_layer(block, 256, nums_block_list[1], stride=2)
        self.layer3 = self._make_layer(block, 512, nums_block_list[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, nums_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024 * block.expansion, nums_class)
        # self.softmax = nn.Softmax(axis=-1)

    def forward(self, input):
        output = self.conv(input)
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = self.flatten(output)
        output = self.fc(output)
        # output = self.softmax(output)
        return output

    def _make_layer(self, block, planes, nums_block, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, 1, stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, nums_block):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

def SKNet50(nums_class=1000):
    return SKNet(nums_class, SKBlock, [3, 4, 6, 3])

def SKNet101(nums_class=1000):
    return SKNet(nums_class, SKBlock, [3, 4, 23, 3])

if __name__ == '__main__':
    model = SKNet50(102)
    paddle.flops(model,[1,3,224,224],print_detail=True)

