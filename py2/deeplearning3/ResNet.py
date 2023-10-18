import paddle
import paddle.nn as nn


class Conv(nn.Layer):
    def __init__(self, in_c, out_c, kernel, stride, bn=False, act=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(in_c, out_c, kernel, stride, padding=kernel//2)
        self.bn = nn.BatchNorm2D(out_c) if bn else False
        self.relu = nn.ReLU() if act else False

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.relu(x) if self.relu else x
        return x

class BasicBlock(nn.Layer):
    def __init__(self, in_c, out_c, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv(in_c, in_c, 3, stride, bn=True, act=True)
        self.conv2 = Conv(in_c, out_c, 3, 1, bn=True, act=False)
        self.down_conv = Conv(in_c, out_c, 1, stride, bn=True, act=False) if stride >= 2 or in_c != out_c else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.down_conv(x) if self.down_conv else out + x
        return self.relu(out)

class BottleNeck(nn.Layer):
    def __init__(self, in_c, out_c, stride):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv(in_c, out_c//4, 1, 1, bn=True, act=True)
        self.conv2 = Conv(out_c//4, out_c//4, 3, stride, bn=True, act=True)
        self.conv3 = Conv(out_c//4, out_c, 1, 1, bn=True, act=False)
        self.down_conv = Conv(in_c, out_c, 1, stride, bn=True, act=False) if stride >= 2 or in_c != out_c else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.down_conv(x) if self.down_conv else out + x
        return self.relu(out)


class ResNet(nn.Layer):
    def __init__(self, struct, num_class=1000):
        super(ResNet, self).__init__()
        self.struct = struct
        layer_info = self._cfg(struct)
        self.stage1 = nn.Sequential(
            Conv(3, 64, 7, 2, True, True),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = self._make_layer(layer_info[0][0], 64, layer_info[0][1], 1)
        self.stage3 = self._make_layer(layer_info[1][0], layer_info[0][1], layer_info[1][1], 2)
        self.stage4 = self._make_layer(layer_info[2][0], layer_info[1][1], layer_info[2][1], 2)
        self.stage5 = self._make_layer(layer_info[3][0], layer_info[2][1], layer_info[3][1], 2)

        self.pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(layer_info[3][1], num_class)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.fc(self.flatten(self.pool(x)))
        return x

    def _make_layer(self, n, in_c, out_c, stride):
        layers = []
        for i in range(n):
            if self.struct < 50:
                layers.append(BasicBlock(in_c, out_c, stride if i == 0 else 1))
            else:
                layers.append(BottleNeck(in_c, out_c, stride if i == 0 else 1))
            in_c = out_c
        return nn.Sequential(*layers)

    def _cfg(self, struct):
        cfg = {18: [[2, 64], [2, 128], [2, 256], [2, 512]],
               34: [[3, 64], [4, 128], [6, 256], [3, 512]],
               50: [[3, 256], [4, 512], [6, 1024], [3, 2048]],
               101: [[3, 256], [4, 512], [23, 1024], [3, 2048]],
               152: [[3, 256], [8, 512], [36, 1024], [3, 2048]]}
        return cfg[struct]