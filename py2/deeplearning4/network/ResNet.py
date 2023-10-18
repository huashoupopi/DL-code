import paddle
from paddle import nn

class Conv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bn=False, relu=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias_attr=bias)
        self.bn = nn.BatchNorm(num_channels=out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.relu(x) if self.relu else x
        return x


# ResNet based on Conv
class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bn=True,
                          relu=True)
        self.conv2 = Conv(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bn=True,
                          relu=False)
        self.shortcut = Conv(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=stride,
                             padding=0,
                             bn=True) if in_channels != out_channels or stride >= 2 else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out = out + shortcut
        out = self.relu(out)
        return out


class Bottleneck(nn.Layer):
    def __init__(self, in_channels, out_channels, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(in_channels=in_channels,
                          out_channels=out_channels // 4,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bn=True,
                          relu=True)
        self.conv2 = Conv(in_channels=out_channels // 4,
                          out_channels=out_channels // 4,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bn=True,
                          relu=True)
        self.conv3 = Conv(in_channels=out_channels // 4,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bn=True,
                          relu=False)
        self.shortcut = Conv(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=stride,
                             padding=0,
                             bn=True) if in_channels != out_channels or stride >= 2 else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out = out + shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(self, configs):
        super(ResNet, self).__init__()
        structure = configs["structure"]
        self.structure, block = self.get_structure(structure)
        self.num_classes = configs["num_classes"]
        self.stage1 = Conv(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2D(2)
        self.stage2 = self._make_layer(self.structure[0][0], 64, self.structure[0][1], 1, block)
        self.stage3 = self._make_layer(self.structure[1][0], self.structure[0][1], self.structure[1][1], 2, block)
        self.stage4 = self._make_layer(self.structure[2][0], self.structure[1][1], self.structure[2][1], 2, block)
        self.stage5 = self._make_layer(self.structure[3][0], self.structure[2][1], self.structure[3][1], 2, block)
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.structure[3][1], self.num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    @staticmethod
    def _make_layer(n, in_channels, out_channels, stride, block):
        layers = []
        for i in range(n):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, stride=stride if i == 0 else 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def get_structure(structure):
        cfg = {
            "18": [[2, 64], [2, 128], [2, 256], [2, 512]],
            "34": [[3, 64], [4, 128], [6, 256], [3, 512]],
            "50": [[3, 256], [4, 512], [6, 1024], [3, 2048]],
            "101": [[3, 256], [4, 512], [23, 1024], [3, 2048]],
            "152": [[3, 256], [8, 512], [36, 1024], [3, 2048]],
        }
        block = Bottleneck if structure >= 50 else BasicBlock
        return cfg[str(structure)], block


if __name__ == "__main__":
    config = {
        "structure": 18,
        "num_classes": 1000
    }
    model = ResNet(config)
    paddle.flops(model, [1, 3, 224, 224], print_detail=True)