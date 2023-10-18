import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,dilation=1,groups=1,bn=True,relu=False):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,dilation,groups)
        self.bn = nn.BatchNorm2D(num_features=out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.relu(x) if self.relu else x
        return x

class Shuffle(nn.Layer):
    def __init__(self,groups):
        super().__init__()
        self.groups = groups

    def forward(self,x):
        """
        shuffle: [N,C,H,W]--->[N,g,C//g,H,W]--->[N,C//g,g,H,W]--->[N,C,H,W]
        """
        num, channels, height, width = x.shape
        x = x.reshape([num,self.groups,channels // self.groups,height,width])       # 将通道按照group分组
        x = x.transpose((0,2,1,3,4))                                                # 交换分组和通道的位置
        x = x.reshape([num,channels,height,width])                                  # 恢复为原始形状
        return x

class Bottleblock(nn.Layer):
    def __init__(self,in_channels,out_channels,stride,groups):
        super().__init__()
        self.stride = stride
        if in_channels == 24:
            groups = 1
        else:
            groups = groups
        self.conv1 = Conv(in_channels,out_channels//4,1,1,groups=groups,bn=True,relu=True)
        self.shuffle = Shuffle(groups)
        self.conv2 = Conv(out_channels//4,out_channels//4,3,stride,1,groups=out_channels//4,bn=True,relu=False)
        self.conv3 = Conv(out_channels//4,out_channels,1,1,groups=groups,bn=True,relu=False)
        if self.stride == 2:
            self.shortcut = nn.AvgPool2D(3,2,1)
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.shuffle(out)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride == 2:
            out = paddle.concat([out,self.shortcut(x)],1)           # 在通道维度上进行拼接
        else:
            out = out + self.shortcut(x)
        return self.relu(out)

class ShuffleNet(nn.Layer):
    def __init__(self,num_class,groups):
        super().__init__()
        self.num_class = num_class
        self.groups = groups
        self.in_channels = 24
        cfg = {
            "1": [144,288,576],
            "2": [200,400,800],
            "3": [240,480,960],
            "4": [272,544,1088],
            "8": [384,768,1536]
        }
        self.deep = [4,8,4]
        self.out_channels = cfg[str(self.groups)]
        self.stage1 = nn.Sequential(
            Conv(in_channels=3,out_channels=24,kernel_size=3,stride=2,padding=1,bn=True,relu=True),
            nn.MaxPool2D(3,2,1)
        )
        self.stage2 = self._make_layer(self.deep[0],self.out_channels[0],self.groups)
        self.stage3 = self._make_layer(self.deep[1],self.out_channels[1],self.groups)
        self.stage4 = self._make_layer(self.deep[2],self.out_channels[2],self.groups)
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.out_channels[2],self.num_class)

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_layer(self,n,out_channels,groups):
        layers = []
        for i in range(n):
            if i == 0:
                stride = 2
                concat_channels = self.in_channels
            else:
                stride = 1
                concat_channels = 0
            layers.append(Bottleblock(self.in_channels,out_channels-concat_channels,stride,groups))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

if __name__ == "__main__":
    model = ShuffleNet(102,3)
    paddle.flops(model,[1,3,224,224],print_detail=True)

