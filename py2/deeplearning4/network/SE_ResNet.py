import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bn=False,relu=False):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,dilation,groups)
        self.bn = nn.BatchNorm2D(out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.relu(x) if self.relu else x
        return x

class SEBlock(nn.Layer):
    def __init__(self,in_channels,ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = Conv(in_channels,in_channels//ratio,1,1)
        self.relu = nn.ReLU()
        self.fc2 = Conv(in_channels//ratio,in_channels,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out * x

class BasicBlock(nn.Layer):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv1 = Conv(in_channels,out_channels,3,stride,1,bn=True,relu=True)
        self.conv2 = Conv(out_channels,out_channels,3,1,1,bn=True,relu=False)
        self.Se_block = SEBlock(out_channels)
        self.shortcut = Conv(in_channels,out_channels,1,stride,bn=True,relu=False) if in_channels != out_channels or stride >= 2 else None
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.Se_block(out)
        out = out + self.shortcut(x) if self.shortcut else out + x
        return self.relu(out)

class BottleNeck(nn.Layer):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv1 = Conv(in_channels,out_channels//4,1,1,bn=True,relu=True)
        self.conv2 = Conv(out_channels//4,out_channels//4,3,stride,1,bn=True,relu=True)
        self.conv3 = Conv(out_channels//4,out_channels,1,1,bn=True,relu=False)
        self.se_block = SEBlock(out_channels)
        self.shortcut = Conv(in_channels,out_channels,1,stride,bn=True,relu=False) if in_channels != out_channels or stride >= 2 else None
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se_block(out)
        out = out + self.shortcut(x) if self.shortcut else out + x
        return self.relu(out)

class SE_ResNet(nn.Layer):
    def __init__(self,cfg):
        super().__init__()
        self.num_class = cfg["num_class"]
        structure = cfg["structure"]
        self.structure, block = self._get_structure(structure)
        self.stage1 = nn.Sequential(
            Conv(3,64,7,2,3,bn=True,relu=True),
            nn.MaxPool2D(3,2,1)
        )
        self.stage2 = self._make_layer(self.structure[0][0],64,self.structure[0][1],1,block)
        self.stage3 = self._make_layer(self.structure[1][0],self.structure[0][1],self.structure[1][1],2,block)
        self.stage4 = self._make_layer(self.structure[2][0],self.structure[1][1],self.structure[2][1],2,block)
        self.stage5 = self._make_layer(self.structure[3][0],self.structure[2][1],self.structure[3][1],2,block)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.structure[3][1],self.num_class)

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_layer(self,n,in_channels,out_channels,stride,block):
        layers = []
        for i in range(n):
            layers.append(block(in_channels,out_channels,stride if i == 0 else 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _get_structure(self,structure):
        cfg = {
            "18": [[2,64],[2,128],[2,256],[2,512]],
            "34": [[3,64],[4,128],[6,256],[3,512]],
            "50": [[3,256],[4,512],[6,1024],[3,2048]],
            "101": [[3,256],[4,512],[23,1024],[3,2048]],
            "152": [[3,256],[8,512],[36,1024],[3,2048]]
        }
        block = BottleNeck if structure >= 50 else BasicBlock
        return cfg[str(structure)], block

if __name__ == "__main__":
    cfg = {
        "num_class": 102,
        "structure": 50
    }
    model = SE_ResNet(cfg)
    paddle.flops(model,[1,3,224,224],print_detail=True)
