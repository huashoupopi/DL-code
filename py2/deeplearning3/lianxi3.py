import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,stride,bn=True,act=False):
        super().__init__()
        self.conv = nn.Conv2D(in_c,out_c,kernel_size,stride,padding=kernel_size//2)
        self.bn = nn.BatchNorm2D(out_c) if bn else False
        self.relu = nn.ReLU() if act else False

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.relu(x) if self.relu else x
        return x

class BasicBlock(nn.Layer):
    def __init__(self,in_c,out_c,stride):
        self.conv1 = Conv(in_c,in_c,3,stride,True,True)
        self.conv2 = Conv(in_c,out_c,3,1,True,False)
        self.down_conv = Conv(in_c,out_c,1,stride,True,False) if in_c != out_c or stride > 2 else None
        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output+self.down_conv(x) if self.down_conv else output + x
        return self.relu(output)

class BottomNeck(nn.Layer):
    def __init__(self,in_c,out_c,stride):
        super().__init__()
        self.conv1 = Conv(in_c,out_c//4,1,1,True,True)
        self.conv2 = Conv(out_c//4,out_c//4,3,stride,True,True)
        self.conv3 = Conv(out_c//4,out_c,1,1,True,False)
        self.down_conv = Conv(in_c,out_c,1,stride,True,False) if in_c != out_c or stride > 2 else None
        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output + self.down_conv(x) if self.down_conv else output + x
        return self.relu(output)

class ResNet(nn.Layer):
    def __init__(self,struct,num_class=2):
        super().__init__()
        self.struct = struct
        layer_info = self._make_cfg(struct)
        self.stage1 = nn.Sequential(
            Conv(3,64,7,2,True,True),
            nn.MaxPool2D(3,2,1)
        )
        self.stage2 = self._make_layers(layer_info[0][0],64,layer_info[0][1],1)
        self.stage3 = self._make_layers(layer_info[1][0],layer_info[0][1],layer_info[1][1],2)
        self.stage4 = self._make_layers(layer_info[2][0],layer_info[1][1],layer_info[2][1],2)
        self.stage5 = self._make_layers(layer_info[3][0],layer_info[2][1],layer_info[3][1],2)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(layer_info[3][1],num_class)

    def _make_layers(self,n,in_c,out_c,stride):
        layers = []
        for i in range(n):
            if self.struct<50:
                layers.append(BasicBlock(in_c,out_c,stride if i == 0 else 1))
            else:
                layers.append(BottomNeck(in_c,out_c,stride if i == 0 else 1))
            in_c = out_c
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_cfg(self,struct):
        cfg = {
            18: [[2, 64], [2, 128], [2, 256], [2, 512]],
            34: [[3, 64], [4, 128], [6, 256], [3, 512]],
            50: [[3, 256], [4, 512], [6, 1024], [3, 2048]],
            101: [[3, 256], [4, 512], [23, 1024], [3, 2048]],
            152: [[3, 256], [8, 512], [36, 1024], [3, 2048]]
        }
        return cfg[struct]

if __name__ == "__main__":
    model = ResNet(50)
    paddle.summary(model,(None,3,224,224))