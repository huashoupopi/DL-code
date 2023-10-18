import paddle
import paddle.nn as nn

class BottleNeck(nn.Layer):
    def __init__(self,in_channels,out_channels,t,stride):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2D(in_channels=in_channels,out_channels=t*in_channels,kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2D(num_features=in_channels*t)
        self.relu1 = nn.ReLU6()
        self.conv2 = nn.Conv2D(in_channels=t*in_channels,
                               out_channels=t*in_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=in_channels*t)
        self.bn2 = nn.BatchNorm2D(t*in_channels)
        self.relu2 = nn.ReLU6()
        self.conv3 = nn.Conv2D(in_channels=t*in_channels,out_channels=out_channels,kernel_size=1,stride=1)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.use_shortcut = True if stride == 1 and in_channels == out_channels else False

    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_shortcut:
            out = out + x
        return out

class MobileNetv2(nn.Layer):
    def __init__(self,num_class):
        super().__init__()
        #    t,c,n,s
        cfg = [
            [1,16,1,1],
            [6,24,2,2],
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,3,1],
        ]
        self.conv1 = nn.Conv2D(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu = nn.ReLU6()
        self.stage2 = self._make_layer(cfg,32)
        self.conv2 = nn.Conv2D(in_channels=320,out_channels=1280,kernel_size=1,stride=1)
        self.bn2 = nn.BatchNorm2D(1280)
        self.relu2 = nn.ReLU6()
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv3 = nn.Conv2D(in_channels=1280,out_channels=num_class,kernel_size=1,stride=1)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stage2(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

    @staticmethod
    def _make_layer(layer_parameters,in_channels):
        layers = []
        for cfg in layer_parameters:
            for i in range(cfg[2]):
                stride = cfg[3] if i == 0 else 1
                layers.append(BottleNeck(in_channels=in_channels,out_channels=cfg[1],t=cfg[0],stride=stride))
                in_channels = cfg[1]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    model = MobileNetv2(102)
    paddle.flops(model,[1,3,224,224],print_detail=True)
    nn.MaxPool2D()