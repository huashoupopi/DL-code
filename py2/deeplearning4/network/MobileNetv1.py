import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,dilation=1,groups=1):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,dilation,groups=groups)
        self.bn = nn.BatchNorm2D(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DwPw(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,groups=1):
        super().__init__()
        self.dw = Conv(in_channels,in_channels,kernel_size,stride,padding=padding,groups=in_channels)
        self.pw = Conv(in_channels,out_channels,1,1)

    def forward(self,x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class MobileNetv1(nn.Layer):
    def __init__(self,num_class):
        super().__init__()
        self.num_class = num_class
        self.stage1 = Conv(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.stage2 = nn.Sequential(
            DwPw(32,64,3,1,1),
            DwPw(64,128,3,2,1),
            DwPw(128,128,3,1,1),
            DwPw(128,256,3,2,1),
            DwPw(256,256,3,1,1),
            DwPw(256,512,3,2,1),
        )
        self.stage3 = DwPw(512,512,3,1,1)
        self.stage4 = nn.Sequential(
            DwPw(512,1024,3,2,1),
            DwPw(1024,1024,3,2,1),
        )
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1024,out_features=self.num_class)

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        for i in range(5):
            x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = MobileNetv1(102)
    paddle.flops(model,[1,3,224,224],print_detail=True)


