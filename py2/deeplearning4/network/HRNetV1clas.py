import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,stride=1,groups=1,act=None):
        super().__init__()
        self.conv = nn.Conv2D(in_c,out_c,kernel_size,stride,kernel_size // 2,groups=groups)
        self.norm = nn.BatchNorm2D(out_c)
        self.act = nn.ReLU() if act is None else act

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class BottleNeck(nn.Layer):
    def __init__(self,in_c,out_c,has_se,stride=1,downsample=False):
        super().__init__()
        self.has_se = has_se
        self.downsample = downsample
        self.conv1 = Conv(in_c,out_c//4,1)
        self.conv2 = Conv(out_c//4,out_c//4,3,stride)
        self.conv3 = Conv(out_c//4,out_c,1,act=nn.Identity())
        if self.downsample:
            self.down_sample = Conv(in_c,out_c,1,act=nn.Identity())
        if self.has_se:
            self.se = SqueezeExcitation(out_c)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        shortcut = self.down_sample(x) if self.downsample else x
        if self.has_se:
            out = self.se(out)
        out = out + shortcut
        out = self.relu(out)
        return out

class BasicBlock(nn.Layer):
    def __init__(self,in_c,out_c,has_se=False):
        super().__init__()
        self.has_se = has_se
        self.conv1 = Conv(in_c,out_c,3,1)
        self.conv2 = Conv(out_c,out_c,3,1,act=nn.Identity())
        if self.has_se:
            self.se = SqueezeExcitation(out_c)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.has_se:
            out = self.se(out)
        shortcut = x
        out = out + shortcut
        out = self.relu(out)
        return out

class SqueezeExcitation(nn.Layer):
    def __init__(self,in_c,ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(in_c,in_c//ratio,1,1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(in_c//ratio,in_c,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out * x
        return out

class Stage(nn.Layer):
    def __init__(self,num_blocks,out_c,has_se=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.stage = nn.LayerList()
        for i in range(self.num_blocks):
            self.stage.append(
                HRBlock(out_c,has_se)
            )

    def forward(self,x):
        x = x
        for idx in range(self.num_blocks):
            x = self.stage[idx](x)
        return x

class HRBlock(nn.Layer):
    def __init__(self,out_c,has_se=False):
        super().__init__()
        self.block_list = nn.LayerList()
        for i in range(len(out_c)):
            self.block_list.append(nn.Sequential(
                *[BasicBlock(out_c[i],out_c[i],has_se) for j in range(4)]
            ))

        self.fuse_func = FuseLayers(out_c,out_c)

    def forward(self,x):
        out = []
        for idx, xi in enumerate(x):
            basic_list = self.block_list[idx]
            for basic in basic_list:
                xi = basic(xi)
            out.append(xi)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Layer):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.in_c = in_c
        self.len_in_c = len(in_c)

        self.func_list = nn.LayerList()
        self.relu = nn.ReLU()
        for i in range(len(in_c)):
            for j in range(len(in_c)):
                if j > i:
                    self.func_list.append(
                        nn.Sequential(Conv(in_c[j],in_c[i],1,1,act=nn.Identity()),
                                      nn.Upsample(scale_factor=2**(j-i)))
                    )
                elif j < i:
                    for k in range(i-j):
                        if k == i-j-1:
                            self.func_list.append(
                                Conv(in_c[j],out_c[i],3,2,act=nn.Identity())
                            )

                        else:
                            self.func_list.append(Conv(in_c[j],out_c[j],3,2))

    def forward(self,x):
        out_list = []
        func_idx = 0
        for i in range(len(self.in_c)):
            out = x[i]
            for j in range(len(self.in_c)):
                if j > i:
                    xj = self.func_list[func_idx](x[j])
                    func_idx += 1
                    out = out + xj
                elif j < i:
                    xj = x[j]
                    for k in range(i-j):
                        xj = self.func_list[func_idx](xj)
                        func_idx += 1
                    out = xj + out

            out = self.relu(out)
            out_list.append(out)
        return out_list

class LastClsOut(nn.Layer):
    def __init__(self,out_c,has_se,num_out_c=[32,64,128,256]):
        super().__init__()
        self.func_list = nn.LayerList()
        for idx in range(len(out_c)):
            self.func_list.append(
                BottleNeck(out_c[idx],num_out_c[idx]*4,has_se,downsample=True)
            )

    def forward(self,x):
        out = []
        for idx, xi in enumerate(x):
            xi = self.func_list[idx](xi)
            out.append(xi)
        return out


class HRNet(nn.Layer):
    def __init__(self,c=18,has_se=False,num_classes=102):
        super().__init__()
        self.c = c
        self.has_se = has_se
        self.num_classes = num_classes

        channels_2 = [self.c,self.c*2]
        channels_3 = [self.c,self.c*2,self.c*4]
        channels_4 = [self.c,self.c*2,self.c*4,self.c*8]

        self.conv1 = Conv(3,64,3,2)
        self.conv2 = Conv(64,64,3,2)
        self.layer1 = nn.Sequential(
            *[BottleNeck(in_c=64 if i == 0 else 256,out_c=256,has_se=has_se,stride=1,downsample=True if i == 0 else False) for i in range(4)]
        )

        self.tr1_1 = Conv(256,c,3,act=nn.Identity())
        self.tr1_2 = Conv(256,c*2,3,2,act=nn.Identity())

        self.st2 = Stage(1,channels_2,self.has_se)
        self.tr2 = Conv(c*2,c*4,3,2,1,act=nn.Identity())
        self.st3 = Stage(4,channels_3,self.has_se)
        self.tr3 = Conv(c*4,c*8,3,2,1,act=nn.Identity())
        self.st4 = Stage(3,channels_4,self.has_se)

        num_out_c = [32,64,128,256]
        self.last_cls = LastClsOut(channels_4,self.has_se,num_out_c)
        last_num_out = [256,512,1024]
        self.cls_head_conv = nn.LayerList()
        for idx in range(3):
            self.cls_head_conv.append(
                Conv(num_out_c[idx]*4,last_num_out[idx],3,2,act=nn.Identity())
            )
        self.conv_last = Conv(1024,2048,1,1,act=nn.Identity())
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048,self.num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        tr1_1 = self.tr1_1(x)
        tr1_2 = self.tr1_2(x)
        x = self.st2([tr1_1,tr1_2])
        tr2 = self.tr2(x[-1])
        x.append(tr2)
        x = self.st3(x)
        tr3 = self.tr3(x[-1])
        x.append(tr3)
        x = self.st4(x)
        x = self.last_cls(x)
        y = x[0]
        for idx in range(3):
            y = x[idx+1] + self.cls_head_conv[idx](y)
        y = self.conv_last(y)
        y = self.avg_pool(y)
        y = self.flatten(y)
        y = self.fc(y)
        return y
# C: 18 30 32 40 44 48 60 64

if __name__ == "__main__":
    model = HRNet(32,num_classes=102)
    paddle.flops(model,[1,3,224,224],print_detail=True)
