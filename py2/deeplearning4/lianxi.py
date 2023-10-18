import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,stride=1,groups=1,dilation=1,act=None):
        super().__init__()
        self.conv = nn.Conv2D(in_c,out_c,kernel_size,stride,kernel_size//2,dilation,groups)
        self.bn = nn.BatchNorm(out_c,act=act)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class rSoftmax(nn.Layer):
    def __init__(self,radix,cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.softmax = nn.Softmax(axis=1)

    def forward(self,x):
        cardinality = self.cardinality
        radix = self.radix
        b, r, h, w = x.shape
        if self.radix > 1:
            x = paddle.reshape(x,[b,cardinality,radix,int(r*h*w/cardinality/radix)])
            x = paddle.transpose(x,[0,2,1,3])
            x = self.softmax(x)
            x = paddle.reshape(x,[b,r*h*w,1,1])
        else:
            x = nn.functional.sigmoid(x)
        return x

class SplatConv(nn.Layer):
    def __init__(self,in_c,c,kernel_size,stride=1,dilation=1,groups=1,radix=2,reduction_factor=4):
        super().__init__()
        self.radix = radix
        self.conv1 = Conv(in_c,c*radix,kernel_size,stride,groups*radix,act=nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        inter_c = int(max(in_c*radix//reduction_factor,32))

        self.conv2 = Conv(c,inter_c,1,1,groups,act=nn.ReLU())
        self.conv3 = nn.Conv2D(inter_c,c*radix,1,1,groups)
        self.rSoftmax = rSoftmax(radix,groups)

    def forward(self,x):
        x = self.conv1(x)

        if self.radix > 1:
            splited = paddle.split(x,num_or_sections=self.radix,axis=1)
            gap = paddle.add_n(splited)
        else:
            gap = x
        gap = self.avg_pool(gap)
        gap = self.conv2(gap)
        atten = self.conv3(gap)
        atten = self.rSoftmax(atten)
        if self.radix > 1:
            attens = paddle.split(atten,num_or_sections=self.radix,axis=1)
            y = paddle.add_n([paddle.multiply(split,att) for (att,split) in zip(attens,splited)])
        else:
            y = paddle.multiply(x,atten)
        return y


class BottleBlock(nn.Layer):
    def __init__(self,in_c,c,stride=1,radix=1,cardinality=1,bottleneck_width=64,
                 avd=False,
                 avd_first=False,
                 is_first = False,
                 dilation=1,
                 avg_down=False
                 ):
        super().__init__()
        self.in_c = in_c
        self.c = c
        self.stride = stride
        self.radix = radix
        self.cardinality = cardinality
        self.avd = avd
        self.avd_first = avd_first
        self.dilation = dilation
        self.is_first = is_first
        self.avg_down = avg_down

        group_width = int(c*(bottleneck_width/64))*cardinality
        self.conv1 = Conv(self.in_c,group_width,1,1,1,act=nn.ReLU())
        if avd and avd_first and (stride > 1 or is_first):
            self.avg_pool_1 = nn.AvgPool2D(3,stride,1)

        if radix > 2:
            self.conv2 = SplatConv(group_width,group_width,3,1,groups=cardinality,radix=radix)
        else:
            self.conv2 = Conv(group_width,group_width,3,1,cardinality,act=nn.ReLU())

        if avd and avd_first == False and (stride > 1 or is_first):
            self.avg_pool_2 = nn.AvgPool2D(kernel_size=3, stride=stride, padding=1)

        self.conv3 = Conv(group_width,c*4,1,1,1)

        if stride != 1 or self.in_c != self.c*4:
            if avg_down:
                if dilation == 1:
                    self.avg_pool_3 = nn.AvgPool2D(
                        kernel_size=stride, stride=stride, padding=0)
                else:
                    self.avg_pool_3 = nn.AvgPool2D(
                        kernel_size=1, stride=1, padding=0, ceil_mode=True)
                self.conv4 = nn.Conv2D(self.in_c,c*4,1,1)

            else:
                self.conv4 = nn.Conv2D(self.in_c, c * 4, 1, stride)
            self.bn = nn.BatchNorm(c*4,act=None)



    def forward(self,x):
        short = x

        x = self.conv1(x)
        if self.avd and self.avd_first and (self.stride > 1 or self.is_first):
            x = self.avg_pool_1(x)
        x = self.conv2(x)
        if self.avd and self.avd_first == False and (self.stride>1 or self.is_first):
            x = self.avg_pool_2(x)
        x = self.conv3(x)

        if self.stride != 1 or self.in_c != self.c*4:
            if self.avg_down:
                short = self.avg_pool_3(short)
            short = self.conv4(short)
            short = self.bn(short)

        y = x + short
        y = nn.functional.relu(y)
        return y


class ResNeStLayer(nn.Layer):
    def __init__(self,):
        pass