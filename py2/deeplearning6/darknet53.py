import paddle
import paddle.nn.functional as F
import numpy as np
import paddle.nn as nn

#YOLO-V3 骨干网络架构 DarkNet53的实现

class ConvBnLayer(paddle.nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,stride,groups=1,dilation=1,act=nn.LeakyReLU()):
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_c,out_c,kernel_size,stride,kernel_size//2,dilation,groups)
        self.bn = nn.BatchNorm2D(out_c)
        self.act = act if act is not None else nn.Identity()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DownSample(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size=3,stride=2):
        super().__init__()
        self.conv = ConvBnLayer(in_c,out_c,kernel_size,stride=stride)

    def forward(self,x):
        x = self.conv(x)
        return x

class BasicBlock(nn.Layer):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv1 = ConvBnLayer(in_c,out_c,1,1)
        self.conv2 = ConvBnLayer(out_c,out_c*2,3,1)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return out

class LayerWarp(nn.Layer):
    def __init__(self,in_c,out_c,n):
        super().__init__()
        self.layer_list = []
        for i in range(n):
            self.layer_list.append(BasicBlock(in_c if i == 0 else out_c*2,out_c))
        self.stage = nn.Sequential(*self.layer_list)

    def forward(self,x):
        x = self.stage(x)
        return x

# DarkNet 每组残差块的个数，来自DarkNet的网络结构图
DarkNet_cfg = {53: [1, 2, 8, 8, 4]}

class DarkNet53_conv_body(nn.Layer):
    def __init__(self):
        super().__init__()
        self.stages = DarkNet_cfg[53]
        self.conv1 = ConvBnLayer(3,32,3,1)
        self.conv2 = DownSample(32,32*2,3)

        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer("stage_%d"%(i),LayerWarp(32*(2**(i+1)),32*(2**i),stage))
            self.darknet53_conv_block_list.append(conv_block)
        for i in range(len(self.stages)-1):
            downsample = self.add_sublayer("stage_%d_downsample"%(i),DownSample(32*(2**(i+1)),32*(2**(i+2))))
            self.downsample_list.append(downsample)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages)-1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]  #将 C0, C1, C2 作为返回值






