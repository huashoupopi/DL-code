import paddle
import paddle.nn as nn
from functools import partial

# def _make_divisible(ch, divisor=8, min_ch=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_ch is None:
#         min_ch = divisor
#     new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_ch < 0.9 * ch:
#         new_ch += divisor
#     return new_ch
#
# """
# 在有 alpha 因子时会用到上面的函数  不然不需要用到   alpha因子×通道数会使通道数变成非8的倍数  上面的函数用来调整
# """

class Conv(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,groups=1,bn=False,activation=None):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding=padding,groups=groups)
        self.bn = nn.BatchNorm2D(out_channels) if bn else None
        self.activation = activation if activation else nn.ReLU6()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.activation(x)
        return x

class SqueezeExcitation(nn.Layer):
    def __init__(self,in_channels,ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(in_channels,in_channels//ratio,1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(in_channels//ratio,in_channels,1)
        self.relu2 = nn.Hardsigmoid()

    def forward(self,x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        return out * x

class BNeck_config:
    def __init__(self,in_channels,kernel_size,exp_size,out_channels,use_se,activation,stride):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.exp_size = exp_size
        self.out_channels = out_channels
        self.use_se = use_se
        self.use_hs = True if activation == "HS" else False
        self.stride = stride

class BNeck(nn.Layer):
    def __init__(self,cfg):
        super().__init__()
        self.use_shortcut = True if cfg.in_channels == cfg.out_channels and cfg.stride == 1 else False
        self.use_se = cfg.use_se
        activation_func = nn.Hardswish() if cfg.use_hs else nn.ReLU()
        self.conv1 = Conv(cfg.in_channels,cfg.exp_size,1,1,bn=True,activation=activation_func)
        self.conv2 = Conv(cfg.exp_size,cfg.exp_size,cfg.kernel_size,cfg.stride,(cfg.kernel_size-1)//2,cfg.exp_size,bn=True,activation=activation_func)
        if self.use_se:
            self.se_block = SqueezeExcitation(cfg.exp_size)
        self.conv3 = Conv(cfg.exp_size,cfg.out_channels,1,1,bn=True,activation=nn.Identity())

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = self.se_block(out)
        out = self.conv3(out)
        if self.use_shortcut:
            out = out + x
        return out

class MobileNetv3(nn.Layer):
    def __init__(self,cfgs,num_class,last_out_channels):
        super().__init__()
        self.num_class = num_class
        self.conv1 = Conv(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1,bn=True,activation=nn.Hardswish())
        self.stage2 = self._make_layers(cfgs)
        last_layer_in_channels = cfgs[-1].out_channels
        last_layer_out_channels = 6 * last_layer_in_channels
        self.conv2 = Conv(last_layer_in_channels,last_layer_out_channels,1,1,bn=True,activation=nn.Hardswish())
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv3 = Conv(last_layer_out_channels,last_out_channels,1,1,activation=nn.Hardswish())
        self.conv4 = nn.Conv2D(last_out_channels,self.num_class,1)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        return x

    def _make_layers(self,cfgs):
        layers = []
        for cfg in cfgs:
            layers.append(BNeck(cfg))
        return nn.Sequential(*layers)

def MobileNetV3_Large(num_class):
    bneck_cfg = partial(BNeck_config)
    cfg = [
        bneck_cfg(16,3,16,16,False,"RE",1),
        bneck_cfg(16,3,64,24,False,"RE",2),
        bneck_cfg(24,3,72,24,False,"RE",1),
        bneck_cfg(24,5,72,40,True,"RE",2),
        bneck_cfg(40,5,120,40,True,"RE",1),
        bneck_cfg(40,5,120,40,True,"RE",1),
        bneck_cfg(40,3,240,80,False,"HS",2),
        bneck_cfg(80,3,200,80,False,"HS",1),
        bneck_cfg(80,3,184,80,False,"HS",1),
        bneck_cfg(80,3,184,80,False,"HS",1),
        bneck_cfg(80,3,480,112,True,"HS",1),
        bneck_cfg(112,3,672,112,True,"HS",1),
        bneck_cfg(112,5,672,160,True,"HS",2),
        bneck_cfg(160,5,960,160,True,"HS",1),
        bneck_cfg(160,5,960,160,True,"HS",1),
    ]
    last_channels = 1280
    return MobileNetv3(cfg,num_class,last_channels)

def MobileNetv3_small(num_class):
    bneck_cfg = partial(BNeck_config)
    cfg = [
        bneck_cfg(16,3,16,16,True,"RE",2),
        bneck_cfg(16,3,72,24,False,"RE",2),
        bneck_cfg(24,3,88,24,False,"RE",1),
        bneck_cfg(24,5,96,40,True,"HS",2),
        bneck_cfg(40,5,240,40,True,"HS",1),
        bneck_cfg(40,5,240,40,True,"HS",1),
        bneck_cfg(40,5,120,48,True,"HS",1),
        bneck_cfg(48,5,144,48,True,"HS",1),
        bneck_cfg(48,5,288,96,True,"HS",2),
        bneck_cfg(96,5,576,96,True,"HS",1),
        bneck_cfg(96,5,576,96,True,"HS",1),
    ]
    last_channels = 1024
    return MobileNetv3(cfg,num_class,last_channels)

if __name__ == "__main__":
    model = MobileNetV3_Large(102)
    paddle.flops(model,[1,3,224,224],print_detail=True)