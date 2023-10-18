import math
import paddle
import paddle.nn as nn

def _make_divisible(ch,divisor=8,min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

def drop_path(x,drop_prob=0.,training=True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    keep_prob_tensor = paddle.to_tensor(keep_prob)
    output = paddle.divide(x,keep_prob_tensor) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self,drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)

class Conv(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,stride=1,groups=1,norm_layer=None,act=None):
        super().__init__()
        self.conv = nn.Conv2D(in_c,out_c,kernel_size,stride,kernel_size // 2, groups=groups)
        self.bn = nn.BatchNorm2D(out_c) if norm_layer is None else norm_layer
        self.act = nn.Swish() if act is None else act

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SqueezeExcitation(nn.Layer):
    def __init__(self,in_c,expand_c,ratio=4):
        super().__init__()
        squeeze_c = in_c // ratio
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(expand_c,squeeze_c,1)
        self.ac1 = nn.Swish()
        self.fc2 = nn.Conv2D(squeeze_c,expand_c,1)
        self.ac2 = nn.Sigmoid()

    def forward(self,x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.fc2(out)
        out = self.ac2(out)
        return out * x

class MBConv(nn.Layer):
    def __init__(self,kernel_size,in_c,out_c,expanded_ratio,stride,use_se,drop_rate,width_coefficient):
        super().__init__()
        self.use_se = use_se
        self.in_c = self.adjust_channels(in_c,width_coefficient)
        self.expanded_c = self.in_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c,width_coefficient)
        self.conv1 = Conv(self.in_c,self.expanded_c,1,1,1)
        self.conv2 = Conv(self.expanded_c,self.expanded_c,kernel_size,stride,self.expanded_c)
        self.se_block = SqueezeExcitation(self.in_c,self.expanded_c)
        self.conv3 = Conv(self.expanded_c,self.out_c,1,1,1,act=nn.Identity())
        self.use_shortcut = (stride == 1 and self.in_c == self.out_c)
        if self.use_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = self.se_block(out)
        out = self.conv3(out)
        out = self.dropout(out)
        if self.use_shortcut:
            out = out + x
        return out

    def adjust_channels(self,channels,width_coefficient):
        return _make_divisible(channels*width_coefficient,8)

class EfficientNet(nn.Layer):
    def __init__(self,width_coefficient,depth_coefficient,num_classes,drop_rate,drop_connect_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.drop_connect_rate = drop_connect_rate
        self.depth_coefficient = depth_coefficient
        self.width_coefficient = width_coefficient
        out_c1 = self.adjust_channels(32,self.width_coefficient)
        self.conv1 = Conv(3,out_c1,3,2,1)
        self.stage2 = self.make_layers()
        self.conv2 = Conv(self.adjust_channels(320,self.width_coefficient),self.adjust_channels(1280,self.width_coefficient),1,1)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.dropout = nn.Dropout(drop_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.adjust_channels(1280,self.width_coefficient),self.num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def make_layers(self):
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        cnfs = [[3, 32, 16, 1, 1, True, self.drop_connect_rate, 1],
                [3, 16, 24, 6, 2, True, self.drop_connect_rate, 2],
                [5, 24, 40, 6, 2, True, self.drop_connect_rate, 2],
                [3, 40, 80, 6, 2, True, self.drop_connect_rate, 3],
                [5, 80, 112, 6, 1, True, self.drop_connect_rate, 3],
                [5, 112, 192, 6, 2, True, self.drop_connect_rate, 4],
                [3, 192, 320, 6, 1, True, self.drop_connect_rate, 1]]
        def round_repeats(repeats):
            return int(math.ceil(self.depth_coefficient * repeats))
        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in cnfs))
        layers = []
        for cnf in cnfs:
            for j in range(round_repeats(cnf[-1])):
                drop_rate = cnf[6] * b / num_blocks
                layers.append(MBConv(cnf[0],cnf[1] if j == 0 else cnf[2],cnf[2],cnf[3],cnf[4] if j == 0 else 1,cnf[5],drop_rate,self.width_coefficient))
                b += 1

        return nn.Sequential(*layers)

    def adjust_channels(self,channels,width_coefficient):
        return _make_divisible(channels*width_coefficient,8)

def efficientnet_b0(num_class=102):
    # 224x224
    return EfficientNet(width_coefficient=1.0,depth_coefficient=1.0,num_classes=num_class,drop_rate=0.2)

def efficientnet_b1(num_classes=102):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,depth_coefficient=1.1,num_classes=num_classes,drop_rate=0.2)


def efficientnet_b2(num_classes=102):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        num_classes=num_classes,
                        drop_rate=0.3)


def efficientnet_b3(num_classes=102):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        num_classes=num_classes,
                        drop_rate=0.3)


def efficientnet_b4(num_classes=102):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        num_classes=num_classes,
                        drop_rate=0.4)


def efficientnet_b5(num_classes=102):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        num_classes=num_classes,
                        drop_rate=0.4)


def efficientnet_b6(num_classes=102):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        num_classes=num_classes,
                        drop_rate=0.5)


def efficientnet_b7(num_classes=102):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        num_classes=num_classes,
                        drop_rate=0.5)

if __name__ == "__main__":
    model = efficientnet_b3(102)
    paddle.flops(model,[1,3,300,300],print_detail=True)