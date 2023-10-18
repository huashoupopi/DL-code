import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,stride,groups,norm=None,act=None):
        super().__init__()
        self.conv = nn.Conv2D(in_c,out_c,kernel_size,stride,kernel_size//2,groups=groups)
        self.norm = nn.BatchNorm2D(out_c) if norm is None else norm
        self.act = nn.ReLU() if act is None else act

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class SE_Block(nn.Layer):
    def __init__(self,in_c,ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(in_c,in_c//ratio,1,1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(in_c//ratio,in_c,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class GhostModle(nn.Layer):
    def __init__(self,in_c,out_c,kernel_size,dw_size=3,ratio=2,stride=1,relu=True):
        super().__init__()
        init_c = out_c // ratio
        new_c = init_c * (ratio-1)
        self.conv1 = Conv(in_c,init_c,kernel_size,stride,1,act=None if relu else nn.Identity())
        self.cheap_operator = Conv(init_c,new_c,dw_size,1,init_c,act = None if relu else nn.Identity())

    def forward(self,x):
        x = self.conv1(x)
        y = self.cheap_operator(x)
        out = paddle.concat([x,y],axis=1)
        return out

class GhostBottle(nn.Layer):
    def __init__(self,in_c,hidden_dim,out_c,kernel_size,stride,use_se):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        self.in_c = in_c
        self.out_c = out_c
        self.ghost_1 = GhostModle(in_c,hidden_dim,1,dw_size=3,stride=1)
        if stride == 2:
            self.dwconv = Conv(hidden_dim,hidden_dim,kernel_size,stride,hidden_dim,act=nn.Identity())
        if use_se:
            self.se_block = SE_Block(hidden_dim)
        self.ghost_2 = GhostModle(hidden_dim,out_c,1,dw_size=3,stride=1,relu=False)
        if stride != 1 or in_c != out_c:
            self.shortcut1 = Conv(in_c,in_c,kernel_size,stride,in_c,act=nn.Identity())
            self.shortcut2 = Conv(in_c,out_c,1,1,1,act=nn.Identity())

    def forward(self,x):
        out = self.ghost_1(x)
        if self.stride == 2:
            out = self.dwconv(out)
        if self.use_se:
            out = self.se_block(out)
        out = self.ghost_2(out)
        if self.stride == 1 and self.in_c == self.out_c:
            shortcut = x
        else:
            shortcut = self.shortcut1(x)
            shortcut = self.shortcut2(shortcut)
        out = out + shortcut
        return out

class GhostNet(nn.Layer):
    def __init__(self,scale,num_class):
        super().__init__()
        self.cfgs = [
            # k, t, c, SE, s
            [3, 16, 16, 0, 1],
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]
        self.scale = scale
        out_c = self._make_divisor(16*self.scale,8)
        self.conv1 = Conv(3,out_c,3,2,1)
        ghost_list = []
        for k, exp_size, c, use_se, stride in self.cfgs:
            in_channels = out_c
            out_c = int(self._make_divisor(self.scale*c,8))
            hidden_dim = int(self._make_divisor(self.scale*exp_size,8))
            ghost_list.append(GhostBottle(in_channels,hidden_dim,out_c,k,stride,use_se))
        self.stage = nn.Sequential(*ghost_list)
        in_channels = out_c
        out_c = int(self._make_divisor(self.scale*exp_size,8))
        self.conv2 = Conv(in_channels,out_c,1,1,1)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        in_channels = out_c
        hidden_out_c = 1280
        self.conv3 = Conv(in_channels,hidden_out_c,1,1,1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_out_c,num_class)

    def forward(self,x):
        x = self.conv1(x)
        x = self.stage(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _make_divisor(self,v,divisor=8,min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value,int(v + divisor / 2)//divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


if __name__ == "__main__":
    model = GhostNet(1,102)
    paddle.flops(model,[1,3,224,224],print_detail=True)