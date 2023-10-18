import paddle
import paddle.nn as nn

class SKConv(nn.Layer):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.LayerList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2D(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias_attr=False),
                nn.BatchNorm2D(features),
                nn.ReLU()
            ))
        self.gap = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Sequential(nn.Conv2D(features, d, kernel_size=1, stride=1, bias_attr=False),
                                nn.BatchNorm2D(d),
                                nn.ReLU())
        self.fcs = nn.LayerList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2D(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = paddle.concat(feats, axis=1)
        feats = paddle.reshape(feats, (batch_size, self.M, self.features, feats.shape[2], feats.shape[3]))

        feats_U = paddle.sum(feats, axis=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = paddle.concat(attention_vectors, axis=1)
        attention_vectors = paddle.reshape(attention_vectors, (batch_size, self.M, self.features, 1, 1))
        attention_vectors = self.softmax(attention_vectors)

        feats_V = paddle.sum(feats * attention_vectors, axis=1)

        return feats_V

class SKUnit(nn.Layer):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2D(in_features, mid_features, 1, stride=1, bias_attr=False),
            nn.BatchNorm2D(mid_features),
            nn.ReLU()
        )

        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)

        self.conv3 = nn.Sequential(
            nn.Conv2D(mid_features, out_features, 1, stride=1, bias_attr=False),
            nn.BatchNorm2D(out_features)
        )

        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_features, out_features, 1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_features)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)

        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Layer):
    def __init__(self, class_num, nums_block_list=[3, 4, 6, 3], strides_list=[1, 2, 2, 2]):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2D(3, 64, 7, stride=2, padding=3, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2D(3, stride=2, padding=1)

        self.stage_1 = self._make_layer(64, 128, 256, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])

        self.gap = nn.AdaptiveAvgPool2D((1, 1))
        self.Flatten = nn.Flatten()
        self.classifier = nn.Linear(2048, class_num)

    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers = [SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = self.Flatten(fea)
        fea = self.classifier(fea)
        return fea

def SKNet26(nums_class=1000):
    return SKNet(nums_class, [2, 2, 2, 2])

def SKNet50(nums_class=1000):
    return SKNet(nums_class, [3, 4, 6, 3])

def SKNet101(nums_class=1000):
    return SKNet(nums_class, [3, 4, 23, 3])

if __name__ == '__main__':
    model = SKNet50(102)
    paddle.flops(model,[1,3,224,224],print_detail=True)