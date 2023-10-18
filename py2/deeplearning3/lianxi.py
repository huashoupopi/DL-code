import paddle
import paddle.nn as nn

class BasicConv(paddle.nn.Layer):
    def __init__(self,in_c,out_c,**kwargs):
        super().__init__()
        self.conv = nn.Conv2D(in_c,out_c,**kwargs)
        # self.bn = nn.BatchNorm2D(out_c)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x

class Inception(paddle.nn.Layer):
    def __init__(self,in_c,ch1x1,ch3x3rdc,ch3x3,ch5x5rdu,ch5x5,pool_prj):
        super().__init__()
        self.conv1 = BasicConv(in_c,ch1x1,kernel_size=1)
        self.conv2 = nn.Sequential(
            BasicConv(in_c,ch3x3rdc,kernel_size=1),
            BasicConv(ch3x3rdc,ch3x3,kernel_size=3,padding=1)
        )
        self.conv3 = nn.Sequential(
            BasicConv(in_c,ch5x5rdu,kernel_size=1),
            BasicConv(ch5x5rdu,ch5x5,kernel_size=5,padding=2)
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3,stride=1,padding=1),
            BasicConv(in_c,pool_prj,kernel_size=1)
        )
    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        outputs = [out1,out2,out3,out4]

        return paddle.concat(outputs,1)

class InceptionAux(paddle.nn.Layer):
    def __init__(self,in_c,num_class):
        super().__init__()
        self.avg_pool1 = nn.AvgPool2D(kernel_size=5,stride=3)
        self.conv1 = BasicConv(in_c,128,kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048,1024)

        self.drop_out = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024,num_class)

    def forward(self,x):
        x = self.avg_pool1(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)

        return x

class GoogLeNet(nn.Layer):
    def __init__(self,num_class,aux_work=True):
        super().__init__()
        self.aux_work = aux_work
        self.conv1 = BasicConv(in_c=3,out_c=64,kernel_size=7,stride=2,padding=3)
        self.max_pool1 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.conv2 = BasicConv(in_c=64,out_c=64,kernel_size=1)
        self.conv3 = BasicConv(in_c=64,out_c=192,kernel_size=3,padding=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.Inception3a = Inception(192,64,96,128,16,32,32)
        self.Inception3b = Inception(256,128,128,192,32,96,64)
        self.max_pool3 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.Inception4a = Inception(480,192,96,208,16,48,64)
        self.Inception4b = Inception(512,192,96,208,16,48,64)
        if self.aux_work:
            self.aux1 = InceptionAux(512,2)
            self.aux2 = InceptionAux(528,2)
        self.Inception4c = Inception(512,128,128,256,24,64,64)
        self.Inception4d = Inception(512,112,144,288,32,64,64)
        self.Inception4e = Inception(528,256,160,320,32,128,128)
        self.max_pool4 = nn.MaxPool2D(kernel_size=3,stride=2,ceil_mode=True)
        self.Inception5a = Inception(832,256,160,320,32,128,128)
        self.Inception5b = Inception(832,384,192,384,48,128,128)
        self.avg_pool = nn.AdaptiveAvgPool2D((1,1))
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024,num_class)

    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.max_pool3(x)
        x = self.Inception4a(x)
        if self.training and self.aux_work:
            aux1 = self.aux1(x)
        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        if self.training and self.aux_work:
            aux2 = self.aux2(x)
        x = self.Inception4e(x)
        x = self.max_pool4(x)
        x = self.Inception5a(x)
        x = self.Inception5b(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_work:
            return x, aux1, aux2
        return x


if __name__ == "__main__":
    model = GoogLeNet(2)
    paddle.summary(model,(None,3,224,224))

