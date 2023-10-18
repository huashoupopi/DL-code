import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.nn import Conv2D,MaxPool2D,Linear,Dropout

class AlexNet(paddle.nn.Layer):
    def __init__(self,num_classes=1):
        super().__init__()
        self.conv1 = Conv2D(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=5)
        self.max_pool1 = MaxPool2D(kernel_size=2,stride=2)
        self.conv2 = Conv2D(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2,stride=2)
        self.conv3 = Conv2D(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.conv4 = Conv2D(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.conv5 = Conv2D(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.max_pool5 = MaxPool2D(kernel_size=2,stride=2)

        self.fc1 = Linear(in_features=12544,out_features=4096)
        self.drop_ratiol = 0.5
        self.drop1 = Dropout(self.drop_ratiol)
        self.fc2 = Linear(in_features=4096,out_features=4096)
        self.drop_ratio2 = 0.5
        self.drop2 = Dropout(self.drop_ratio2)
        self.fc3 = Linear(in_features=4096,out_features=num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool5(x)
        x = paddle.reshape(x,[x.shape[0],-1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x