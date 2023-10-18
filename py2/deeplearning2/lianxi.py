import paddle
import numpy as np

class VGG(paddle.nn.Layer):
    def __init__(self,image_size=[224,224],struct='D',num_class=2):
        super().__init__()
        self.num_class = num_class
        self.config = self.make_config(struct)

        self.stage1 = self._make_stage(self.config[0],3,64,struct)
        self.stage2 = self._make_stage(self.config[1],64,128,struct)
        self.stage3 = self._make_stage(self.config[2],128,256,struct)
        self.stage4 = self._make_stage(self.config[3],256,512,struct)
        self.stage5 = self._make_stage(self.config[4],512,512,struct)
        assert image_size[0] % 32==0,"图像尺寸不合规"
        assert image_size[1] % 32==0,"图像尺寸不合规"

        self.flatten = paddle.nn.Flatten()
        self.dropout1 = paddle.nn.Dropout(0.5,mode="upscale_in_train")
        self.dropout2 = paddle.nn.Dropout(0.5,mode="upscale_in_train")
        self.fc1 = self._make_fc(int(image_size[0]*image_size[1]/(32**2)*512),4096)
        self.fc2 = self._make_fc(4096,4096)
        self.fc3 = self._make_fc(4096,num_class,True)

    def _make_stage(self,n,in_channels,out_channels,struct):
        layers = []
        for i in range(n):
            if struct == 'C' and i == 2:
                layers.append(paddle.nn.Conv2D(in_channels,out_channels,stride=1,kernel_size=1))
            else:
                layers.append(paddle.nn.Conv2D(in_channels,out_channels,stride=1,kernel_size=3,padding=1))
            in_channels = out_channels
            layers.append(paddle.nn.ReLU())
        layers.append(paddle.nn.MaxPool2D(kernel_size=2,stride=2))

        return paddle.nn.Sequential(*layers)

    def _make_fc(self,in_nodes,out_nodes,tail=False):
        layers = [paddle.nn.Linear(in_nodes,out_nodes)]
        if not tail:
            layers.append(paddle.nn.ReLU())
        return paddle.nn.Sequential(*layers)

    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def make_config(self,struct):
        config = {
            'A': [1,1,2,2,2],
            'B': [2,2,2,2,2],
            'C': [2,2,3,3,3],
            'D': [2,2,3,3,3],
            'E': [2,2,4,4,4]
        }
        return config[struct]

if __name__ == "__main__":
    model = VGG()
    print(model)
    paddle.summary(model,(None,3,224,224))