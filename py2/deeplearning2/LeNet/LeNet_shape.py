import paddle
from LeNet_on_mnist import LeNet
import numpy as np

#输入数据形状 [N,1,H,W]
#这里用np.random创建一个随机数组作为输入数据
x = np.random.randn(*[3,1,28,28])
x = x.astype("float32")

#创建LeNet类的实例，指定模型名称和分类的类别数目
model = LeNet(num_class=10)
# 通过调用LeNet从基类继承的sublayers()函数，
# 查看LeNet中所包含的子层
print(model.sublayers())
x = paddle.to_tensor(x)
for item in model.sublayers():
    #item是LeNet类中的一个子层
    #查看经过子层之后的输出数据形状
    try:
        x = item(x)
    except:
        x =paddle.reshape(x,[x.shape[0],-1])
        x = item(x)
    if len(item.parameters())==2:
    # 查看卷积和全连接层的数据和参数的形状，
    # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
        print(item.full_name(),x.shape,item.parameters()[0].shape,item.parameters()[1].shape)
    else:
    #池化层没有参数
        print(item.full_name(),x.shape)