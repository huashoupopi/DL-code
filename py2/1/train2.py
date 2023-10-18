import paddle
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from evaluation import evaluation
from dataset import *
from network2 import MNIST
def train(model):
    model.train()
    #调用加载数据的函数，获得MNIST数据集
    # opt = paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01,parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01,momentum=0.9,parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters())
    EPOCH_NUM = 1
    # loss_list = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images,labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            #前向计算的过程,同时拿到模型输出值和分类准确率
            if batch_id == 0 and epoch_id==0:
                #打印模型参数和每层输出的尺寸
                predicts,acc = model(images,labels,check_shape=True,check_content=False)
            elif batch_id == 401:
                #打印模型参数和每层的输出的值
                predicts,acc = model(images,labels,check_shape=False,check_content=True)
            else:
                predicts,acc = model(images,labels)
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts,labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 200 == 0:
                loss = avg_loss.numpy()[0]
                # loss_list.append(loss)
                print(f"epoch :{epoch_id}, batch:{batch_id} ,loss is {loss} acc is{acc.numpy()}")

            #后向传播，更新参数
            avg_loss.backward()
            #最小化loss，更新参数
            opt.step()
            #清除梯度
            opt.clear_grad()
        acc_train_mean = evaluation(model,train_loader)
        acc_val_mean = evaluation(model,val_loader)
        print(f"train_acc {acc_train_mean} val_acc:{acc_val_mean}")


model = MNIST()
train(model)