import paddle
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from dataset import *
from network1 import MNIST
def train(model):
    model.train()
    #调用加载数据的函数，获得MNIST数据集
    #各种优化算法均可以加入正则化项，避免过拟合，参数regularization_coff调节正则化项的权重
    # opt = paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01,parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01,momentum=0.9,parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters(),weight_decay=paddle.regularizer.L2Decay(coeff=1e-5))
    EPOCH_NUM = 10
    # loss_list = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images,labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            #前向计算的过程
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
        # acc_train_mean = evaluation(model,train_loader)
        # acc_val_mean = evaluation(model,val_loader)
        # print(f"train_acc {acc_train_mean} val_acc:{acc_val_mean}")
    # paddle.save(model.state_dict(),"mnist.pdparams")
    # return loss_list

model = MNIST()
# loss_list = train(model)
train(model)

# def plot(loss_list):
#     plt.figure(figsize=(10, 5))
#
#     freqs = [i for i in range(len(loss_list))]
#     # 绘制训练损失变化曲线
#     plt.plot(freqs, loss_list, color='#e4007f', label="Train loss")
#
#     # 绘制坐标轴和图例
#     plt.ylabel("loss", fontsize='large')
#     plt.xlabel("freq", fontsize='large')
#     plt.legend(loc='upper right', fontsize='x-large')
#
#     plt.show()
#
#
# plot(loss_list)