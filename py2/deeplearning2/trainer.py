import os
import random
import paddle
import numpy as np
import paddle.nn.functional as F
from dataset import EyeDataset

datadir1 = "./work/palm/PALM-Training400/PALM-Training400"
datadir2 = "./work/palm/PALM-Validation400"
csvfile = "./labels.csv"

epoch_num = 5

def train_pm(model,optimizer):
    model.train()
    #定义数据读取器，训练数据读取器和验证数据读取器
    # train_loader = data_loader(datadir1,batch_size=10,mode="train")
    # valid_loader = valid_data_loader(datadir2,csvfile)
    train_dataset = EyeDataset(datadir1,"")
    val_dataset = EyeDataset(datadir2,csvfile,mode="valid")
    train_loader = paddle.io.DataLoader(train_dataset,shuffle=True,batch_size=10)
    valid_loader = paddle.io.DataLoader(val_dataset,batch_size=10)
    for epoch in range(epoch_num):
        for batch_id,data in enumerate(train_loader()):
            x_data,y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            #运行模型前向计算 获得预测值
            predicts = model(img)
            # loss = F.binary_cross_entropy_with_logits(predicts,label)
            loss = F.cross_entropy(predicts,label)
            avg_loss = paddle.mean(loss)

            if batch_id % 20 == 0:
                print(f"epoch {epoch} batch_id {batch_id} loss is {avg_loss.numpy()}")

            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id ,data in enumerate(valid_loader()):
            x_data,y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            #获得预测值
            pred = model(img)
            # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
            # 计算sigmoid后的预测概率，进行loss计算
            # pred = F.sigmoid(pred)
            # loss = F.binary_cross_entropy_with_logits(pred,label)
            loss = F.cross_entropy(pred,label)
            #计算预测概率小于0.5的类别
            # pred2 = pred*(-1.0)+1.0
            # #得到两个类别的预测概率，并沿第一个维度级联
            # pred = paddle.concat([pred2,pred],axis=1)
            pred = F.softmax(pred)
            acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print(f"[validation] accuracy/loss: {np.mean(accuracies)}/{np.mean(losses)}")
        model.train()

        paddle.save(model.state_dict(), 'palm.pdparams')
        paddle.save(optimizer.state_dict(), 'palm.pdopt')
