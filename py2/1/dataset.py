import os
import random
import paddle
import numpy as np
import gzip
import json

class MnistDataset(paddle.io.Dataset):
    def __init__(self,mode):
        super().__init__()
        datafile = "F:\code\py\deepLearning\work\mnist.json.gz"
        data = json.load(gzip.open(datafile))
        #读取到的数据区分训练集，验证集，测试集
        train_set,val_set,eval_set = data

        #数据集的相关参数，图片高度为IMG_ROWS 图片宽度IMG_COLS
        self.IMG_ROWS = 28
        self.IMG_COLS = 28

        if mode == "train":
            imgs,labels = train_set[0],train_set[1]
        elif mode == "valid":
            imgs,labels = val_set[0],val_set[1]
        elif mode == "eval":
            imgs,labels = eval_set[0],eval_set[1]
        else:
            raise Exception("mode can only be one of ['train','valid','eval']")

        #校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
        "length of train_imgs shoule be the same as train_labels"

        self.imgs = imgs
        self.labels = labels
    def __getitem__(self,idx):
        img = np.reshape(self.imgs[idx],[1,self.IMG_ROWS,self.IMG_COLS]).astype("float32")
        label = np.reshape(self.labels[idx],[1]).astype("int64")
        """
        以上数据加载函数load_data返回一个数据迭代器train_loader，
        该train_loader在每次迭代时的数据shape为[batch_size, 784]，
        因此需要将该数据形式reshape为图像数据形式[batch_size, 1, 28, 28]，
        其中第二维代表图像的通道数（在MNIST数据集中每张图片的通道数为1，传统RGB图片通道数为3）。
        """
        return img,label
    def __len__(self):
        return len(self.imgs)

train_dataset = MnistDataset(mode="train")
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
train_loader = paddle.io.DataLoader(train_dataset,batch_size=100,shuffle=True,drop_last=True)
val_dataset = MnistDataset(mode='valid')
val_loader = paddle.io.DataLoader(val_dataset, batch_size=128,drop_last=True)

if __name__ == "__main__":
    for batch_id,data in enumerate(train_loader):
        images,labels = data
        print(f"batch_id: {batch_id}, 训练数据shape :{images.shape} ,标签数据shape: {labels.shape}")