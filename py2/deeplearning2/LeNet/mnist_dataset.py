import paddle
import numpy as np
import json
import gzip

class MnistDataset(paddle.io.Dataset):
    def __init__(self,mode):
        super().__init__()
        datafile = "F:/code/py2/deeplearning/work/mnist.json.gz"
        data = json.load(gzip.open(datafile))
        train_set,val_set,eval_set = data
        self.IMG_ROWS = 28
        self.IMG_COLS = 28
        if mode == "train":
            imgs,labels = train_set[0],train_set[1]
        elif mode == "valid":
            imgs,labels = val_set[0],val_set[1]
        elif mode =="eval":
            imgs,labels = eval_set[0],eval_set[1]
        else:
            raise Exception("mode can only...")
        assert len(imgs) == len(labels),\
        "length of ..."

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self,idx):
        img = np.reshape(self.imgs[idx],[1,self.IMG_ROWS,self.IMG_COLS]).astype("float32")
        label = np.reshape(self.labels[idx],[1]).astype("int64")
        return img,label
    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    train_set = MnistDataset("train")
    train_loader = paddle.io.DataLoader(train_set,batch_size=100,shuffle=True,drop_last=True)
    for batch_id ,data in enumerate(train_loader()):
        images,labels = data
        print(f"batch_id: {batch_id}, 训练数据shape :{images.shape} ,标签数据shape: {labels.shape}")

