import os
import paddle
import numpy as np
import cv2

class EyeDataset(paddle.io.Dataset):
    def __init__(self,datadir,csvfile="",mode="train"):
        super().__init__()
        if mode == "train":
            self.filenames = os.listdir(datadir)
            self.images_list = []
            self.labels_list = []
            for name in self.filenames:
                filepath = os.path.join(datadir,name)
                self.images_list.append(filepath)
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                if name[0] == "H" or name[0] == "N":
                    self.labels_list.append(0)
                # P开头的是病理性近视，属于正样本，标签为1
                elif name[0] == "P":
                    self.labels_list.append(1)
        elif mode == "valid":
            filelists = open(csvfile).readlines()
            self.images_list = []
            self.labels_list = []
            for line in filelists[1:]:
                line = line.strip().split(",")
                name = line[1]
                label = int(line[2])
                self.filepath = os.path.join(datadir,name)
                self.images_list.append(self.filepath)
                self.labels_list.append(label)

    def __getitem__(self,idx):
        image = cv2.imread(self.images_list[idx])
        image = cv2.resize(image,(224,224))
        # 读入的图像数据格式是[H, W, C]
        # 使用转置操作将其变成[C, H, W]
        image = np.transpose(image,(2,0,1))
        image = image.astype("float32")
        # 将数据范围调整到[-1.0,1.0]之间
        image = image/255.
        image = image*2.0-1.0
        return paddle.to_tensor(image,dtype=paddle.float32),paddle.to_tensor(self.labels_list[idx],dtype=paddle.int64)

    def __len__(self):
        return len(self.images_list)

if __name__ == "__main__":
    dataset = EyeDataset("work/palm/PALM-Training400/PALM-Training400")
    train_loader = paddle.io.DataLoader(dataset,shuffle=True,batch_size=10)
    for batch_id,data in enumerate(train_loader()):
        images,labels = data
        print(f"batch :{batch_id} images' shape {images.shape} labels' shape {labels.shape}")
