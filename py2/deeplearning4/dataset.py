import paddle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import paddle.vision.transforms as T

class CaltechDataset(paddle.io.Dataset):
    def __init__(self,path,mode="train"):
        super().__init__()
        self.class_list = os.listdir(path)
        self.use_tran = True if mode == "train" else False
        self.class_list.sort()
        self.image_list = []
        if self.use_tran:
            self.transform = T.Compose(
                [T.RandomResizedCrop((224,224)),    #这里一般的话 大家都是随机大小  给定某个范围 然后随机 这里我直接 给固定数值了
                 T.ColorJitter(0.05,0.05,0.05,0.05),
                 T.RandomHorizontalFlip(0.4),
                 T.RandomVerticalFlip(0.4),
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ]
            )
        else:
            self.transform = T.Compose(
                [T.RandomResizedCrop((224,224)),   #这里一般的话 大家都是随机大小  给定某个范围 然后随机 这里我直接 给固定数值了
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ]
            )
        for kind in range(len(self.class_list)):
            image_path_list = os.listdir(os.path.join(path, self.class_list[kind]))
            for image_path in image_path_list:
                self.image_list.append([os.path.join(path, self.class_list[kind], image_path), kind])

    def __getitem__(self, idx):
        image, label = self.image_list[idx]
        image = cv2.imread(image).astype(np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.uint8)
        label = np.array(label).astype(np.int64)
        image = self.transform(image).astype(np.float32)
        return image, label

    def __len__(self):
        return len(self.image_list)

if __name__ == "__main__":
    dataset = CaltechDataset("work//Caltech101//train",mode="train")
    train_loader = paddle.io.DataLoader(dataset,batch_size=128)
    for batch_id, data in enumerate(train_loader):
        images, labels = data
        print(f"image's shape{images.shape} label's shape:{labels.shape}")