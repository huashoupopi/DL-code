# import cv2
# import random
# import numpy as np
# import os
#
# #对读入的图像数据进行预处理
# def transform_img(img):
#     #将图片尺寸缩放到 224x224
#     img = cv2.resize(img,(224,224))
#     # 读入的图像数据格式是[H, W, C]
#     # 使用转置操作将其变成[C, H, W]
#     img = np.transpose(img,(2,0,1))
#     img = img.astype("float32")
#     #将数据范围调整到[-1.0,1.0]之间
#     img = img / 255.
#     img = img*2.0-1.0
#     return img
#
# #定义训练集数据读取器
# def data_loader(datadir,batch_size=10,mode='train'):
#     #将datadir目录下的文件列出来，每条文件都要读入
#     filenames = os.listdir(datadir)
#     def reader():
#         if mode == "train":
#             #训练时随机打乱数据顺序
#             random.shuffle(filenames)
#         batch_imgs = []
#         batch_labels = []
#         for name in filenames:
#             filepath = os.path.join(datadir,name)
#             img = cv2.imread(filepath)
#             img = transform_img(img)
#             if name[0] == "H" or name[0] == "N":
#                 # H开头的文件名表示高度近似，N开头的文件名表示正常视力
#                 # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
#                 label = 0
#             elif name[0] == "P":
#                 #P开头的是病理性近视，属于正样本，标签为1
#                 label = 1
#             else:
#                 raise  ("Not excepted file name")
#             # 每读取一个样本的数据，就将其放入数据列表中
#             batch_imgs.append(img)
#             batch_labels.append(label)
#             if len(batch_imgs) == batch_size:
#                 # 当数据列表的长度等于batch_size的时候，
#                 # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
#                 imgs_array = np.array(batch_imgs).astype('float32')
#                 # labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
#                 # labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
#                 labels_array = np.array(batch_labels).reshape(-1, 1)
#                 yield imgs_array, labels_array
#                 batch_imgs = []
#                 batch_labels = []
#         if len(batch_imgs) > 0:
#             # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
#             imgs_array = np.array(batch_imgs).astype('float32')
#             # labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
#             # labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
#             labels_array = np.array(batch_labels).reshape(-1, 1)
#             yield imgs_array, labels_array
#     return reader
# #定义验证集读取器
# def valid_data_loader(datadir,csvfile,batch_size=10,mode="valid"):
#     # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
#     # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
#     # csvfile文件所包含的内容格式如下，每一行代表一个样本，
#     # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
#     # 第四列和第五列是Fovea的坐标，与分类任务无关
#     # ID,imgName,Label,Fovea_X,Fovea_Y
#     # 1,V0001.jpg,0,1157.74,1019.87
#     # 2,V0002.jpg,1,1285.82,1080.47
#     # 打开包含验证集标签的csvfile，并读入其中的内容
#     filelists = open(csvfile).readlines()
#     def reader():
#         batch_imgs = []
#         batch_labels = []
#         for line in filelists[1:]:
#             line = line.strip().split(",")
#             name = line[1]
#             label = int(line[2])
#             #根据图片文件名加载图片，并对图像数据作预处理
#             filepath = os.path.join(datadir,name)
#             img = cv2.imread(filepath)
#             img = transform_img(img)
#             # 每读取一个样本的数据，就将其放入数据列表中
#             batch_imgs.append(img)
#             batch_labels.append(label)
#             if len(batch_imgs) == batch_size:
#                 # 当数据列表的长度等于batch_size的时候，
#                 # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
#                 imgs_array = np.array(batch_imgs).astype("float32")
#                 # labels_array = np.array(batch_labels).astype("int64").reshape(-1,1)
#                 # labels_array = np.array(batch_labels).astype("float32").reshape(-1,1)
#                 labels_array = np.array(batch_labels).reshape(-1,1)
#                 yield imgs_array,labels_array
#                 batch_imgs = []
#                 batch_labels = []
#
#         if len(batch_imgs) > 0:
#             # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
#             imgs_array = np.array(batch_imgs).astype('float32')
#             # labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
#             # labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
#             labels_array = np.array(batch_labels).reshape(-1, 1)
#             yield imgs_array, labels_array
#
#     return reader
#
# if __name__ == "__main__":
#     datadir = "work/palm/PALM-Training400/PALM-Training400"
#     train_loader = data_loader(datadir,10,mode="train")
#     data_reader = train_loader()
#     data = next(data_reader)
#     print(data[0].shape,data[1].shape)
#     # eval_loader = data_loader(datadir, batch_size=10, mode='eval')
#     # data_reader = eval_loader()
#     # data = next(data_reader)
#     # print(data[0].shape, data[1].shape)
import cv2
import paddle
import os
import numpy as np

class EyeDataset(paddle.io.Dataset):
    def __init__(self,datadir,csvfile,mode="train"):
        super().__init__()
        if mode == "train":
            self.filenames = os.listdir(datadir)
            self.images_list = []
            self.labels_list = []
            for name in self.filenames:
                filepath = os.path.join(datadir,name)
                self.images_list.append(filepath)
                if name[0] == "H" or name[0] == "N":
                    self.labels_list.append(0)
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
        image = np.transpose(image,(2,0,1))
        image = image.astype("float32")
        # 将数据范围调整到[-1.0,1.0]之间
        image = image / 255.
        image = image * 2.0 - 1.0
        return paddle.to_tensor(image,dtype=paddle.float32),paddle.to_tensor(self.labels_list[idx],dtype=paddle.int64)

    def __len__(self):
        return len(self.images_list)

if __name__ == "__main__":
    dataset = EyeDataset("work/palm/PALM-Training400/PALM-Training400")
    train_loader = paddle.io.DataLoader(dataset,shuffle=True,batch_size=10)
    for batch_id,data in enumerate(train_loader()):
        images,labels = data
        print(f"batch_id: {batch_id}, 训练数据shape: {images.shape}, 标签数据shape: {labels.shape}")