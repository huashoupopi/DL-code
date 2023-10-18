# import paddle
# import cv2
# from PIL import Image
# import numpy as np
# from lianxi3 import ResNet
# import matplotlib.pyplot as plt
# import os
#
# def transform_img(img):
#     img = cv2.resize(img,(224,224))
#     img = np.transpose(img,(2,0,1))
#     img = img.astype("float32")
#     img = img/255.
#     img = img*2.0-1.0
#     return img
#
# model = ResNet(50)
# params_file_path = "output/ResNet/ResNet_best.pdparam"
# img_path = "./work/palm/PALM-Validation400"
# filelists = open("labels.csv").readlines()
# line = filelists[241].strip().split(',')
# name, label = line[1], int(line[2])
# param_dict = paddle.load(params_file_path)
# model.load_dict(param_dict)
#
# img = cv2.imread(os.path.join(img_path,name))
# tran_img = transform_img(img)
# unsqueeze_img = paddle.unsqueeze(paddle.to_tensor(tran_img),axis=0)
# model.eval()
# logits = model(unsqueeze_img)
# result = paddle.nn.functional.softmax(logits)
# pred_class = paddle.argmax(result).numpy()
# print(f"the true category is {label} and the predicts category is {pred_class}")
#
# show_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.imshow(show_img)
# plt.show()
import paddle
import cv2
import numpy as np
import os
from lianxi3 import ResNet

def transform_img(img):
    img = cv2.resize(img,(224,224))
    img = np.transpose(img,(2,0,1))
    img = img.astype("float32")
    img = img/255.
    img = img*2.0-1.0
    return img

params_file_path = "output/ResNet/ResNet_best.pdparam"
img_path = "./work/palm/PALM-Validation400"
filelists = open("labels.csv").readlines()
model = ResNet(50)
num = 0
for i in range(1,401):
    line = filelists[i].strip().split(',')
    name,labels = line[1], int(line[2])
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    img = cv2.imread(os.path.join(img_path,name))
    tran_img = transform_img(img)
    unsqueeze_img = paddle.unsqueeze(paddle.to_tensor(tran_img),axis=0)
    model.eval()
    logits = model(unsqueeze_img)
    result = paddle.nn.functional.softmax(logits)
    pred_class = paddle.argmax(result).numpy()
    if pred_class == labels:
        num += 1

print(num,num/400*100)