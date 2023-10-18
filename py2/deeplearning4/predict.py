import paddle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from network.ResNet import ResNet


def transform_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.transpose(img,(2,0,1))
    img = img.astype("float32")
    img = img / 255
    img = img*2.0 - 1.0
    return img
def infer_img(path,model_filepath,img_path):
    configs = {
        "structure": 50,
        "num_classes": 102}
    model = ResNet(configs)
    model.load_dict(paddle.load(model_filepath))
    img = transform_img(img_path)
    unsqueeze_img = paddle.unsqueeze(paddle.to_tensor(img),axis=0)
    model.eval()
    labels_list = os.listdir(path)
    labels_list.sort()
    result = model(unsqueeze_img)
    result = paddle.nn.functional.softmax(result)
    pred_class = paddle.argmax(result)
    print(f"样本:{os.path.basename(os.path.dirname(img_path))} 被预测为:{labels_list[int(pred_class)]}")
infer_img("work//Caltech101//val","work//model.pdparam","work/Caltech101/val/panda/image_0008.jpg")