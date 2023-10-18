import os.path
import matplotlib.pyplot as plt
import paddle.nn.functional as F
import cv2
from PIL import Image
import paddle
import numpy as np
from network2 import AlexNet

def transform_img(img):
    img = cv2.resize(img,(224,224))
    img = np.transpose(img,(2,0,1))
    img = img.astype("float32")
    img = img/255.
    img = img*2.0-1.0
    return img

model = AlexNet(2)
params_file_path = "palm.pdparams"
img_path = "./work/palm/PALM-Validation400"
filelists = open("labels.csv").readlines()
line = filelists[6].strip().split(",")
name ,label = line[1],int(line[2])
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)

img = cv2.imread(os.path.join(img_path,name))
trans_img = transform_img(img)
unsqueeze_img = paddle.unsqueeze(paddle.to_tensor(trans_img),axis=0)
model.eval()
logits = model(unsqueeze_img)
result = F.softmax(logits)
pred_class = paddle.argmax(result).numpy()
print(f"the ture category is {label} and the predicted category is {pred_class}")

show_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.imshow(show_img)
plt.show()