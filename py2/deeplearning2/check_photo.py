import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

datafile = "./work/palm/PALM-Training400/PALM-Training400"
#文件名以N开头的是正常眼底图片，以P开头的是病变眼底图片
file1 = "N0012.jpg"
file2 = "P0095.jpg"

#读取图片
img1 = Image.open(os.path.join(datafile,file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(datafile,file2))
img2 = np.array(img2)

#画出读取的图片
plt.figure(figsize=(16,8))
f = plt.subplot(121)
f.set_title("Normal",fontsize=20)
plt.imshow(img1)
f = plt.subplot(122)
f.set_title("PM",fontsize=20)
plt.imshow(img2)
plt.show()

print(img1.shape,img2.shape)
"""
(2056, 2124, 3) (2056, 2124, 3)
"""
