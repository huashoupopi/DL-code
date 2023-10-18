import paddle
import numpy as np
from dataset import MnistDataset
import paddle.nn.functional as F
from network1 import MNIST

def evaluation(model):
    #定义预测过程
    params_file_path = "mnist.pdparams"
    #加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_dataset = MnistDataset("eval")
    eval_loader = paddle.io.DataLoader(eval_dataset,shuffle=True,batch_size=100)
    acc_set = []
    avg_loss_set = []
    for batch_id,data in enumerate(eval_loader()):
        images,labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts,acc = model(images,labels)
        loss = F.cross_entropy(predicts,labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    #计算多个batch的平均损失率和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print(f"loss {avg_loss_val_mean} acc {acc_val_mean}")

model = MNIST()
evaluation(model)