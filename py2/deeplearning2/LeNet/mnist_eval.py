import paddle
import numpy as np
from mnist_dataset import MnistDataset
import paddle.nn.functional as F
from LeNet_on_mnist import LeNet

def evaluation(model):
    #定义预测过程
    param_file_path = "mnist.pdparams"
    param_dict = paddle.load(param_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_dataset = MnistDataset("eval")
    eval_loader = paddle.io.DataLoader(eval_dataset,batch_size=100,shuffle=True)
    acc_set = []
    avg_loss_set = []
    for batch_id,data in enumerate(eval_loader()):
        images,labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts = model(images)
        acc = paddle.metric.accuracy(F.softmax(predicts),labels)
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
        # 计算多个batch的平均损失率和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print(f"loss {avg_loss_val_mean} acc {acc_val_mean}")


model = LeNet(10)
evaluation(model)