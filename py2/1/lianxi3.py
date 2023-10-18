import paddle
import numpy as np

def evaluation(model,datasets):
    model.eval()

    acc_set = list()
    for batch_id,data in enumerate(datasets()):
        images,labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)

        pred = model(images,labels)
        acc = paddle.metric.accuracy(input=pred,label=labels)
        acc_set.extend(acc.numpy())

    acc_val_mean = np.array(acc_set).mean()
    return acc_val_mean

