import paddle
import paddle.nn.functional as F
import numpy as np
from dataset import EyeDataset

def evaluation(model,params_file_path):
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)

    model.eval()
    datadir = "./work/palm/PALM-Validation400"
    csvfile = "./labels.csv"
    eval_dataset = EyeDataset(datadir,csvfile,mode="valid")
    eval_loader = paddle.io.DataLoader(eval_dataset,batch_size=10,shuffle=False)

    acc_set = []
    avg_loss_set = []
    for batch_id ,data in enumerate(eval_loader()):
        x_data,y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # pred = model(img)
        # pred2 = F.sigmoid(pred)
        # pred2= paddle.concat([1.0 - pred2, pred2], axis=1)
        # acc = paddle.metric.accuracy(pred2, paddle.cast(label_64, dtype='int64'))
        # loss = F.binary_cross_entropy_with_logits(pred,label)
        pred = model(img)
        loss = F.cross_entropy(pred,label)
        pred = F.softmax(pred)
        acc = paddle.metric.accuracy(pred, label)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    #求平均精度
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print(f"loss {avg_loss_val_mean} acc={acc_val_mean}")