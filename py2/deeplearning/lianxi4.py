import paddle.nn.functional as F
import paddle
from lianxi import train_loader
from lianxi2 import MNIST
from lianxi3 import evaluation

def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())
    epoch_num=10
    for epoch_id in range(epoch_num):
        for batch_id,data in enumerate(train_loader):
            images,labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            predicts,acc = model(images,labels)

            loss = F.cross_entropy(predicts,labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 200 ==0:
                loss = avg_loss.numpy()[0]
                print(f"epoch {epoch_id} batch {batch_id} loss is {loss} ,acc is {acc.numpy()}")

            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        # acc_train_mean = evaluation(model,train_loader)
        # acc_val_mean = evaluation(model,val_loader)
        # print(f"train_acc {acc_train_mean} val_acc {acc_val_mean}")

model = MNIST()
train(model)