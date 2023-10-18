import os
import paddle
import paddle.nn.functional as F

class Trainer():
    def __init__(self,model_path,model,optimizer):
        self.model_path = model_path
        self.model = model
        self.opt = optimizer

    def save(self):
        paddle.save(self.model.state_dict(),self.model_path)

    def train_step(self,data):
        images,labels = data
        predicts = self.model(images)
        loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        loss = loss_func(predicts,labels)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        self.opt.step()
        self.opt.clear_grad()
        return avg_loss

    def train_epoch(self,dataset,epoch):
        self.model.train()
        for batch_id,data in enumerate(dataset):
            loss = self.train_step(data)
            if batch_id % 200 == 0:
                print(f"epoch = {epoch} batch {batch_id} loss is {loss.numpy()[0]}")

    def train(self,train_dataset,start_epoch,end_epoch,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(start_epoch,end_epoch):
            self.train_epoch(train_dataset,i)
            paddle.save(self.opt.state_dict(),f"./{save_path}/mnist_epoch{i}"+".pdopt")
            paddle.save(self.model.state_dict(),f"./{save_path}/mnist_epoch{i}"+".pdparams")
        self.save()