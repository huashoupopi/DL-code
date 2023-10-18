import paddle
import paddle.nn.functional as F
import os

class Trainer(object):
    def __init__(self, model_path, model, optimizer):
        self.model_path = model_path   # 模型存放路径
        self.model = model             # 定义的模型
        self.optimizer = optimizer     # 优化器

    def save(self):
        # 保存模型
        paddle.save(self.model.state_dict(), self.model_path)

    def train_step(self, data):
        images, labels = data
        # 前向计算的过程
        predicts = self.model(images)
        # 计算损失
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        # 后向传播，更新参数的过程
        avg_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return avg_loss

    def train_epoch(self, datasets, epoch):
        self.model.train()
        for batch_id, data in enumerate(datasets()):
            loss = self.train_step(data)
            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

    def train(self, train_datasets, start_epoch, end_epoch, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(start_epoch, end_epoch):
            self.train_epoch(train_datasets, i)
            paddle.save(self.optimizer.state_dict(), f'./{save_path}/mnist_epoch{i}'+'.pdopt')
            paddle.save(self.model.state_dict(), f'./{save_path}/mnist_epoch{i}'+'.pdparams')
        self.save()
