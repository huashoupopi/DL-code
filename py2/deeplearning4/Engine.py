import paddle
import os
from dataset import CaltechDataset
import visualdl
from network.GhostNet import GhostNet
import paddle.nn.functional as F
import numpy as np
from config import cfg
import tqdm
from tqdm import tqdm

class Engine:
    def __init__(self,cfg):
        self.model_cfg = cfg["model"]
        self.train_cfg = cfg["train"]
        self.val_cfg = cfg["val"]
        self.total_step = cfg["steps"]
        self.num_class = cfg["num_class"]
        self.logger = visualdl.LogWriter(logdir="./log")
        # config = {
        #     "structure": 50,
        #     "num_classes": self.num_class
        # }
        self.model = GhostNet(1,102)
        self.lr, self.opt = self.set_opt(cfg["lr"],self.total_step)
        self.now_step = self._load(cfg)
        self.loss = paddle.nn.CrossEntropyLoss()
        self.metric = paddle.metric.Accuracy((1,5))

    def train(self):
        best_answer = 0
        self.model.train()
        dataset = CaltechDataset(self.train_cfg["root"],mode="train")
        train_loader = paddle.io.DataLoader(dataset,batch_size=self.train_cfg["batch_size"],shuffle=True,drop_last=True)
        loss_set = []
        while True:
            for batch_id, data in enumerate(train_loader()):
                self.now_step += 1
                images, labels = data
                predict = self.model(images)
                loss = self.loss(predict, labels)
                loss_set.append(loss)
                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
                if self.now_step % 100 == 0:
                    loss_value = np.array(loss_set).mean()
                    loss_set.clear()
                    print(f"epoch:{self.total_step} step:{self.now_step} epoch_id:{self.now_step // len(train_loader)} lr:{self.lr.get_lr()} loss:{loss_value}")
                    self.logger.add_scalar(tag="train/loss",step=self.now_step,value=loss_value)
                if self.now_step % 1000 == 0 and self.now_step != 0:
                    answer = self.val()
                    self.logger.add_scalar(tag="val/top1", step=self.now_step, value=answer[0][0])
                    self.logger.add_scalar(tag="val/top5", step=self.now_step, value=answer[0][1])
                    self.logger.add_scalar(tag="val/loss", step=self.now_step, value=answer[1])
                    if answer[0][0] > best_answer:
                        best_answer = answer[0][0]
                        self._save(self.now_step,True)
                    else:
                        self._save(self.now_step,False)
                    print(f"epoch:{self.now_step} epoch_id:{self.now_step // len(train_loader)} top1:{answer[0][0]*100} top5:{answer[0][1]*100} best_top1 {best_answer*100} loss:{answer[1]}")
                self.lr.step()
                if self.now_step >= self.total_step:
                    return

    @paddle.no_grad()
    def val(self):
        self.model.eval()
        loss_set = []
        self.metric.reset()
        dataset = CaltechDataset(self.val_cfg["root"],mode="valid")
        val_loader = paddle.io.DataLoader(dataset,batch_size=self.val_cfg["batch_size"])
        # loop = tqdm(enumerate(val_loader), total=len(val_loader))
        # for batch_id, data in loop:
        #     images, labels = data
        #     predict = self.model(images)
        #     loss = self.loss(predict, labels)
        #     predicts = F.softmax(predict)
        #     correct = self.metric.compute(predicts, labels)
        #     loss_set.append(loss)
        #     self.metric.update(correct)
        #     loop.set_description(f"Epoch:{batch_id}/{len(val_loader)}")
        #     loop.set_postfix(acc=self.metric.accumulate()[0], loss=np.array(loss_set).mean())
        # self.model.train()
        # return [self.metric.accumulate(), np.array(loss_set).mean()]
        for batch_id, data in enumerate(tqdm(val_loader)):
            images, labels = data
            predict = self.model(images)
            loss = self.loss(predict,labels)
            predicts = F.softmax(predict)
            correct = self.metric.compute(predicts,labels)
            loss_set.append(loss)
            self.metric.update(correct)
        self.model.train()
        return [self.metric.accumulate(), np.array(loss_set).mean()]

    def _save(self,step,best):
        if self.model_cfg["save_path"] is not None:
            sub_dir = "best" if best else str(step)
            paddle.save(self.model.state_dict(), os.path.join(self.model_cfg["save_path"], sub_dir, 'model.pdparam'))
            paddle.save(self.opt.state_dict(), os.path.join(self.model_cfg["save_path"], sub_dir, 'model.pdopt'))

    def set_opt(self,lr,max_step):
        lr = paddle.optimizer.lr.CosineAnnealingDecay(lr,max_step)
        opt = paddle.optimizer.Adam(lr,parameters=self.model.parameters(),weight_decay=paddle.regularizer.L2Decay(0.0005))
        return lr, opt

    def _load(self,cfg):
        step = 0
        if cfg["continue"] is not None:
            self.model.set_state_dict(paddle.load(os.path.join(cfg["continue"],"model.pdparam")))
            self.opt.set_state_dict(paddle.load(os.path.join(cfg["continue"],"model.pdopt")))
            step = int(os.path.basename(cfg["continue"]))
            self.lr.step(step)
        return step