import paddle
import numpy as np
import visualdl
from dataset import EyeDataset
from VGG import VGG
from lianxi3 import ResNet
import paddle.nn.functional as F

class Engine:
    def __init__(self,cfg):
        self.model_cfg = cfg["model"]
        self.train_cfg = cfg["train"]
        self.val_cfg = cfg["val"]
        self.step = cfg["step"]
        # self.model = VGG()
        self.model = ResNet(50)
        self._load()
        self.lr, self.opt = self.set_opt(cfg["lr"],cfg["step"],self.model.parameters())
        # self.logger = visualdl.LogWriter(logdir="./log")
        self.loss = paddle.nn.CrossEntropyLoss()
        self.metric = paddle.metric.Accuracy((1,5))

    def train(self):
        best_answer = 0
        run = True
        self.model.train()
        # ops = transforms.Compose([
        #     transforms.HueTransform(0.4),
        #     transforms.RandomResizedCrop(size=[224, 224], scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3)),
        #     transforms.Transpose((2, 0, 1)),
        #     transforms.Normalize(mean=[127.5], std=[127.5])
        # ])
        dataset = EyeDataset(self.train_cfg["root"],mode="train")
        train_loader = paddle.io.DataLoader(dataset,batch_size=self.train_cfg["batch_size"],shuffle=True)
        epoch, item = 0, 0
        loss_set = []
        while run :
            epoch += 1
            for batch_id,data in enumerate(train_loader()):
                item+=1
                if item>self.step:
                    run = False
                    break
                images,labels = data
                # predict1, predict2, predict3 = self.model(images)
                # predict = predict1+predict2*0.3+predict3*0.3
                predict = self.model(images)
                loss = self.loss(predict,labels)
                loss_set.append(loss)
                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
                if item % 20 == 0:
                    loss_value = np.array(loss_set).mean()
                    loss_set.clear()
                    print(f"step:{item} epoch:{self.step} lr:{self.lr.get_lr()} loss {loss_value}")
                    # self.logger.add_scalar(tag="train/loss", step=item, value=loss_value)
                if item % 80 == 0:
                    answer = self.val()
                    # self.logger.add_scalar(tag="val/top1", step=item, value=answer[0][0])
                    # self.logger.add_scalar(tag="val/top5", step=item, value=answer[0][1])
                    # self.logger.add_scalar(tag="val/loss", step=item, value=answer[1])
                    if answer[0][0] > best_answer:
                        best_answer = answer[0][0]
                        self._save(item, True)
                    else:
                        self._save(item, False)
                    print(f"step:{item} epoch:{self.step} top1:{answer[0][0]*100} top5:{answer[0][1]*100} best_top1 {best_answer*100} loss:{answer[1]}")
                self.lr.step()
    @paddle.no_grad()    #阻止记录方向传播，防止梯度爆炸
    def val(self):
        self.model.eval()
        # ops = transforms.Compose([       # 数据增强
        #     transforms.Resize(size=224, interpolation="bicubic"),    # 短边缩放到224
        #     transforms.CenterCrop(size=224),
        #     transforms.Transpose((2, 0, 1)),
        #     transforms.Normalize(mean=[127.5], std=[127.5])
        # ])
        loss_set = []               #记录损失
        self.metric.reset()         #每次验证数据前，清空评估器
        #定义数据集
        dataset = EyeDataset(self.val_cfg["root"],self.val_cfg["csvfile"],mode="valid")
        val_dataloader = paddle.io.DataLoader(dataset,batch_size=self.val_cfg["batch_size"],shuffle=True)
        #对验证集数据进行迭代
        for batch_id,data in enumerate(val_dataloader()):
            images,labels = data
            # image = transforms.to_tensor(image)
            # label = transforms.to_tensor(label)
            predict = self.model(images)
            loss = self.loss(predict,labels)
            predicts = F.softmax(predict)
            correct = self.metric.compute(predicts,labels)
            self.metric.update(correct)
            loss_set.append(loss)
        self.model.train()              #开启训练模式
        return [self.metric.accumulate(),np.array(loss_set).mean()]    #返回[[top1,top5],loss_avg]

    def _load(self):
        pass

    #     if self.model_cfg["model_path"]:
    #         self.model.set_state_dict(paddle.load(self.model_cfg["model_path"]+".pdparam"))
    #         self.opt.set_state_dict(paddle.load(self.model_cfg["model_path"] + ".pdoptim"))

    def _save(self,item,best):
        if self.model_cfg["save_path"]:
            if best:
                paddle.save(self.model.state_dict(),self.model_cfg["save_path"]+ "_best.pdparam")
                paddle.save(self.opt.state_dict(),self.model_cfg["save_path"]+ "_best.pdopt")
            if item:
                paddle.save(self.model.state_dict(),self.model_cfg["save_path"] + f"_{item}.pdparam")
                paddle.save(self.opt.state_dict(),self.model_cfg["save_path"] + f"_{item}.pdopt")

    def set_opt(self,learning_rate,max_step,model_param):
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate,max_step)
        # opt = paddle.optimizer.Momentum(learning_rate,0.9,model_param,weight_decay=paddle.regularizer.L2Decay())
        opt = paddle.optimizer.Adam(learning_rate,parameters=model_param,weight_decay=paddle.regularizer.L2Decay())
        return learning_rate, opt

