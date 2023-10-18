import paddle
import warnings
from dataset import MnistDataset
from network1 import MNIST
from Trainer import Trainer

warnings.filterwarnings("ignore")
epochs = 10
batch_size = 100
model_path = "./mnist.pdparams"
train_dataset = MnistDataset("train")
train_loader = paddle.io.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

model = MNIST()
total_step = (int(50000//batch_size)+1) * epochs
lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01,decay_steps=total_step,end_lr=0.001)
opt = paddle.optimizer.Momentum(learning_rate=lr,parameters=model.parameters())

trainer = Trainer(
    model_path=model_path,
    model=model,
    optimizer=opt
)
trainer.train(train_datasets=train_loader,start_epoch=0,end_epoch=epochs,save_path="checkpoint")