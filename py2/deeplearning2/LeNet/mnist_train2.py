import paddle
from mnist_train1 import Trainer
from LeNet_on_mnist import LeNet
from mnist_dataset import MnistDataset

train_dataset = MnistDataset("train")
train_loader = paddle.io.DataLoader(train_dataset,shuffle=True,batch_size=100,drop_last=True)
model = LeNet(10)
opt = paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters(),weight_decay=paddle.regularizer.L2Decay(coeff=1e-5))

trainer = Trainer(
    model_path="./mnist.pdparams",
    model=model,
    optimizer=opt
)
trainer.train(train_loader,0,10,"checkpoint")