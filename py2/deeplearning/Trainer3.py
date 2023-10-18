import warnings
import paddle
from Trainer import Trainer
from dataset import MnistDataset
from network1 import MNIST

warnings.filterwarnings('ignore')
# MLP继续训练
paddle.seed(1024)

epochs = 3
BATCH_SIZE = 32
model_path = './mnist_retrain.pdparams'

train_dataset = MnistDataset(mode='train')
# 这里为了使每次的训练精度都保持一致，因此先选择了shuffle=False，真正训练时应改为shuffle=True
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = MNIST()
# lr = 0.01
total_steps = (int(50000//BATCH_SIZE) + 1) * epochs
lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
opt = paddle.optimizer.Momentum(learning_rate=lr, parameters=model.parameters())

params_dict = paddle.load('./checkpoint/mnist_epoch0.pdparams')
opt_dict = paddle.load('./checkpoint/mnist_epoch0.pdopt')

# 加载参数到模型
model.set_state_dict(params_dict)
opt.set_state_dict(opt_dict)

trainer = Trainer(
    model_path=model_path,
    model=model,
    optimizer=opt
)
# 前面训练模型都保存了，这里save_path设置为新路径，实际训练中保存在同一目录就可以
trainer.train(train_datasets=train_loader,start_epoch = 1, end_epoch = epochs, save_path='checkpoint_con')