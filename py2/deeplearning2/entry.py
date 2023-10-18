from network2 import AlexNet
import paddle
from trainer import train_pm
from eval import evaluation

model = AlexNet(2)
opt = paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters(),weight_decay=paddle.regularizer.L2Decay(coeff=1e-5))
train_pm(model,optimizer=opt)
evaluation(model, params_file_path="palm.pdparams")

