import time
import os
import numpy as np
import paddle
from reader import TrainDataset
from YOLOV3 import YOLOv3

# 提升点： 可以改变anchor的大小，注意训练和测试时要使用同样的anchor
ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

IGNORE_THRESH = .7
NUM_CLASSES = 7

def get_lr(base_lr=1e-4,lr_decay=0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr*lr_decay, base_lr*lr_decay*lr_decay]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd,values=lr)
    return learning_rate

if __name__ == "__main__":
    traindir = "work//insects//train"
    validdir = "work//insects//val"
    testdir = "work//insects//test"
    train_dataset = TrainDataset(traindir,mode="train")
    val_dataset = TrainDataset(validdir,mode="valid")
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
    valid_loader = paddle.io.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)
    model = YOLOv3(NUM_CLASSES)
    lr = get_lr()
    opt = paddle.optimizer.Momentum(learning_rate=lr,momentum=0.9,parameters=model.parameters(),
                                    weight_decay=paddle.regularizer.L2Decay(coeff=5e-4))
    #opt = paddle.optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005),
    #                            parameters=model.parameters())

    epochs = 200
    for epoch in range(epochs):
        for i, data in enumerate(train_loader()):
            model.train()
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype("float32")
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)
            loss = model.get_loss(outputs,gt_boxes,gt_labels,gt_scores,ANCHORS,ANCHOR_MASKS,IGNORE_THRESH)
            loss.backward()
            opt.step()
            opt.clear_grad()
            if i % 10 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print(f"{timestring}[TRAIN]epoch:{epoch}, iter:{i}, loss{loss.numpy()}")

        if (epoch % 5 == 0) or (epoch == epochs - 1):
            paddle.save(model.state_dict(), os.path.join("output",f"yolo_epoch{epoch}"))
            paddle.save(opt.state_dict(), os.path.join("output",f"yolo_epoch{epoch}"))

        model.eval()
        for i, data in enumerate(valid_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors=ANCHORS,
                                  anchor_masks=ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)

            if i % 1 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print(f"{timestring}[VALID]epoch:{epoch}, iter:{i}, loss:{loss.numpy()}")




