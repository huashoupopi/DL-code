import numpy as np
import time
import paddle
import paddle.nn as nn

from darknet53 import ConvBnLayer, DarkNet53_conv_body

# 定义生成 YOLO-V3 预测输出的模块
# C0 --> r0, p0
class YoloDetectionBlock(nn.Layer):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv1 = ConvBnLayer(in_c,out_c,1,1)
        self.conv2 = ConvBnLayer(out_c,out_c*2,3,1)
        self.conv3 = ConvBnLayer(out_c*2,out_c,1,1)
        self.conv4 = ConvBnLayer(out_c,out_c*2,3,1)
        self.route = ConvBnLayer(out_c*2,out_c,1,1)
        self.tip = ConvBnLayer(out_c,out_c*2,3,1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        route = self.route(x)
        tip = self.tip(route)
        return route, tip

# 定义YOLO-V3 模型
class YOLOv3(nn.Layer):
    def __init__(self,num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []

        for i in range(3):
            yolo_block = self.add_sublayer(
                "yolo_detection_block_%d"%(i),
                YoloDetectionBlock(in_c=512//(2**i)*2 if i == 0 else 512//(2**i)*2 + 512//(2**i),
                                   out_c=512//(2**i)))
            self.yolo_blocks.append(yolo_block)
            num_filters = 3 * (self.num_classes+5)
            # 添加从ti生成pi的模块，这是一个Conv2D操作，输出通道数为 num_filters
            block_out = self.add_sublayer("block_out_%d"%(i),
                                          nn.Conv2D(in_channels=512//(2**i)*2,
                                                    out_channels=num_filters,
                                                    kernel_size=1,
                                                    stride=1
                                                    ))
            self.block_outputs.append(block_out)
            if i < 2:
                route = self.add_sublayer("route2_%d"%(i),
                                          ConvBnLayer(in_c=512//(2**i),
                                                      out_c=256//(2**i),
                                                      kernel_size=1,
                                                      stride=1))
                self.route_blocks_2.append(route)
            self.upsample = nn.Upsample(scale_factor=2)

    def forward(self,x):
        outputs = []
        blocks = self.block(x)
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route,block],axis=1)
            route, tip = self.yolo_blocks[i](block)
            block_out = self.block_outputs[i](tip)
            outputs.append(block_out)

            if i < 2:
                route = self.route_blocks_2[i](route)
                route = self.upsample(route)

        return outputs

    def get_loss(self,outputs,gtbox,gtlabel,gtscore=None,
                 anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_thresh=0.7,
                 use_label_smooth=False):
        """
        使用paddle.vision.ops.yolo_loss，直接计算损失函数，过程更简洁，速度也更快
        """
        self.losses = []
        down_sample = 32
        for i, out in enumerate(outputs):    # 对三个层级分别求 损失函数
            anchor_mask_i = anchor_masks[i]
            loss = paddle.vision.ops.yolo_loss(
                x=out,  # out是P0, P1, P2中的一个
                gt_box=gtbox,  # 真实框坐标
                gt_label=gtlabel,  # 真实框类别
                gt_score=gtscore,  # 真实框得分，使用mixup训练技巧时需要，不使用该技巧时直接设置为1，形状与gtlabel相同
                anchors=anchors,  # 锚框尺寸，包含[w0, h0, w1, h1, ..., w8, h8]共9个锚框的尺寸
                anchor_mask=anchor_mask_i,  # 筛选锚框的mask，例如anchor_mask_i=[3, 4, 5]，将anchors中第3、4、5个锚框挑选出来给该层级使用
                class_num=self.num_classes,  # 分类类别数
                ignore_thresh=ignore_thresh,  # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
                downsample_ratio=down_sample,  # 特征图相对于原图缩小的倍数，例如P0是32， P1是16，P2是8
                use_label_smooth=False)  # 使用label_smooth训练技巧时会用到，这里没用此技巧，直接设置为False
            self.losses.append(paddle.mean(loss))  # mean对每张图片求和
            down_sample = down_sample // 2  # 下一级特征图的缩放倍数会减半
        return sum(self.losses)  # 对每个层级求和

    def get_pred(self,
                 outputs,
                 im_shape=None,
                 anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 valid_thresh=0.01):
        downsample = 32
        total_boxes = []
        total_scores = []
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])

            boxes, scores = paddle.vision.ops.yolo_box(
                x=out,
                img_size=im_shape,
                anchors=anchors_this_level,
                class_num=self.num_classes,
                conf_thresh=valid_thresh,
                downsample_ratio=downsample,
                name="yolo_box" + str(i))
            total_boxes.append(boxes)
            total_scores.append(
                paddle.transpose(
                    scores, perm=[0, 2, 1]))
            downsample = downsample // 2

        yolo_boxes = paddle.concat(total_boxes, axis=1)
        yolo_scores = paddle.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores

