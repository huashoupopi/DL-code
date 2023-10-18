import numpy as np
import paddle
from reader import single_image_data_loader
from multinms import multiclass_nms
from YOLOV3 import YOLOv3
from draw_result import draw_result

image_name = "work//insects//test//images//2061.jpeg"
weight_file = "yolo_epoch75.yolo_epoch75"

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

VALID_THRESH = 0.01

NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

NUM_CLASSES = 7

if __name__ == "__main__":
    model = YOLOv3(num_classes=NUM_CLASSES)
    model.set_state_dict(paddle.load(weight_file))
    model.eval()

    total_results = []
    test_loader = single_image_data_loader(image_name, mode='test')
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale_data)

        outputs = model(img)
        bboxes, scores = model.get_pred(outputs,
                                        im_shape=img_scale,
                                        anchors=ANCHORS,
                                        anchor_masks=ANCHOR_MASKS,
                                        valid_thresh=VALID_THRESH)

        bboxes_data = np.array(bboxes)
        scores_data = np.array(scores)
        results = multiclass_nms(bboxes_data, scores_data,
                                 score_thresh=VALID_THRESH,
                                 nms_thresh=NMS_THRESH,
                                 pre_nms_topk=NMS_TOPK,
                                 pos_nms_topk=NMS_POSK)

        result = results[0]
        draw_result(result, image_name, draw_thresh=0.5)