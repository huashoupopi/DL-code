import numpy as np
import json
import paddle
from reader import test_data_loader
from multinms import multiclass_nms
from YOLOV3 import YOLOv3

test_dir = "work//insects//test//images"
weight_file = ""

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

VALID_THRESH = 0.01

NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

NUM_CLASSES = 7
if __name__ == "__main__":
    model = YOLOv3(NUM_CLASSES)
    params_file_path = weight_file
    model.set_state_dict(paddle.load(params_file_path))
    model.eval()

    total_results = []
    test_loader = test_data_loader(test_dir,batch_size=1)
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale_data)

        outputs = model(img)
        bboxes, scores = model.get_pred(outputs,img_scale,ANCHORS,ANCHOR_MASKS,VALID_THRESH)
        bboxes_data = np.array(bboxes)
        scores_data = np.array(scores)
        result = multiclass_nms(bboxes_data,scores_data,VALID_THRESH,NMS_THRESH,NMS_TOPK,NMS_POSK)
        for j in range(len(result)):
            result_j = result[j]
            img_name_j = img_name[j]
            total_results.append([img_name_j,result_j.tolist()])
        print(f"processed {total_results}pictures")
    print("")
    json.dump(total_results,open("pred_result.json","w"))

