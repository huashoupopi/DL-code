import numpy as np

def bbox_area(bbox,is_bbox_normalized):
    """
        计算边界框的面积

        参数:
        bbox (list): 边界框坐标 [xmin, ymin, xmax, ymax]
        is_bbox_normalized (bool): 边界框是否已归一化到 [0, 1] 范围

        返回:
        float: 边界框的面积
    """
    norm = 1.-float(is_bbox_normalized)
    width = bbox[2]-bbox[0] + norm
    height = bbox[3]-bbox[1] + norm
    return width * height

def jaccard_overlap(pred,gt,is_bbox_normalized=False):
    """
    计算两个边界框的 Jaccard 重叠比率（IoU）
    pred (list): 预测边界框坐标 [xmin, ymin, xmax, ymax]
    gt (list): 实际边界框坐标 [xmin, ymin, xmax, ymax]
    is_bbox_normalized (bool): 边界框是否已归一化到 [0, 1] 范围
    :return:
    """
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
            pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax],
                           is_bbox_normalized)
    pred_size = bbox_area(pred, is_bbox_normalized)
    gt_size = bbox_area(gt, is_bbox_normalized)
    overlap = float(inter_size) / (pred_size + gt_size - inter_size)
    return overlap

class DetectionMAP(object):
    """
    计算检测均值平均精度（mAP）的类
    参数:
    class_num (int): 类别数量
    overlap_thresh (float): 预测边界框和实际边界框之间的重叠阈值，用于确定真正例/假正例，默认为 0.5
    map_type (str): mAP 计算方法，支持 '11point' 和 'integral' 两种方法，默认为 '11point'
    is_bbox_normalized (bool): 边界框是否已归一化到 [0, 1] 范围，默认为 False
    evaluate_difficult (bool): 是否评估困难样本，默认为 False
    """
    def __init__(self,class_num,overlap_thresh=0.5,map_type="11point",is_bbox_normalized=False,evaluate_difficult=False):
        self.class_num = class_num
        self.overlap_thresh = overlap_thresh
        assert map_type in ["11point", "integral"], "map_type currently only support '11point' and 'integral'"
        self.map_type = map_type
        self.is_bbox_normalized = is_bbox_normalized
        self.evaluate_difficult = evaluate_difficult
        self.reset()

    def update(self):
        """
        从给定的预测和实际标注信息中更新度量统计信息
        """
        pass

    def reset(self):
        """
        重置度量统计信息
        """
        pass

    def accumulate(self):
        """
        累积度量结果并计算 mAP
        """
        pass

    def get_map(self):
        """
        获取 mAP 结果
        """
        pass

    def _get_tp_fp_accum(self):
        """
        从[分数，正样本]记录中计算累积的真正例和假正例结果
        """
        pass
