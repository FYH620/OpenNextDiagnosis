import numpy as np
import torch
from PIL import Image
from copy import deepcopy


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def preprocess_input(inputs):
    MEANS = (104, 117, 123)
    return inputs - MEANS


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class GetJaccardMatrix(object):
    def __init__(self, boxes_a, boxes_b):
        self.boxes_a = boxes_a
        self.boxes_b = boxes_b
        self.num_boxes_a = self.boxes_a.shape[0]
        self.num_boxes_b = self.boxes_b.shape[0]

    def __call__(self):
        area_a = (self.boxes_a[:, 2] - self.boxes_a[:, 0]) * (
            self.boxes_a[:, 3] - self.boxes_a[:, 1]
        ).unsqueeze(1).expand(self.num_boxes_a, self.num_boxes_b)
        area_b = (self.boxes_b[:, 2] - self.boxes_b[:, 0]) * (
            self.boxes_b[:, 3] - self.boxes_b[:, 1]
        ).unsqueeze(0).expand(self.num_boxes_a, self.num_boxes_b)
        inter = self._get_intersact()
        union = area_a + area_b - inter
        return inter / union

    def _get_intersact(self):
        left_top_coord = torch.max(
            self.boxes_a[:, :2]
            .unsqueeze(1)
            .expand(self.num_boxes_a, self.num_boxes_b, 2),
            self.boxes_b[:, :2]
            .unsqueeze(0)
            .expand(self.num_boxes_a, self.num_boxes_b, 2),
        )
        right_bottom_coord = torch.min(
            self.boxes_a[:, 2:]
            .unsqueeze(1)
            .expand(self.num_boxes_a, self.num_boxes_b, 2),
            self.boxes_b[:, 2:]
            .unsqueeze(0)
            .expand(self.num_boxes_a, self.num_boxes_b, 2),
        )
        intersact_width_and_height = torch.clamp(
            (right_bottom_coord - left_top_coord), min=0
        )
        return intersact_width_and_height[:, :, 0] * intersact_width_and_height[:, :, 1]


def soft_nms(
    boxes,
    scores_raw,
    soft_threshold=0.01,
    iou_threshold=0.7,
    weight_method="guassian",
    sigma=0.5,
):
    scores = deepcopy(scores_raw)
    keep = []
    ids = scores.argsort()
    while ids.numel() > 0:
        ids = scores.argsort()
        if ids.size(0) == 1:
            keep.append(ids[-1])
            break
        keep_len = len(keep)
        max_score_index = ids[-(keep_len + 1)]
        keep.append(max_score_index)
        ids = ids[: -(keep_len + 1)]
        max_score_box = boxes[max_score_index, :].unsqueeze(0)
        other_boxes = boxes[ids]
        iou = GetJaccardMatrix(max_score_box, other_boxes)().squeeze(0)
        if weight_method == "linear":
            ge_threshold_bool = iou >= iou_threshold
            ge_threshold_ids = ids[ge_threshold_bool]
            scores[ge_threshold_ids] *= 1.0 - iou[ge_threshold_bool]
        elif weight_method == "guassian":
            scores[ids] *= torch.exp(-(iou * iou) / sigma)
        else:
            raise ValueError(
                'Given a wrong parameter "{}",only "linear" and "guassian" are supported.'.format(
                    str(weight_method)
                )
            )
    keep = ids.new(keep)
    keep = keep[scores[keep] > soft_threshold]
    boxes = boxes[keep]
    scores = scores[keep]
    return keep, boxes, scores
