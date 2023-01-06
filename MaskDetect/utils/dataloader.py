import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input
from config import (
    mix_up_prob,
    is_mix_up,
    mosiac_prob,
    is_mosiac,
    is_random_paste,
    paste_area_threshold,
    paste_prob,
)


class SSDDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        anchors,
        batch_size,
        num_classes,
        train,
        overlap_threshold=0.5,
    ):
        super(SSDDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.overlap_threshold = overlap_threshold
        self.mixup = is_mix_up
        self.mosiac = is_mosiac
        self.paste = is_random_paste

    def __len__(self):
        return self.length

    def pull_item(self, index):
        index = index % self.length
        image, box = self.get_random_data(
            self.annotation_lines[index], self.input_shape, random=self.train
        )
        image_data = np.transpose(
            preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1)
        )
        if len(box) != 0:
            box = np.array(box[:, :], dtype=np.float32)
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
        return np.array(image_data), np.array(box)

    def __getitem__(self, index):

        if self.mixup and self.train and np.random.uniform(0, 1) <= mix_up_prob:
            index_mix = np.random.randint(self.length)
            img_mix, box_mix = self.pull_item(index_mix)
            img, box = self.pull_item(index)
            lamb = np.random.beta(32, 32)
            img_mix_together = img * lamb + img_mix * (1 - lamb)
            box_mix_together = np.concatenate([box, box_mix], axis=0)
            one_hot_label = np.eye(self.num_classes - 1)[
                np.array(box_mix_together[:, 4], np.int32)
            ]
            box_mix_together = np.concatenate(
                [box_mix_together[:, :4], one_hot_label], axis=-1
            )
            box_mix_together = self.assign_boxes(box_mix_together)
            return np.array(img_mix_together), np.array(box_mix_together)

        elif self.mosiac and self.train and np.random.uniform(0, 1) <= mosiac_prob:
            img_t = np.zeros((3, 600, 600), dtype=np.float)
            index_mix = np.random.randint(self.length, size=3)
            img_a, box_a = self.pull_item(index)
            img_b, box_b = self.pull_item(index_mix[0])
            img_c, box_c = self.pull_item(index_mix[1])
            img_d, box_d = self.pull_item(index_mix[2])
            box_a, box_b, box_c, box_d = (
                box_a.astype("float64"),
                box_b.astype("float64"),
                box_c.astype("float64"),
                box_d.astype("float64"),
            )
            box_a[:, :4] *= 0.5
            box_b[:, :4] *= 0.5
            box_b[:, [0, 2]] += 0.5
            box_c[:, :4] *= 0.5
            box_c[:, [1, 3]] += 0.5
            box_d[:, :4] = box_d[:, :4] * 0.5 + 0.5
            box_mix = np.concatenate([box_a, box_b, box_c, box_d], axis=0)
            img_t[:, :300, :300] = img_a
            img_t[:, :300, 300:600] = img_b
            img_t[:, 300:600, :300] = img_c
            img_t[:, 300:600, 300:600] = img_d
            img_mix = cv2.resize(img_t.transpose((1, 2, 0)), dsize=(300, 300))
            one_hot_label = np.eye(self.num_classes - 1)[
                np.array(box_mix[:, 4], np.int32)
            ]
            box_mix = np.concatenate([box_mix[:, :4], one_hot_label], axis=-1)
            box_mix = self.assign_boxes(box_mix)
            return np.array(img_mix).transpose((2, 0, 1)), np.array(box_mix)

        elif self.paste and self.train and np.random.uniform(0, 1) <= paste_prob:
            img, box = self.pull_item(index)
            area_box = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
            small_boxes = box[area_box <= paste_area_threshold]
            new_boxes = []
            for small_box in small_boxes:
                while True:
                    new_box = small_box
                    random_dx, random_dy = np.random.rand(), np.random.rand()
                    if np.random.uniform(0, 1) <= 0.5:
                        new_box[[0, 2]] = small_box[[0, 2]] + random_dx
                        new_box[[1, 3]] = small_box[[1, 3]] + random_dy
                        if (new_box[2] > 1) or (new_box[3] > 1):
                            continue
                    else:
                        new_box[[0, 2]] = small_box[[0, 2]] - random_dx
                        new_box[[1, 3]] = small_box[[1, 3]] - random_dy
                        if (new_box[0] < 0) or (new_box[1] < 0):
                            continue
                    _, h, w = img.shape
                    new_box_up = int(h * new_box[1])
                    new_box_left = int(w * new_box[0])
                    small_box_up = int(h * small_box[1])
                    small_box_left = int(w * small_box[0])
                    box_dw = int(w * (small_box[2] - small_box[0]))
                    box_dh = int(h * (small_box[3] - small_box[1]))
                    img[
                        :,
                        new_box_up : new_box_up + box_dh,
                        new_box_left : new_box_left + box_dw,
                    ] = img[
                        :,
                        small_box_up : small_box_up + box_dh,
                        small_box_left : small_box_left + box_dw,
                    ]
                    new_boxes.append(new_box)
                    break
            new_boxes = np.array(new_boxes)
            if len(new_boxes) != 0:
                box_mix = np.concatenate([new_boxes, box], axis=0)
            else:
                box_mix = box
            one_hot_label = np.eye(self.num_classes - 1)[
                np.array(box_mix[:, 4], np.int32)
            ]
            box_mix = np.concatenate([box_mix[:, :4], one_hot_label], axis=-1)
            box_mix = self.assign_boxes(box_mix)
            return img, np.array(box_mix)

        else:
            img, box = self.pull_item(index)
            one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]
            box = np.concatenate([box[:, :4], one_hot_label], axis=-1)
            box = self.assign_boxes(box)
            return np.array(img), np.array(box)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(
        self,
        annotation_line,
        input_shape,
        jitter=0.3,
        hue=0.1,
        sat=1.5,
        val=1.5,
        random=True,
    ):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
            return image_data, box
        new_ar = (
            w
            / h
            * self.rand(1 - jitter, 1 + jitter)
            / self.rand(1 - jitter, 1 + jitter)
        )
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    def iou(self, box):
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (
            self.anchors[:, 3] - self.anchors[:, 1]
        )
        union = area_true + area_gt - inter
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_anchors = self.anchors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_anchors_center = (
            assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]
        ) * 0.5
        assigned_anchors_wh = assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2]
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[
            best_iou_idx, np.arange(assign_num), :4
        ]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -1][best_iou_mask] = 1
        return assignment


def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes
