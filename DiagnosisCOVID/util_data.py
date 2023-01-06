import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import LABEL_TO_IDX, HOME


class ReadDiagnosisDataset(Dataset):
    def __init__(self, is_train, transforms):
        super(ReadDiagnosisDataset, self).__init__()
        self.is_train = is_train
        self.transforms = transforms
        self.image_paths, self.image_labels = self.read_texts_annotations()

    def __getitem__(self, index):
        raw_one_img, raw_one_label = self.pull_item(index)
        one_img, one_label = self.transforms(raw_one_img), raw_one_label
        return torch.from_numpy(one_img).permute(2, 0, 1), one_label

    def __len__(self):
        return len(self.image_labels)

    def pull_item(self, index):
        raw_one_img = cv2.imread(self.image_paths[index])
        raw_one_label = self.image_labels[index]
        return raw_one_img[:, :, (2, 1, 0)], raw_one_label

    def read_texts_annotations(self):

        file_name = "train.txt" if self.is_train else "test.txt"
        dir_name = "train" if self.is_train else "test"

        with open(os.path.join(HOME, "dataset", file_name)) as f:
            contents = f.readlines()
            labels, paths = [], []

            for content in contents:
                information = content.split(" ")
                path = information[1]
                label = information[2].strip().lower()
                if label not in LABEL_TO_IDX.keys():
                    raise ValueError(
                        "Wrong with your label:check the file {}".format(path)
                    )

                paths += [os.path.join(HOME, "dataset", dir_name, path)]
                labels += [LABEL_TO_IDX[label]]
        return paths, labels
