import os
import numpy as np


############################ train config ###############################
is_label_smoothing = False
is_mix_up = False
is_random_erasing = False
INIT_SIZE = {
    "resnet": [224, 256, 288, 320, 352, 388, 416, 448],
    "densenet": [224],
    "inception": [299],
}
FREEZE_LAYER_NUM = {"resnet": 72, "densenet": 114, "inception": 128}
#########################################################################

########################## auxiliary config #############################
MODEL_DIR = {
    "resnet": "./model_data/ResNet50.pth",
    "densenet": "./model_data/DenseNet121.pth",
    "inception": "./model_data/InceptionV3.pth",
}
HOME = os.getcwd()
LABEL_TO_IDX = {"positive": 0, "negative": 1}
IDX_TO_LABEL = {0: "positive", 1: "negative"}
MEAN = np.array([123, 117, 104], dtype=np.uint8)
#########################################################################
