import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from seaborn.matrix import heatmap
from util_data import ReadDiagnosisDataset
from util_augmentation import BaseTransform
from time import localtime, strftime
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torchvision.models import resnet50, inception_v3, densenet121
from config import INIT_SIZE, MEAN, LABEL_TO_IDX


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


parser = argparse.ArgumentParser(description="Configuration of testing parameters.")
parser.add_argument(
    "--basenet",
    default="resnet",
    help="Backbone network for classification.",
    choices=["resnet", "inception", "densenet"],
)
parser.add_argument(
    "--model_path",
    default="model_data/best.pth",
    type=str,
    help="Trained model for prediction.",
)
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help=" The Number of workers used in dataloading.",
)
parser.add_argument(
    "--batchsize",
    default=4,
    type=int,
    help="Batch size for testing.",
)
parser.add_argument(
    "--cuda", default=False, type=str2bool, help="Use CUDA to train model."
)
parser.add_argument(
    "--save_folder",
    default="evaluation/",
    help="Directory for save evaluation results.",
)
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def plot_confusion_matrix(y_test, predict, class_names):
    conf_matrix = confusion_matrix(y_test, predict)
    data_frame_confusion_matrix = pd.DataFrame(
        conf_matrix, index=class_names, columns=class_names
    )

    heatmap = sns.heatmap(
        data_frame_confusion_matrix, annot=True, fmt="d", cmap="YlGnBu"
    )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right"
    )

    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.savefig(args.save_folder + "confusion_matrix.jpg")
    plt.show()


def test():
    print("Generating the {} model.".format(args.basenet))
    if args.basenet == "resnet":
        model = resnet50(num_classes=2)
    elif args.basenet == "densenet":
        model = densenet121(num_classes=2)
    elif args.basenet == "inception":
        model = inception_v3(num_classes=2)
    model = torch.nn.DataParallel(model)

    print("Loading the pretrained weights.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    if args.cuda:
        model = model.cuda()

    print("Start testing.")
    multi_pred = []
    for i in range(len(INIT_SIZE[args.basenet])):
        test_dataset = ReadDiagnosisDataset(
            is_train=False,
            transforms=BaseTransform(size=INIT_SIZE[args.basenet][i], mean=MEAN),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        pred_labels = []
        real_labels = []
        for imgs, labels in test_dataloader:
            model.eval()
            with torch.no_grad():
                if args.cuda:
                    imgs = imgs.float().cuda()
                    labels = labels.long().cuda()
                else:
                    imgs = imgs.float()
                    labels = labels.long()
            output = model(imgs)
            pred = torch.argmax(output, dim=1)
            pred_labels += [*pred.data.cpu()]
            real_labels += [*labels.data.cpu()]
        multi_pred += [pred_labels]

    multi_pred, real_labels = np.array(multi_pred), np.array(real_labels)
    pred_labels = multi_pred.shape[0] - np.sum(multi_pred, axis=0) < np.sum(
        multi_pred, axis=0
    )
    accuracy = accuracy_score(real_labels, pred_labels)
    precision = precision_score(real_labels, pred_labels)
    recall = recall_score(real_labels, pred_labels)
    f1 = f1_score(real_labels, pred_labels)
    print("accuracy:{}".format(accuracy))
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("f1_score:{}".format(f1))

    with open(args.save_folder + "results.txt", "a") as f:
        f.write(
            "test_time:{}".format(strftime("%Y-%m-%d %H:%M:%S", localtime())) + "\n"
        )
        f.write("trained_model_name:{}".format(args.model_path) + "\n")
        f.write("accuracy:{}".format(accuracy) + "\n")
        f.write("precision:{}".format(precision) + "\n")
        f.write("recall:{}".format(recall) + "\n")
        f.write("f1_score:{}".format(f1) + "\n")
    plot_confusion_matrix(real_labels, pred_labels, LABEL_TO_IDX.keys())
    print("Test done.")


if __name__ == "__main__":
    test()
