import argparse
import torch
import cv2
import numpy as np
from util_augmentation import BaseTransform
from torchvision.models import inception_v3, resnet50, densenet121
from config import INIT_SIZE, MEAN, IDX_TO_LABEL


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


parser = argparse.ArgumentParser(description="Configure of predicting parameters.")
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
    "--cuda", default=False, type=str2bool, help="Use CUDA to train model."
)
parser.add_argument(
    "--img_path",
    default="dataset/test/MIDRC-RICORD-1C-SITE2-000125-08616-1.png",
    type=str,
    help="The image path for prediction.",
)
args = parser.parse_args()


def predict():
    if args.basenet == "resnet":
        model = resnet50(num_classes=2)
    elif args.basenet == "densenet":
        model = densenet121(num_classes=2)
    elif args.basenet == "inception":
        model = inception_v3(num_classes=2)
    model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    if args.cuda:
        model = model.cuda()

    model.eval()
    print("init done.")
    while True:
        img_path = input()
        raw_img = cv2.imread(img_path, flags=1)
        preds = []
        for i in range(len(INIT_SIZE[args.basenet])):
            with torch.no_grad():
                img = BaseTransform(size=INIT_SIZE[args.basenet][i], mean=MEAN)(
                    raw_img[:, :, (2, 1, 0)]
                )
                img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            output = model(img)
            pred = torch.argmax(output, dim=1)
            preds += [*pred.data]
        pred = (len(preds) - np.sum(preds)) < np.sum(preds)
        class_name = IDX_TO_LABEL[int(pred.item())]
        print(class_name)
        print("one image done.")


if __name__ == "__main__":
    predict()
