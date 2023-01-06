import torch
import os
import argparse
from build_model import (
    build_densenet121,
    build_resnet50,
    build_inception_v3,
)
from torch.backends import cudnn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from math import cos, pi
from util_fit import fit_one_epoch
from util_data import ReadDiagnosisDataset
from util_augmentation import DiagnosisAugmentation
from loss.LabelSmoothing import LabelSmoothLoss
from config import MEAN, INIT_SIZE, is_label_smoothing, FREEZE_LAYER_NUM
from random import choice


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


def warmup_cosine_lr(epoch):
    if epoch < args.warmup_epoch:
        return (1 + epoch) / args.warmup_epoch
    else:
        return 0.95 ** (epoch - args.warmup_epoch + 1)


def cosine_unfreeze_lr(epoch):
    return (
        args.min_lr
        + 0.5
        * (args.unfreeze_lr - args.min_lr)
        * (1 + cos((epoch - args.unfreeze_epoch) / 8 * pi))
    ) / args.unfreeze_lr


parser = argparse.ArgumentParser(description="Configuration of training parameters.")
parser.add_argument(
    "--basenet",
    default="resnet",
    help="Backbone network for classification.",
    choices=["resnet", "inception", "densenet"],
)
parser.add_argument(
    "--init_batchsize", default=64, type=int, help="Batch size for training."
)
parser.add_argument(
    "--unfreeze_batchsize",
    default=16,
    type=int,
    help="Batch size for unfreeze training.",
)
parser.add_argument(
    "--resume", default=False, type=str2bool, help="Whether to continue training."
)
parser.add_argument(
    "--resume_path", default=None, type=str, help="Resume model path for your training."
)
parser.add_argument("--warmup_epoch", default=4, type=int, help="Warmup epoch.")
parser.add_argument("--init_epoch", default=0, type=int, help="Start epoch.")
parser.add_argument("--unfreeze_epoch", default=40, type=int, help="Unfreeze epoch.")
parser.add_argument("--end_epoch", default=100, type=int, help="End epoch.")
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help=" The Number of workers used in dataloading.",
)
parser.add_argument(
    "--cuda", default=False, type=str2bool, help="Use CUDA to train model."
)
parser.add_argument(
    "--init_lr", default=1e-4, type=float, help="Initial learning rate."
)
parser.add_argument(
    "--unfreeze_lr", default=1e-5, type=float, help="Unfreeze learning rate."
)
parser.add_argument("--min_lr", default=1e-6, type=float, help="Min learning rate.")
parser.add_argument(
    "--save_folder",
    default="weights/",
    help="Directory for saving checkpoint models.",
)
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    print("Generating the {} model.".format(args.basenet))
    if args.basenet == "resnet":
        model = build_resnet50(num_classes=2)
    elif args.basenet == "densenet":
        model = build_densenet121(num_classes=2)
    elif args.basenet == "inception":
        model = build_inception_v3(num_classes=2)

    if args.cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.resume:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.resume_path))
        print("Loading the resume weights {} done.".format(args.resume_path))
    else:
        print("Loading the pretrained weights done.")

    if args.cuda:
        model = model.cuda()

    print("Loading the dataset.")
    train_dataset = ReadDiagnosisDataset(
        is_train=True,
        transforms=DiagnosisAugmentation(
            size=INIT_SIZE[args.basenet][0],
            mean=MEAN,
        ),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.init_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print("Initializing the optimizer,scheduler and the loss function.")
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=5e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    if not is_label_smoothing:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothLoss(smoothing=0.1)

    print("Freeze some layers to train.")
    num_train = train_dataset.__len__()
    epoch_step = num_train // args.init_batchsize
    for step, param in enumerate(model.parameters()):
        if step < FREEZE_LAYER_NUM[args.basenet]:
            param.requires_grad = False
    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        fit_one_epoch(
            epoch=epoch,
            end_epoch=args.unfreeze_epoch,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            epoch_step=epoch_step,
            cuda=args.cuda,
            scheduler=scheduler,
            save_dir=args.save_folder,
        )

    print("The model has been unfreezed.")
    for step, param in enumerate(model.parameters()):
        if step < FREEZE_LAYER_NUM[args.basenet]:
            param.requires_grad = True
    optimizer = Adam(model.parameters(), lr=args.unfreeze_lr, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=cosine_unfreeze_lr)
    epoch_step = num_train // args.unfreeze_batchsize

    for epoch in range(args.unfreeze_epoch, args.end_epoch):
        train_dataset = ReadDiagnosisDataset(
            is_train=True,
            transforms=DiagnosisAugmentation(
                size=choice(INIT_SIZE[args.basenet]), mean=MEAN
            ),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.unfreeze_batchsize,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        fit_one_epoch(
            epoch=epoch,
            end_epoch=args.end_epoch,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            epoch_step=epoch_step,
            cuda=args.cuda,
            scheduler=scheduler,
            save_dir=args.save_folder,
        )


if __name__ == "__main__":
    train()
