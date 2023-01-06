import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.ssd import SSD300
from torch.optim.lr_scheduler import LambdaLR
from nets.ssd_training import MultiboxLoss, weights_init
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import get_classes
from math import cos, pi
from utils.utils_fit import fit_one_epoch

warnings.filterwarnings("ignore")


def warmup_cos_lr_freeze(epoch):
    if epoch < Warmup_Epoch:
        return (1 + epoch) / Warmup_Epoch
    else:
        return 0.94 ** (epoch - Warmup_Epoch)


def warmup_cos_lr_unfreeze(epoch):
    return (
        Min_lr
        + 0.5 * (Unfreeze_lr - Min_lr) * (1 + cos((epoch - Freeze_Epoch) / 5 * pi))
    ) / Unfreeze_lr


if __name__ == "__main__":
    Cuda = True
    classes_path = "model_data/voc_classes.txt"
    model_path = "model_data/ssd_weights.pth"
    input_shape = [300, 300]
    backbone = "vgg"
    pretrained = True
    # 512：[36, 77, 154, 230, 307, 384, 460]
    # 300：[21, 45, 99, 153, 207, 261, 315]
    anchors_size = [21, 45, 99, 153, 207, 261, 315]
    Init_Epoch = 0
    Warmup_Epoch = 5
    Freeze_Epoch = 50
    UnFreeze_Epoch = 150
    Init_lr = 5e-4
    Unfreeze_lr = 3e-5
    Min_lr = 1e-6
    Freeze_batch_size = 16
    Unfreeze_batch_size = 8
    Freeze_Train = True
    num_workers = 6
    train_annotation_path = "2007_train.txt"
    val_annotation_path = "2007_val.txt"

    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, anchors_size, backbone)
    model = SSD300(num_classes, backbone, pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != "":
        print("Load weights {}.".format(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if np.shape(model_dict[k]) == np.shape(v)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    criterion = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    loss_history = LossHistory("logs/")
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        batch_size = Freeze_batch_size
        lr = Init_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        optimizer = optim.Adam(model_train.parameters(), Init_lr, weight_decay=5e-4)
        lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_cos_lr_freeze)
        train_dataset = SSDDataset(
            train_lines, input_shape, anchors, batch_size, num_classes, train=True
        )
        val_dataset = SSDDataset(
            val_lines, input_shape, anchors, batch_size, num_classes, train=False
        )
        gen = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ssd_dataset_collate,
        )
        gen_val = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ssd_dataset_collate,
        )
        if Freeze_Train:
            if backbone == "vgg":
                for param in model.vgg[:28].parameters():
                    param.requires_grad = False
            else:
                for param in model.mobilenet.parameters():
                    param.requires_grad = False
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(
                model_train,
                model,
                criterion,
                loss_history,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                gen,
                gen_val,
                end_epoch,
                Cuda,
            )
            lr_scheduler.step()
    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        optimizer = optim.Adam(model_train.parameters(), Unfreeze_lr, weight_decay=1e-4)
        lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_cos_lr_unfreeze)
        train_dataset = SSDDataset(
            train_lines, input_shape, anchors, batch_size, num_classes, train=True
        )
        val_dataset = SSDDataset(
            val_lines, input_shape, anchors, batch_size, num_classes, train=False
        )
        gen = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ssd_dataset_collate,
        )
        gen_val = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ssd_dataset_collate,
        )
        if Freeze_Train:
            if backbone == "vgg":
                for param in model.vgg[:28].parameters():
                    param.requires_grad = True
            else:
                for param in model.mobilenet.parameters():
                    param.requires_grad = True
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(
                model_train,
                model,
                criterion,
                loss_history,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                gen,
                gen_val,
                end_epoch,
                Cuda,
            )
            lr_scheduler.step()
