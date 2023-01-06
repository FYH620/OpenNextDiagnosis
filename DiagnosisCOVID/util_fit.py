import torch
from tqdm import tqdm
import numpy as np
from config import is_mix_up


def fit_one_epoch(
    epoch,
    end_epoch,
    model,
    dataloader,
    criterion,
    optimizer,
    epoch_step,
    cuda,
    scheduler,
    save_dir,
):

    train_loss = 0
    train_corrects = 0
    train_num = 0
    val_loss = 0
    val_corrects = 0
    val_num = 0

    with tqdm(
        total=epoch_step,
        desc=f"Epoch {epoch + 1}/{end_epoch}",
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for iteration, (imgs, labels) in enumerate(dataloader):
            with torch.no_grad():
                if cuda:
                    imgs = imgs.float().cuda()
                    labels = labels.long().cuda()
                else:
                    imgs = imgs.float()
                    labels = labels.long()

            if iteration < epoch_step * 0.9:
                model.train()
                if is_mix_up:
                    lamb = np.random.beta(32, 32)
                    index = (
                        torch.randperm(imgs.size(0)).cuda()
                        if cuda
                        else torch.randperm(imgs.size(0))
                    )
                    imgs = lamb * imgs + (1 - lamb) * imgs[index]
                    label_a, label_b = labels, labels[index]
                    output = model(imgs)
                    pred = torch.argmax(output, dim=1)
                    loss = lamb * criterion(output, label_a) + (1 - lamb) * criterion(
                        output, label_b
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * imgs.size(0)
                    train_corrects += torch.sum(pred == labels.data)
                    train_num += imgs.size(0)
                else:
                    model.train()
                    output = model(imgs)
                    pred = torch.argmax(output, dim=1)
                    loss = criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * imgs.size(0)
                    train_corrects += torch.sum(pred == labels.data)
                    train_num += imgs.size(0)
                pbar.set_postfix(
                    {
                        "train_loss": train_loss / train_num,
                        "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )
                pbar.update(1)
            else:
                model.eval()
                output = model(imgs)
                pred = torch.argmax(output, 1)
                loss = criterion(output, labels)
                val_num += imgs.size(0)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += torch.sum(pred == labels.data)
                pbar.set_postfix(
                    {
                        "val_loss": val_loss / val_num,
                        "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )
                pbar.update(1)

        if scheduler is not None:
            scheduler.step()
        if epoch < 40 and epoch % 2 == 0:
            torch.save(
                model.state_dict(),
                save_dir
                + f"ep{epoch+1}-train{train_corrects/train_num}-val{val_corrects/val_num}.pth",
            )
        if epoch >= 40:
            torch.save(
                model.state_dict(),
                save_dir
                + f"ep{epoch+1}-train{train_corrects/train_num}-val{val_corrects/val_num}.pth",
            )
        print(
            f"TRAIN_ACC:{train_corrects / train_num};VAL_ACC:{val_corrects / val_num}"
        )
        print(f"TRAIN_LOSS:{train_loss / train_num};VAL_LOSS:{val_loss / val_num}")
