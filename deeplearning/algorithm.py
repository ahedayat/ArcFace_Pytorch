"""
In this file, the basic function for training and evaluating `Classification Network` and `Siamese Network`
"""
import pandas as pd

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable as V

import nets as nets


def classification_train(
    net,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    epoch=1,
    batch_size=16,
    num_workers=1,
    saving_path=None,
    saving_prefix="checkpoint_",
    saving_frequency=1,
    # saving_model_every_epoch=False,
        gpu=False):
    """
    Training Classification network
    --------------------------------------------------
    Parameters:
        - net (nets.ClassificationNetwork)
            * Classification Netwrok

        - train_dataloader (dataloaders.ClassifierDataLoader)
            * Data loader for train set

        - val_dataloader (dataloaders.ClassifierDataLoader)
            * Data loader for validation set

        - optimizer (torch.optim)
            * Optimizer Algorithm

        - device (torch.device)
            * Device for training network

        - epoch (int)
            * Number of training epochs

        - batch_size (int)
            * Data loading batch size

        - num_workers (int)
    """

    train_dataloader = DataLoader(dataset=train_dataloader,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=gpu and torch.cuda.is_available(),
                                  num_workers=num_workers
                                  )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc"])

    net = net.float()

    for e in range(epoch):
        net.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for (X, Y) in tepoch:
                optimizer.zero_grad()

                tepoch.set_description(f"Epoch {e}")

                # X = torch.unsqueeze(X, dim=1).float()
                # X = X.float

                # if isinstance(criterion, nn.CrossEntropyLoss):
                #     Y = torch.tensor(Y, dtype=torch.float)
                if isinstance(criterion, nn.NLLLoss):
                    Y = torch.argmax(Y, dim=1)
                    # Y = torch.tensor(Y, dtype=torch.long)

                X, Y = V(X), V(Y)

                if device != 'cpu' and gpu and torch.cuda.is_available():
                    if device.type == 'cuda':
                        X, Y = X.cuda(device=device), Y.cuda(device=device)
                    elif device == 'multi':
                        X, Y = nn.DataParallel(X), nn.DataParallel(Y)

                out = net(X)

                correct = (torch.argmax(out, dim=1) ==
                           torch.argmax(Y, dim=1)).sum().item()
                accuracy = (correct / batch_size)*100

                loss = criterion(out, Y)

                current_report = pd.DataFrame({
                    "epoch": epoch,
                    "train/eval": "eval",
                    "batch_size": Y.shape[0],
                    "loss": loss.item(),
                    "acc": accuracy})

                report = pd.concat([report, current_report])

                accuracy = int(accuracy)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(
                    loss="{:.3f}".format(loss.item()),
                    accuracy=f"{accuracy:03d}%")

        val_report = classification_eval(
            net=net,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu=gpu,
            tqbar_description=f"Validation @ Epoch  {e}",
            epoch=e
        )

        report = pd.concat([report, val_report])

        if e % saving_frequency == 0:
            nets.save(
                file_path=saving_path,
                file_name="{}_epoch_{}".format(saving_prefix, e),
                model=net,
                optimizer=optimizer
            )

    return net, report


def classification_eval(
    net,
    dataloader,
    criterion,
    device,
    batch_size=16,
    num_workers=1,
    gpu=False,
    tqbar_description="Test",
    epoch=None
):
    """
    Evaluation Function
    """
    dataloader = DataLoader(dataset=dataloader,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=gpu and torch.cuda.is_available(),
                            num_workers=num_workers
                            )

    report = pd.DataFrame(
        columns=["epoch", "train/eval", "batch_size", "loss", "acc"])

    net = net.float()
    net.train(mode=False)

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, Y) in tepoch:
            tepoch.set_description(tqbar_description)

            # X = torch.unsqueeze(X, dim=1).float()

            # if isinstance(criterion, nn.CrossEntropyLoss):
            #     Y = torch.tensor(Y, dtype=torch.float)
            if isinstance(criterion, nn.NLLLoss):
                Y = torch.argmax(Y, dim=1)
                # Y = torch.tensor(Y, dtype=torch.long)

            X, Y = V(X), V(Y)

            if device != 'cpu' and gpu and torch.cuda.is_available():
                if device.type == 'cuda':
                    X, Y = X.cuda(device=device), Y.cuda(device=device)
                elif device == 'multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)

            out = net(X)

            correct = (torch.argmax(out, dim=1) ==
                       torch.argmax(Y, dim=1)).sum().item()
            accuracy = (correct / batch_size)*100

            loss = criterion(out, Y)

            current_report = pd.DataFrame({
                "epoch": epoch,
                "train/eval": "eval",
                "batch_size": Y.shape[0],
                "loss": loss.item(),
                "acc": accuracy})

            report = pd.concat([report, current_report])

            accuracy = int(accuracy)

            tepoch.set_postfix(
                loss="{:.3f}".format(loss.item()),
                accuracy=f"{accuracy:03d}")

    return report


def verification(
        net,
        dataloader,
        distance,
        device,
        batch_size=16,
        num_workers=1,
        gpu=False,
        tqbar_description="Verification"):

    dataloader = DataLoader(dataset=dataloader,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=gpu and torch.cuda.is_available(),
                            num_workers=num_workers
                            )

    report = pd.DataFrame(
        columns=["image_A", "image_B", "distance"])

    net = net.float()
    net.train(mode=False)

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X_1, X_2, file_1, file_2, _) in tepoch:
            tepoch.set_description(tqbar_description)

            X_1, X_2 = V(X_1), V(X_2)

            if device != 'cpu' and gpu and torch.cuda.is_available():
                if device.type == 'cuda':
                    X, Y = X.cuda(device=device), Y.cuda(device=device)
                elif device == 'multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)

            feature_1 = net(X_1)
            feature_2 = net(X_1)

            dist = distance(feature_1, feature_2)

            file_1 = list(file_1)
            file_2 = list(file_2)

            current_report = pd.DataFrame({
                "image_A": file_1,
                "image_B": file_2,
                "distance": dist.tolist()
            })

            # import pdb
            # pdb.set_trace()
            report = pd.concat([report, current_report])

            # tepoch.set_postfix(
            #     loss="{:.3f}".format(loss.item()),
            #     accuracy=f"{accuracy:03d}")

    return report
