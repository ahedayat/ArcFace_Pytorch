import os
import utils as utils
from datetime import datetime, date


import torch
import nets as nets
import deeplearning as dl
import dataloaders as data
import torch.optim as optim
from torchvision import transforms, models


def save_report(df, backbone_name, saving_path):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "{}_{}_{}_{}_{}_{}_{}".format(
        backbone_name, year, month, day, hour, minute, second)

    df.to_csv(os.path.join(saving_path, report_name))


def _main(args):
    # Hardware
    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # CNN Backbone
    assert args.backbone in ["resnet18", "resnet34",
                             "resnet50"], "CNN backbone must be one of this items: ['resnet18', 'resnet34', 'resnet50']"
    backbone = None
    if args.backbone == "resnet18":
        backbone = models.resnet18(pretrained=args.pretrained, progress=True)
    elif args.backbone == "resnet34":
        backbone = models.resnet34(pretrained=args.pretrained, progress=True)
    else:
        backbone = models.resnet50(pretrained=args.pretrained, progress=True)

    net = nets.ClassificationNetwork(backbone, 1000)

    if args.gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            net = net.cuda(device=device)

    # Input Size
    input_transformation = transforms.Compose([transforms.Resize(100)])

    # Data Path
    # - Train
    train_base_dir, train_df_path = args.train_base_dir, args.train_df_path
    train_dataloader = data.ClassifierDataLoader(
        train_base_dir, train_df_path, input_transformation)

    # - Validation
    val_base_dir, val_df_path = args.val_base_dir, args.val_df_path
    val_dataloader = data.ClassifierDataLoader(
        val_base_dir, val_df_path, input_transformation)

    # Loss Function
    assert args.criterion in [
        'arcface', 'cross_entropy'], "Loss Function must be one of this items: ['arcface', 'cross_entropy']"

    if args.criterion == "arcface":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    assert args.optimizer in [
        "sgd", "adam"], "Optimizer must be one of this items: ['sgd', 'adam']"

    if args.optimizer == "sgd":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Checkpoint Address
    saving_path, saving_prefix = args.ckpt_path, args.ckpt_prefix
    saving_frequency = args.save_freq

    # Training
    net, report = dl.classification_train(
        net=net,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        saving_path=saving_path,
        saving_prefix=saving_prefix,
        saving_frequency=saving_frequency,
        # saving_model_every_epoch=False,
        gpu=args.gpu)

    save_report(df=report, backbone_name=args.backbone,
                saving_path=args.report)


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
