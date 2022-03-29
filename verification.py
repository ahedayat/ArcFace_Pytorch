import os
import utils as utils
from datetime import datetime, date
import nets.distances as distances


import torch
import nets as nets
import deeplearning as dl
import dataloaders as data
import torch.optim as optim
from torchvision import transforms, models


def save_report(df, saving_path):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "verification_{}_{}_{}_{}_{}_{}".format(
        year, month, day, hour, minute, second)

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

    net = nets.ClassificationNetwork(backbone, args.input_size)

    # net.load_state_dict(args.ckpt_load)
    net.embedding()
    net.train(mode=False)

    if args.gpu and torch.cuda.is_available():
        if device.type == 'cuda':
            net = net.cuda(device=device)

    # Input Size
    input_transformation = transforms.Compose([transforms.Resize(100)])

    # Data Path
    # - verification
    verify_base_dir, verify_df_path = args.verify_base_dir, args.verify_df_path

    verfication_dataloader = data.VerifierDataLoader(
        base_dir=verify_base_dir, csv_path=verify_df_path, transformation=input_transformation)

    # Distance Metric
    assert args.distance in [
        "cosine", "euclidean"], "Distance Metrics must be one of this item: ['cosine', 'euclidean']"

    _distance = None
    if args.distance == "cosine":
        _distance = distances.CosineDist()
    elif args.distance == "euclidean":
        _distance = distances.EuclideanDist()

    # Verification
    report = dl.verification(
        net=net,
        dataloader=verfication_dataloader,
        distance=_distance,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu=args.gpu,
        tqbar_description="Verfication"
    )

    save_report(df=report, saving_path=args.report)


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
