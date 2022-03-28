# from sre_parse import FLAGS
import torch
# import utils as utils
import nets as nets
import deeplearning as dl
import dataloaders as data
import torch.optim as optim
from torchvision import transforms


def _main(args):

    net = nets.ClassificationNetwork(1000)

    lr = 1e-5
    gpu = False
    epoch, batch_size = 20, 2
    num_worker = 2

    saving_path, saving_prefix = "./checkpoints", "checkpoint_"
    saving_frequency = 1

    train_base_dir, train_df_path = (
        "./dataset/classification/train1000", "./dataset/classification/train.csv")
    val_base_dir, val_df_path = (
        "./dataset/classification/test1000", "./dataset/classification/test.csv")

    input_transformation = transforms.Compose([transforms.Resize(100)])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    train_dataloader = data.ClassifierDataLoader(
        train_base_dir, train_df_path, input_transformation)
    val_dataloader = data.ClassifierDataLoader(
        val_base_dir, val_df_path, input_transformation)

    net, report = dl.classification_train(
        net=net,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=epoch,
        batch_size=batch_size,
        num_workers=num_worker,
        saving_path=saving_path,
        saving_prefix=saving_prefix,
        saving_frequency=saving_frequency,
        # saving_model_every_epoch=False,
        gpu=gpu)


if __name__ == "__main__":
    # args = utils.get_args()
    # _main(args)
    _main(None)
