import os
import torch


def save_net(file_path, file_name, model, optimizer=None):
    """
        In this function, a model is saved.
        ------------------------------------------------
        Parameters:
            - file_path (str): saving path
            - file_name (str): saving name
            - model (torch.nn.Module) 
            - optimizer (torch.optim)
    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()

    torch.save(state_dict, os.path.join(file_path, file_name))
