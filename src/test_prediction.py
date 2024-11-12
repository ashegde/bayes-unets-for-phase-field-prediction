import argparse
from datetime import datetime
import glob
import os
import pickle 

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from pipeline.dataset.loaders import H5Dataset
from pipeline.model.model import UNet2d
from pipeline.inference.sampler import 
from pipeline.inference.rollout import
from simulator.simulator import CahnHilliardSimulator


def load_model(path_to_model: str, device: torch.device) -> torch.nn.Module:
    """
    Initializes the UNet model with predefined input and output channels.

    Args:
        path_to_model : str
            Path to model
        device : torch.device
            The device to move the model to ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded UNet model.
    """
    in_channels = 1
    out_channels = in_channels
    init_features = 16

    model = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
    )
    model.load_state_dict(
        torch.load(path_to_model, map_location=device)
    )
    return model.to(device)


# Setting Seeds
seed_val = 2023
torch.manual_seed(seed_val)
np.random.seed(seed_val)

# Setup results directory
results_path = 'results'
os.makedirs(results_path, exist_ok=True)

# load model
model_path = glob.glob('model_*/')
path_to_model = glob.glob(model_path+'/checkpoint*.pt')
t_skip = int(saved_model[0].split('_')[-1][:-3] )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = load_model(path_to_model=path_to_model, device=device)

# load training and test datasets
train_dataset = H5Dataset(path='data', mode='train', skip=t_skip)
test_dataset = H5Dataset(path='data', mode='valid')


#