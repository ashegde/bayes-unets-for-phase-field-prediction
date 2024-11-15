import glob
import os
import pickle 

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch.func as tf

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
    features = 16

    model = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
    )
    model.load_state_dict(
        torch.load(
            path_to_model,
            map_location=device,
            weights_only=True,
        )
    )
    return model.to(device)




# Setup results directory
results_path = 'results'
os.makedirs(results_path, exist_ok=True)

# Load model
path_to_model = glob.glob('model_*/checkpoint*.pt')[0]
t_skip = int(path_to_model.split('_')[-1][:-3] )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = load_model(path_to_model=path_to_model, device=device)

# Training loss function
loss_fn = MSELoss(reduction='mean')

# Load training and test datasets
train_dataset = H5Dataset(path='data', mode='train', skip=t_skip)
test_dataset = H5Dataset(path='data', mode='valid')

# Extract nominal model parameters
params = {k: v.detach() for k, v in model.named_parameters()}

# Wrappers for single point evaluations
def model_single(params, x):
    return tf.functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)

def loss_single(params, x, y):
    pred = model_single(params, x)
    return loss_fn(pred, y)

#
n_post_samples = 30