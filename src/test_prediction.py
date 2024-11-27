"""
In this module, we use the trained surrogate model to predict
simulation runs (with the coarsened time step).

To do: incorporate UQ in a cost effective manner.
"""

import glob
import os

import numpy as np
import torch
from torch.nn import MSELoss
import torch.func as tf

from pipeline.dataset.loaders import H5Dataset
from pipeline.model.model import UNet2d
# from pipeline.inference.sampler import
from pipeline.inference.prediction import run_surrogate
from pipeline.postprocess.plotting import create_anim


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

    net = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
    )
    net.load_state_dict(
        torch.load(
            path_to_model,
            map_location=device,
            weights_only=True,
        )
    )
    return net.to(device)


# Setup results directory
results_path = 'results'
os.makedirs(results_path, exist_ok=True)

# Load model
path_to_model = glob.glob('model_*/checkpoint*.pt')[0]
t_skip = int(path_to_model.split('_')[-1][:-3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(path_to_model=path_to_model, device=device)
model.eval()

# Compile model
compiled_model = torch.compile(model)

# params = {k: v.detach() for k, v in model.named_parameters()}

# Training loss function
loss_fn = MSELoss(reduction='mean')

# Load test dataset
test_dataset = H5Dataset(path='data', mode='test')

# number of initial time steps to burn (CH dynamics too fast), hessian
n_burn = 50

for sim_id in range(test_dataset.n_groups):
    sim_time, sim_field = test_dataset.get_simulation(sim_id)
    x_grid, y_grid = test_dataset.get_meshgrid(sim_id)
    dt = sim_time[1] - sim_time[0]

    # burn the first n_burn indices
    t_start = n_burn * dt
    u_start = sim_field[n_burn]
    t_final = sim_time[-1]

    # run the surrogate at the specified settings
    surr_time, surr_field = run_surrogate(
        compiled_model,
        u_start,
        t_start,
        t_final,
        dt,
        t_skip,
    )

    # plot / animate the results
    save_path = f'{results_path}/result_{sim_id}.gif'
    create_anim(
        surr_field,
        surr_time,
        sim_field,
        sim_time,
        x_grid,
        y_grid,
        save_path,
    )

# For later, rather than looping over models with different parameter settings,
# we can also vectorize using torch.func.stack_module_state and vmap.
# https://pytorch.org/tutorials/intermediate/ensembling.html
