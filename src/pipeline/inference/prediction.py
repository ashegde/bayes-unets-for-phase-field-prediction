"""
This module contains functionality for
"""

import os
import pickle
from typing import Tuple, List, Callable, Dict

import h5py
import numpy as np
import torch
from torch import nn


# For later, rather than looping over models with different parameters,
# we can also vectorize using torch.func.stack_module_state and vmap.
# https://pytorch.org/tutorials/intermediate/ensembling.html

def surrogate_run(
    model: nn.Module,
    u_start: torch.Tensor,
    t_start: float,
    t_final: float,
    dt: float,
    t_skip: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts simulation run using the surrogate.
    """
    model.eval()
    device = next(model.parameters()).device

    # assuming u_start is of dimension (B,C,H,W)
    # where B = n_runs, and C = 1 for this problem.
    # n_runs = u_start.size(0)
    fields = [u_start.cpu()]
    times = [t_start]

    while times[-1] < t_final:
        xb = fields[-1].to(device)
        prediction = model(xb)
        fields.append(prediction.cpu())
        times.append(times[-1]+dt*t_skip)

    fields = torch.stack(fields, dim=1) #(B, T, C, H, W)
    times = torch.tensor(times) #(T,)
    return times, fields