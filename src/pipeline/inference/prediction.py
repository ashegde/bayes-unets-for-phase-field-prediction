"""
This module contains functionality for model prediction.

Specifically, the function `surrogate_run` autoregressively
predicts the field evolution across a specified time horizon.
"""
from typing import Tuple, Callable

import torch
from torch import nn


# For later, rather than looping over models with different parameters,
# we can also vectorize using torch.func.stack_module_state and vmap.
# https://pytorch.org/tutorials/intermediate/ensembling.html

def surrogate_run(
    model: nn.Module | Callable,
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

    # assuming u_start is of dimension (C,H,W)
    fields = [u_start.to(device)]
    times = [t_start]

    while times[-1] < t_final:
        xb = fields[-1].to(device)
        prediction = model(xb[None,...])
        fields.append(prediction[0])
        times.append(times[-1]+dt*t_skip)

    fields = torch.stack(fields, dim=0).cpu() #(T, C, H, W)
    times = torch.tensor(times) #(T,)
    return times, fields