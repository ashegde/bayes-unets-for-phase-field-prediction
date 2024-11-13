"""
This module contains functionality for
"""

import os
import pickle
from typing import Tuple, Callable, Dict

import h5py
import numpy as np
import torch

def rollout(
    model_fn: Callable,
    params: Dict,
    u_start: torch.Tensor,
    t_skip: int,
    dt: float,
    t_start: float,
    t_final: float,
) -> Tuple(torch.Tensor, torch.Tensor):
    
    def model_eval(u):
        return model_fn(params, u)

    fields = [u_start]
    times = [t_start]

    while times[-1] < t_final:
        fields.append(model_eval(fields[-1]))
        times.append(times[-1]+dt*t_skip)        

    return times, fields