"""
This module contains functonality for posterior sampling using the
loss projected posterior described in:

(1) Miani, Marco, Hrittik Roy, and Søren Hauberg. "Bayes without Underfitting: 
Fully Correlated Deep Learning Posteriors via Alternating Projections." 
arXiv preprint arXiv:2410.16901 (2024).

More general details of these methods can be found in:

(2) Roy, Hrittik, Marco Miani, Carl Henrik Ek, Philipp Hennig,
Marvin Pförtner, Lukas Tatzel, and Søren Hauberg.
"Reparameterization invariance in approximate Bayesian inference."
arXiv preprint arXiv:2406.03334 (2024).

"""
import copy
import os
import glob
import pickle
import random
from typing import Tuple, Callable, Dict, List


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.func as tf
import tqdm

def randn_params(
    param_template: Dict,
    precision: torch.tensor,
    n_samples: int = 1,
) -> Tuple[Dict,...]:
    """
    Samples model parameters from a normal distribution with mean 0 and given precision. 

    Parameters:
    -----------
    params_template : dict
        model parameters, used as a template for the samples

    precision : torch.tensor
        single input, typically of dimension (C, H, W)

    n_samples : int
        number of samples requested

    Returns:
    --------
    tuple
        tuple of sampled model parameters

    """

    return ({
        k: 1/torch.sqrt(precision) * torch.randn_like(v) for k, v in param_template.items()
    } for _ in range(n_samples))

def linearized_predict(
    func: Callable,
    param0: Dict,
    param1: Dict,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates a linearized model, as recommended in (2).

    f_{lin}(x; param1) = f(x; param0) + J(x; param0) (param1-param0)

    Parameters:
    -----------

    func : callable
        single evaluation function

    params0 : dict
        nominal model parameters

    params1 : dict
        model parameters to-be-evaluated

    x : torch.tensor
        single input, typically of dimension (C, H, W)

    Returns:
    --------
    torch.tensor
        model output, typically of dimension (C, H, W)

    """
    def fp(params):
        return func(params, x)

    param_diff = {k: param1[k]-v for k, v in param0.items()}
    outputs, jvp_val = tf.jvp(fp, (param0,), (param_diff,))
    return outputs+jvp_val


def batched_jjt(func: Callable, params: Dict, xb: torch.Tensor, yb: torch.Tensor):
    """
    Given func: R^d --> R^O and a batch of B data points, computes BO-by-BO the matrix JJt,
    where J is the jacobian of func (with respect to params) evaluated at points in (xb, yb). 

    In our case, the typical usage is that func is a loss function, and therefore takes as input
    batches (xb, yb) and produces a single output (O = 1).

    Parameters:
    -----------

    func : callable
        single evaluation function -- typically, a loss function (in which O = 1)

    params : dict
        loss / model parameters

    xb : torch.tensor 
        batch of inputs -- typically of dimension (B, C, H, W)

    yb : torch.tensor 
        batch of corresponding targets -- typically of dimension (B, C, H, W)

    Returns:
    --------
    torch.tensor
        JJ.T -- of dimension BO-by-BO
    """
    # Compute J(xb,yb)
    jac = tf.vmap(tf.jacrev(func), (None, 0, 0))(params, xb, yb)
    # jac is a dict of torch tensors (keys correspond to params)
    # the 0-th dimension of each tensor is the batch dimension (vmapped dim).
    jac = jac.values()

    # flattens the tensor in each batch dimension
    jac = [j.flatten(1) for j in jac]

    # Compute J@J.T where J is (N,P) and J.T is (P, M=N) 
    # contraction across parameter dimension of the Jacobian
    einsum_expr = 'NP,MP->NM'

    # for each block of parameter derivatives in the jacobian, contract across the parameter dimension
    result = torch.stack([torch.einsum(einsum_expr, j, j) for j in jac]) 
    # sum across all parameter blocks to complete the contraction
    result = result.sum(0) 
    return result

def precompute_inv_jjt(
    func: Callable,
    params: Dict,
    train_loader: DataLoader,
    save_path: str = None,
) -> List:
    """
    Precomputes the core inverse matrices for all batches in the DataLoader.

    Recall, for a batch Jacobian Jb, the projection is:
            Proj(new_params) = (I - Jb.T @ inv(Jb@Jb.T) @ Jb) @ new_params
    In this function, we precompute all inv(Jb@Jb.T).

    Parameters:
    -----------

    func : callable
        single evaluation function -- typically, a loss function

    params : dict
        nominal model parameters

    train_loader : torch.utils.data.dataloader
        DataLoader containing the training data

    Returns:
    --------
    list
        list containing inv(Jb@Jb.T) for each batch b.
    """

    inv_jjt_cache = []

    for b, data in enumerate(tqdm.tqdm(train_loader)):
        xb, yb = data
        inv_jjt_cache.append(
            torch.linalg.pinv(batched_jjt(func, params, xb, yb))
        )
    
    # save
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(inv_jjt_cache, f)

    return inv_jjt_cache


def batched_proj(
    func: Callable,
    base_params: Dict,
    new_params: Dict,
    xb: torch.Tensor,
    yb: torch.Tensor,
    inv_jjt: torch.Tensor,
) -> Dict:
    """
    Project model_parameters `new_params` onto the null space of J,
    where J is the jacobian of `func` with respect to parameter `params` and batch data
    `xb` and `yb`.

    Proj(new_params) = (I - J.T @ inv(JJt) @ J) @ new_params

    In our case, the typical usage is that func is a loss function, and therefore takes as input
    batches (xb, yb) and produces a single output.

    Parameters:
    -----------

    func : callable
        single evaluation function -- typically, a loss function (in which O = 1)

    base_params : dict
        nominal model parameters

    new_params : dict
        model parameters to-be-projected

    xb : torch.tensor
        batch of inputs -- typically of dimension (B, C, H, W)

    yb : torch.tensor
        batch of corresponding targets -- typically of dimension (B, C, H, W)

    Returns:
    --------
    Dict
        projection of new_params into the Jacobian null space
    """
    def fp(p):
        return tf.vmap(func, (None, 0, 0))(p, xb, yb)
    # let v = new_params. First, Jv:
    _, jvp = tf.jvp(fp, (base_params,), (new_params,))
    # next, inv_jjt
    inv_jjt_jv = torch.matmul(inv_jjt, jvp)
    # finally,
    _, vjp_fn = tf.vjp(fp, (base_params,))
    vjps = vjp_fn(inv_jjt_jv)[0]
    return vjps[0]

def apply_proj_cycle(
    func: Callable,
    base_params: Dict,
    params: Dict,
    dataloader: DataLoader,
    inv_jjt_cache: List,
) -> Dict:
    proj_params = copy.deepcopy(params)

    for b, data in enumerate(tqdm.tqdm(dataloader, desc='batches', leave=False)):
        xb, yb = data
        proj_params = batched_proj(
            func,
            base_params,
            proj_params,
            xb,
            yb,
            inv_jjt_cache[b],
        )
    return proj_params

def alternating_projection_sampler(
    n_samples: int,
    n_cycle: int,
    func: Callable,
    base_params: Dict,
    prior_precision: torch.Tensor,
    dataloader: DataLoader,
    inv_jjt_cache_path: str = None,
) -> List:
    """
    pass
    """
    # load cache if it exists, else precompute and save
    if inv_jjt_cache_path:
        with open(inv_jjt_cache_path, 'rb') as f:
            inv_jjt_cache = pickle.load(f)
    else:
        inv_jjt_cache = precompute_inv_jjt(func, base_params, dataloader, inv_jjt_cache_path)

    # generate samples from prior
    param_samples = randn_params(base_params, prior_precision, n_samples)

    # perform alternating projections for n_cycles
    proj_samples = []
    for i, p_samp in enumerate(tqdm.tqdm(param_samples, desc ='samples')):
        p = copy.deepcopy(p_samp)
        for _ in tqdm.tqdm(range(n_cycle), desc='cycles', leave=False):
            p = apply_proj_cycle(func, base_params, p, dataloader, inv_jjt_cache)
        proj_samples.append(p)
    return proj_samples

