"""
This module contains helper functions for plotting and animating simulation and surrogate results.
"""
import copy
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plot settings
cmap1 = cm.get_cmap('bwr')
normalizer1 = Normalize(-1.2, 1.2)
cm_phase = cm.ScalarMappable(cmap=cmap1, norm=normalizer1)

cmap2 = cm.get_cmap('inferno')
normalizer2 = Normalize(0, 2)
cm_error = cm.ScalarMappable(cmap=cmap2, norm=normalizer2)


def align_surr_to_sim(
    surr_field: torch.Tensor,
    surr_time: torch.Tensor,
    sim_field: torch.Tensor,
    sim_time: torch.Tensor,
):
    """
    Aligns and pads the surrogate results to the simulation results,
    based on their corresponding times.
    """
    # surr_field is of dimension (T_surr, C, H, W)
    # sim_field is of dimension (T_sim, C, H, W)

    padded_surr_time = copy.deepcopy(sim_time)
    padded_surr_field = []

    sidx = 0 # surrogate time index
    for i, time in enumerate(sim_time.tolist()):
        if time < surr_time[0]:
            padded_surr_field.append(
                float('nan') * torch.ones_like(sim_field[i]),
            )
        elif time == surr_time[sidx]:
            padded_surr_field.append(surr_field[sidx])
            sidx += 1
        else:
            padded_surr_field.append(surr_field[sidx])
    padded_surr_field = torch.stack(padded_surr_field, dim=0)
    return padded_surr_time, padded_surr_field


def plot_states(
    fig: mpl.figure.Figure,
    axs: plt.Axes,
    surr_field: np.ndarray,
    sim_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    title: str,
):

    """
    Plots result for a particular time step
    """

    axs[0, 0].contourf(x_grid, y_grid, surr_field, cmap=cmap1, norm=normalizer1)
    axs[0, 0].set_title('UNet')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm_phase, cax=cax, orientation='vertical')


    axs[0, 1].contourf(x_grid, y_grid, sim_field, cmap=cmap1, norm=normalizer1)
    axs[0, 1].set_title('Simulation')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm_phase, cax=cax, orientation='vertical')

    mse = np.mean((sim_field-surr_field)**2)
    axs[1, 0].contourf(x_grid, y_grid, np.abs(sim_field-surr_field), cmap=cmap2, norm=normalizer2)
    axs[1, 0].set_title(f'Absolute error (mse = {mse:0.3f})')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm_error, cax=cax, orientation='vertical')
    fig.suptitle(title)


def create_anim(
    surr_field: torch.Tensor,
    surr_time: torch.Tensor,
    sim_field: torch.Tensor,
    sim_time: torch.Tensor,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    save_path: str,
):
    """
    Creates animation of simulation and surrogate results.
    """

    # surr_field is of dimension (T_surr, C, H, W)
    # sim_field is of dimension (T_sim, C, H, W)

    # assuming a uniform time skip
    t_skip = surr_time[1]-surr_time[0]


    _, padded_surr_field = align_surr_to_sim(
        surr_field,
        surr_time,
        sim_field,
        sim_time,
    )

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    fig.delaxes(axs[1, 1])

    def animate(j):
        current_time = sim_time[j]
        current_field_sim = sim_field[j].squeeze().numpy().cpu()
        current_field_surr = padded_surr_field[j].squeeze().numpy().cpu()
        title = f'Time = {current_time}, UNet skips {t_skip}s.'
        plot_states(
            fig,
            axs,
            current_field_surr,
            current_field_sim,
            x_grid,
            y_grid,
            title,
        )

        # Render video
        anim = animation.FuncAnimation(
            fig, animate, frames=range(sim_time.size(0)), interval=20,
        )
        # writervideo = animation.FFMpegWriter(fps=args.animfps)
        # anim.save(args.animname, writer=writervideo)
        anim.save(save_path)
        plt.close()