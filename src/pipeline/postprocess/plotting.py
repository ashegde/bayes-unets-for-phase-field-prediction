import copy
import numpy as np
import torch
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    sidx = 0  # surrogate time index
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


def create_anim(
    surr_field: torch.Tensor,
    surr_time: torch.Tensor,
    sim_field: torch.Tensor,
    sim_time: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    save_path: str,
):
    """
    Creates animation of simulation and surrogate results.
    """

    # surr_field is of dimension (T_surr, C, H, W)
    # sim_field is of dimension (T_sim, C, H, W)
    x_grid_np = x_grid.numpy()
    y_grid_np = y_grid.numpy()

    # assuming a uniform time skip
    t_skip = surr_time[1] - surr_time[0]

    _, padded_surr_field = align_surr_to_sim(
        surr_field,
        surr_time,
        sim_field,
        sim_time,
    )

    time_np = sim_time.numpy()
    sim_field_np = sim_field.squeeze().detach().numpy()
    surr_field_np = padded_surr_field.squeeze().detach().numpy()
    error_field_np = np.abs(sim_field_np - surr_field_np)

    # Setup and animate plots
    cmap1 = mpl.colormaps['bwr']
    normalizer1 = Normalize(-1.2, 1.2)
    cm_phase = cm.ScalarMappable(cmap=cmap1, norm=normalizer1)

    cmap2 = mpl.colormaps['inferno']
    normalizer2 = Normalize(0, 2)
    cm_error = cm.ScalarMappable(cmap=cmap2, norm=normalizer2)

    plot_axs = [None, None, None]  # A list to hold contours for each subplot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    fig.delaxes(axs[1, 1])  # Remove unused subplot (1, 1)

    # Plot for the first subplot (UNet)
    axs[0, 0].set_title('UNet (time-coarsened)')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm_phase, cax=cax, orientation='vertical')
    plot_axs[0] = axs[0, 0].contourf(
        x_grid_np,
        y_grid_np,
        surr_field_np[0],
        cmap=cmap1,
        norm=normalizer1,
    )

    # Plot for the second subplot (Simulation)
    axs[0, 1].set_title('Simulation')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm_phase, cax=cax, orientation='vertical')
    plot_axs[1] = axs[0, 1].contourf(
        x_grid_np,
        y_grid_np,
        sim_field_np[0],
        cmap=cmap1,
        norm=normalizer1,
    )

    # Plot for the third subplot (Absolute error)
    axs[1, 0].set_title('Absolute error')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cm_error, cax=cax, orientation='vertical')
    plot_axs[2] = axs[1, 0].contourf(
        x_grid_np,
        y_grid_np,
        error_field_np[0],
        cmap=cmap2,
        norm=normalizer2,
    )

    # Function to animate the plots
    def animate(frame):
        # Remove old contour collections for all subplots
        for p in plot_axs:
            for tp in p.collections:
                tp.remove()

        # Update each subplot with new contours
        plot_axs[0] = axs[0, 0].contourf(
            x_grid_np,
            y_grid_np,
            surr_field_np[frame],
            cmap=cmap1,
            norm=normalizer1,
        )
        plot_axs[1] = axs[0, 1].contourf(
            x_grid_np,
            y_grid_np,
            sim_field_np[frame],
            cmap=cmap1,
            norm=normalizer1,
        )
        plot_axs[2] = axs[1, 0].contourf(
            x_grid_np,
            y_grid_np,
            error_field_np[frame],
            cmap=cmap2,
            norm=normalizer2,
        )

        # Update title with the current time
        title = f'Time = {time_np[frame]:0.2f}, UNet skips {t_skip}s.'
        fig.suptitle(title)

        # Return all collections for blitting
        return sum([p.collections for p in plot_axs], [])

    # Create the animation using FuncAnimation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=range(time_np.size(0)),
        interval=10,
        blit=True,
        repeat=False,
    )

    # Save the animation to the specified path
    anim.save(
        filename=save_path,
        fps=24,
        extra_args=['-vcodec', 'libx264'],
        dpi=300,
    )
    plt.close()
