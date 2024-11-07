import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
import h5py


class H5Dataset(Dataset):
    """
    A dataset class for loading Cahn-Hilliard simulation data stored in HDF5 format.

    This dataset provides functionality for loading simulation data fields and corresponding
    time values. It is used for training, validation, or testing machine learning models
    on simulation data, specifically for problems like the Cahn-Hilliard equation.

    Attributes
    ----------
    path : str
        Path to the directory containing the HDF5 data files.
    skip : int
        The number of time steps to skip when retrieving data.
    mode : str
        Specifies which subset of data to load: 'train', 'valid', or 'test'.
    dtype : torch.dtype
        The data type for the tensors, typically set to torch.float32 for efficiency.
    h5f : h5py.File
        A handle for the opened HDF5 file.
    num_runs : int
        The total number of simulation runs (groups) in the dataset.
    group_names : list of str
        List of names for each group (simulation run) in the HDF5 file.
    group_boundaries : np.ndarray
        Cumulative sum of the number of time steps per group (simulation run).
    num_groups : int
        The number of simulation runs (groups) in the dataset.
    """

    def __init__(self, path: str, mode: str, skip: int = 1):
        """
        Initialize the dataset object for loading simulation data from an HDF5 file.

        Parameters
        ----------
        path : str
            Path to the directory containing the HDF5 data files.
        mode : str
            The mode specifying which dataset to load. Should be one of 'train', 'valid', or 'test'.
        skip : int, optional
            The number of time steps to skip when retrieving data. Default is 1.

        Raises
        ------
        ValueError
            If the provided mode is not one of 'train', 'valid', or 'test'.
        """
        super().__init__()

        # Validate the mode input
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("mode must be one of 'train', 'valid', or 'test'")

        self.path = path
        self.skip = skip
        self.mode = mode
        self.dtype = torch.float32  # Use float32 for efficiency in memory and computation

        # Open the HDF5 file corresponding to the chosen mode
        self.h5f = h5py.File(f'{self.path}/{self.mode}_data.h5', 'r')

        # Retrieve the names of the groups (simulation runs) in the HDF5 file
        self.group_names = list(self.h5f.keys())

        # Compute the cumulative sum of the number of time steps per group, considering the skip factor
        self.group_boundaries = np.cumsum(
            [0] + [
                len(self.h5f[group_name]['time'][:]) - self.skip for group_name in self.group_names
            ]
        )

        # The number of simulation runs (groups) in the dataset
        self.num_groups = len(self.group_names)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        The length of the dataset is the cumulative sum of the number of valid time steps across all groups.

        Returns
        -------
        int
            The total number of samples in the dataset.
        """
        return int(self.group_boundaries[-1])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        Given an index, this method returns a tuple of tensors containing the field data 
        at a particular time step and the field data at the subsequent time step (with a skip of `skip`).

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two tensors:
            - field_data: Tensor at the current time step.
            - next_field_data: Tensor at the subsequent time step, after skipping `skip` steps.

        Raises
        ------
        IndexError
            If the index is out of range for the dataset.
        """
        # Identify the group (simulation run) and the index within that group
        group_id = np.digitize(index, self.group_boundaries, right=False) - 1
        index_within_group = index - self.group_boundaries[group_id]

        # Load the field data for the current time step and the subsequent time step (after skipping `skip` steps)
        field_data = torch.from_numpy(
            self.h5f[self.group_names[group_id]]['field_values'][index_within_group]
        ).to(self.dtype)

        next_field_data = torch.from_numpy(
            self.h5f[self.group_names[group_id]]['field_values'][index_within_group + self.skip]
        ).to(self.dtype)

        return field_data[:, None, :, :], next_field_data[:, None, :, :] # (B, C=1, H, W)

    def close(self):
        """
        Close the HDF5 file to free up resources.

        This method should be called after the dataset is no longer needed to ensure that the HDF5 file
        is properly closed, releasing any held resources.
        """
        self.h5f.close()

    def get_meshgrid(self, group_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the X and Y grid coordinates of the field values.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two tensors:
            - X: Tensor of x-coordinates.
            - Y: Tensor of y-coordinates.

        Note
        ------
        For simplicity, we assume that the coordinate grids are identical across runs / groups
        """
        x_grid = torch.from_numpy(
            self.h5f[self.group_names[group_id]]['x_coordinates'][:],
        ).to(self.dtype)

        y_grid = torch.from_numpy(
            self.h5f[self.group_names[group_id]]['y_coordinates'][:],
        ).to(self.dtype)
        return x_grid, y_grid
