import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
import h5py


class H5Dataset(Dataset):
    """
    Base HDF5 Cahn-Hilliard Dataset.

    This dataset class loads Cahn-Hilliard simulation data stored in HDF5 format.
    It provides functionality to access simulation fields and time values for training,
    validation, or testing purposes.

    Attributes:
    ----------
    path : str
        Path to the directory containing the HDF5 data files.
    skip : int
        Number of time steps to skip when retrieving data.
    mode : str
        Specifies which dataset to load: 'train', 'valid', or 'test'.
    dtype : torch.dtype
        Data type for tensors, set to torch.float32 for efficiency.
    h5f : h5py.File
        Handle for the opened HDF5 file.
    num_runs : int
        Total number of simulation runs (groups) in the dataset.
    group_names : list
        List of names for each group in the HDF5 file.
    """

    def __init__(self, path: str, mode: str, skip: int = 1):
        """
        Initialize a torch dataset object.

        Parameters:
        ----------
        path : str
            Path to the directory containing the HDF5 data files.
        mode : str
            Mode specifying which data to load: 'train', 'valid', or 'test'.
        skip : int, optional
            Number of time steps to skip when retrieving data. Default is 1.
        """
        super().__init__()

        self.path = path
        self.skip = skip
        self.mode = mode
        self.dtype = torch.float32  # Using float32 for efficiency
        
        # Open the HDF5 file for reading
        self.h5f = h5py.File(f'{self.path}/{self.mode}_data.h5', 'r')
        
        # Retrieve the number of runs (groups) in the dataset
        self.group_names = list(self.h5f.keys())
        self.num_runs = len(self.group_names)

    def __len__(self) -> int:
        """
        Get the total number of experiments (groups) in the dataset.

        Returns:
        -------
        int
            The number of runs (groups) in the dataset.
        """
        return self.num_runs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the trajectory and corresponding coordinates for a specific experiment.

        Parameters:
        ----------
        index : int
            Index of the experiment/trajectory to retrieve.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - The trajectory of the experiment (shape: (nt, ndim)).
            - The spatial coordinates for each snapshot (shape: (2, ndim)),
              with [0, :] representing x-coordinates and [1, :] representing y-coordinates.
        """
        if index < 0 or index >= self.num_runs:
            raise IndexError("Index out of bounds for the dataset.")

        group_name = self.group_names[index]

        # Load time values from the specified group
        times = self.h5f[group_name]['time'][:]
        
        if len(times) <= self.skip:
            raise ValueError(f"Not enough time steps to skip {self.skip} in group {group_name}.")

        # Randomly select a valid index to retrieve the field data
        iterate = np.random.randint(0, len(times) - self.skip)

        # Load the field data and return it as a tuple of tensors
        field_data = torch.from_numpy(self.h5f[group_name]['field'][iterate]).to(self.dtype)
        next_field_data = torch.from_numpy(self.h5f[group_name]['field'][iterate + self.skip]).to(self.dtype)

        return field_data, next_field_data

    def close(self):
        """Close the HDF5 file to free resources."""
        self.h5f.close()
