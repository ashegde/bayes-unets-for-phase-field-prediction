"""
Dataset Preparation

This module generates training, validation, and test datasets.
These datasets are constructed using the imported Cahn-Hilliard simulator,
and are saved in the HDF5 format.
"""

import os
import pickle
import h5py
import numpy as np

from simulator.simulator import CahnHilliardSimulator

# Create a directory for storing data if it does not exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Set a random seed for reproducibility
seed_val = 2023
np.random.seed(seed_val)

# Dataset parameters
dt = 0.01                # Time step for the simulation
n_steps = 500          # Number of simulation steps to run
n_train = 10           # Number of training datasets to generate
n_valid = 5            # Number of validation datasets to generate
n_test = 5             # Number of test datasets to generate

u_init = 0.0
u_noise_scale = 0.5

# Define the number of experiments for each mode
experiments = {'train': n_train, 'valid': n_valid, 'test': n_test}

# Initialize and save the Cahn-Hilliard simulator
simulator = CahnHilliardSimulator(dt=dt)
with open(f'{data_dir}/simulator.pkl', 'wb') as f:
    pickle.dump(simulator, f)

# Generate datasets for each mode (train, valid, test)
for mode, n_exp in experiments.items():
    # Define file name for the current mode's dataset
    file_name = os.path.join(data_dir, f'{mode}_data.h5')

    # Create an HDF5 file to store the simulation data
    with h5py.File(file_name, 'w') as h5f:
        for ii in range(n_exp):
            # Initialize concentration field and time lists
            noise = 2.0*np.random.rand(simulator.x_res, simulator.y_res)-1.0
            u = [u_init + u_noise_scale*noise]
            t = [0.0]

            # Initialize the simulator with the initial concentration field
            simulator.initialize(u=u[0])

            # Run the simulation for the specified number of steps
            for step in range(n_steps):
                u.append(simulator.step())  # Update concentration field
                t.append(simulator.t)       # Record the current time

            # Stack the concentration fields into a 3D array
            u = np.stack(u, axis=0)
            t = np.array(t)  # Convert time list to numpy array

            # Create a group in the HDF5 file for this run
            run_group = h5f.create_group(f'run_{ii}')
            run_group.create_dataset('x_coordinates', data=simulator.X)
            run_group.create_dataset('y_coordinates', data=simulator.Y)
            run_group.create_dataset('field_values', data=u)
            run_group.create_dataset('time', data=t)
            run_group.create_dataset('length', data=len(t))
