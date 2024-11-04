import os
import pickle
import h5py
import numpy as np
from simulator import CahnHilliardSimulator

# Create a directory for storing data if it does not exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Set a random seed for reproducibility
seed_val = 2023
np.random.seed(seed_val)

# Dataset parameters
dt = 0.01                # Time step for the simulation
num_steps = 500          # Number of simulation steps to run
num_train = 2            # Number of training datasets to generate
num_valid = 1            # Number of validation datasets to generate
num_test = 1             # Number of test datasets to generate

# Define the number of experiments for each mode
experiments = {'train': num_train, 'valid': num_valid, 'test': num_test}

# Initialize and save the Cahn-Hilliard simulator
simulator = CahnHilliardSimulator(dt=dt)
with open(f'{data_dir}/simulator.pkl', 'wb') as f:
    pickle.dump(simulator, f)

# Generate datasets for each mode (train, valid, test)
for mode in experiments.keys():
    # Define file name for the current mode's dataset
    file_name = os.path.join(data_dir, f'{mode}_data.h5')
    
    # Create an HDF5 file to store the simulation data
    with h5py.File(file_name, 'w') as h5f:
        for ii in range(experiments[mode]):
            # Initialize concentration field and time lists
            u = [0.0 + 0.1 * (np.random.rand(simulator.x_res, simulator.y_res) - 0.5)]
            t = [0.0]
            
            # Initialize the simulator with the initial concentration field
            simulator.initialize(u=u[0])
            
            # Run the simulation for the specified number of steps
            for step in range(num_steps):
                u.append(simulator.step())  # Update concentration field
                t.append(simulator.t)       # Record the current time
            
            # Stack the concentration fields into a 3D array
            u = np.stack(u, axis=0)
            t = np.array(t)  # Convert time list to numpy array
            
            # Create a group in the HDF5 file for this run
            run_group = h5f.create_group(f'run_{ii}')
            run_group.create_dataset('field', data=u)  # Store the concentration field
            run_group.create_dataset('time', data=t)    # Store the time values
            run_group.create_dataset('length', data=len(t))
