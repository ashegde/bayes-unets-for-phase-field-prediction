import argparse
from datetime import datetime
import os
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from pipeline.dataset.loaders import H5Dataset
from pipeline.model.model import UNet2d
from pipeline.inference.


def setup_directories(timestring: str, args) -> tuple[str, str, str]:
    """
    Creates necessary directories for saving model and logs.

    Args:
        timestring (str): Unique timestamp used for naming files and directories.
        args (argparse.Namespace): Parsed arguments containing batch_size and time_skip.

    Returns:
        str: Path to the final model save location.
    """
    model_path = f'model_{timestring}'
    log_path = f'{model_path}/log'
    save_path = (
        f'{model_path}/model_savetime_{timestring}'
        f'_batchsize_{args.batch_size}_timeskip_{args.time_skip}.pt'
    )

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    return save_path, model_path, log_path


def configure_logging(path: str) -> None:
    """
    Sets up the logging configuration to log training and validation details.

    This function configures logging to both console and a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'{path}/train.log', mode='w', encoding='utf-8')
        ]
    )


def create_model(device: torch.device) -> torch.nn.Module:
    """
    Initializes the UNet model with predefined input and output channels.

    Args:
        device (torch.device): The device to move the model to ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The initialized UNet model.
    """
    in_channels = 1
    out_channels = in_channels
    init_features = 16

    model = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
    )

    model.to(device)

    return model


def calculate_parameters(model: torch.nn.Module) -> int:
    """
    Calculates the total number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The UNet model.

    Returns:
        int: Total number of trainable parameters.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])

    return n_params


def main(args: argparse.Namespace) -> None:
    """
    Main function for training the UNet model.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    # Setup directories and logging
    date_time = datetime.now()
    timestring = (
        f'{date_time.month}{date_time.day}{date_time.hour}{date_time.minute}'
    )
    save_path, model_path, log_path = setup_directories(timestring, args)
    configure_logging(log_path)

    # Set device (cuda if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set random seed for reproducibility
    seed_val = 2023
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)

    # Load datasets
    train_dataset = H5Dataset(path='data', mode='train', skip=args.time_skip)
    valid_dataset = H5Dataset(path='data', mode='valid', skip=args.time_skip)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = create_model(device)
    
    # Report the number of model parameters
    n_params = calculate_parameters(model)
    logging.info(f'Model size: {n_params} trainable parameters')

    # Define loss function and optimizer
    loss_fn = MSELoss(reduction='mean')



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Make predictions using an already trained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples in each minibatch')
    
    # Parse arguments and run main function
    args = parser.parse_args()
    main(args)