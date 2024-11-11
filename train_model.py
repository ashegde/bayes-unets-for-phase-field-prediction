"""
Training Script

This module provides functionality for training a UNet surrogate model
for the Cahn-Hilliard equation.
"""

import argparse
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from loaders import H5Dataset
from model import UNet2d


def main(args):
    """
    Main function for model training.
    """
    # setup directories
    date_time = datetime.now()
    timestring = (
        f'{date_time.date().month}{date_time.date().day}'
        f'{date_time.time().hour}{date_time.time().minute}'
    )
    data_path = 'data'
    model_path = f'model_{timestring}'
    log_path = f'{model_path}/log'
    save_path = (
        f'{model_path}/model_savetime_{timestring}'
        f'_batchsize_{args.batch_size}_timeskip_{args.time_skip}.pt'
    )

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setting Seeds
    seed_val = 2023
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)

    # Load datasets

    train_dataset = H5Dataset(
        path=data_path,
        mode='train',
        skip=args.time_skip,
    )
    valid_dataset = H5Dataset(
        path=data_path,
        mode='valid',
        skip=args.time_skip,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f'Device: {device}')
    print(f'Save to: {save_path}')

    # Model settings
    in_channels = 1
    out_channels = in_channels
    init_features = 16

    model = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
    )

    model.to(device)

    # Report number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Model size: {n_params} trainable parameters')

    # Optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=args.lr_decay,
        patience=20,
        min_lr=1e-5,
    )

    # Logging
    log = 'train_log.txt'
    date_time = datetime.now()
    current_date = (
        f'mm/dd/yyyy - {date_time.date().month}/'
        f'{date_time.date().day}/'
        f'{date_time.date().year}'
    )
    current_time = (
        f'hh:mm:ss - {date_time.time().hour}:'
        f'{date_time.time().minute}:'
        f'{date_time.time().second}'
    )
    with open(log, 'w', encoding='utf-8') as f:
        f.write(f'Initialized on {current_date} {current_time}')

    # Training loop
    min_val_loss = 10e30
    n_batches_per_epoch = len(train_loader)

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch}")

        # training step
        model.train()
        step = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            # Log training results
            train_loss = loss.item()
            with open(log, 'a', encoding='utf-8'):
                f.write(
                    (
                        f'Epoch {epoch} ||'
                        f' train step {step} / {n_batches_per_epoch} ||'
                        f' loss {train_loss}'
                    )
                )
            step += 1

        # validation step
        if epoch % args.valid_freq == 0:
            model.eval()
            valid_loss = []
            for xb, yb in valid_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                valid_loss.append(loss.item())

            # Log validation results
            val_loss = np.mean(valid_loss)
            with open(log, 'a', encoding='utf-8'):
                f.write(f'Validation || loss {val_loss}')

            # Save best model
            if val_loss < min_val_loss:
                torch.save(
                    model.state_dict(),
                    (
                        f'{model_path}/best_model_savetime_{timestring}'
                        f'_batchsize_{args.batch_size}_'
                        f'timeskip_{args.time_skip}.pt'
                    )
                )
                min_val_loss = val_loss
                print(f"Saving model at {save_path}\n")

            # Update scheduler
            scheduler.step(val_loss)

    # Saving final model
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Train an UNET-based PDE solver'
            'for the Cahn-Hilliard system'
        )
    )
    # training parameters
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Number of samples in each minibatch',
    )
    parser.add_argument(
        '--time_skip', type=int, default=25,
        help='Number of time steps to skip during prediction/inference',
    )
    parser.add_argument(
        '--n_epochs', type=int, default=30,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--lr_decay', type=float, default=0.9,
        help='Learning rate decay',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-6,
        help='weight decay',
    )
    # Misc
    parser.add_argument(
        '--valid_freq', type=int, default=1,
        help='number of epochs between validation steps',
    )
    args = parser.parse_args()
    main(args)
