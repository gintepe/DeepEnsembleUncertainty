import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn


import mnist
import cifar10
import constants
from models import *
from train import train, train_simple_ensemble

def get_train_and_val_loaders(dataset_type, data_dir, batch_size, val_fraction, num_workers):
    """
    Function to load validation and training data for an appropriate dataset
    
    For now a dummy implementation 
    """
    if dataset_type == 'mnist':
        return mnist.get_mnist_train_valid_loader(data_dir, batch_size, random_seed=1, 
                                                    valid_size=val_fraction, num_workers=num_workers)
    if dataset_type == 'cifar10':
        return cifar10.get_cifar10_train_valid_loader(data_dir, batch_size, augment=False, 
                                                        random_seed=1, valid_size=val_fraction,
                                                        num_workers=num_workers)

parser = argparse.ArgumentParser(description='Train a configurable ensemble on a given dataset.')
parser.add_argument('--dataset_type', type=str, default='mnist',
                    choices=['cifar10', 'mnist'], help='Dataset name')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of CPU cores to load data on')
parser.add_argument('--method', type=str, default='single',
                    choices=['single', 'ensemble'],
                    help='method to run')
parser.add_argument('--batch-size', type=int, default=250,
                    help='Batch size to use in training')
parser.add_argument('--epochs', type=int, default=15,
                    help='Maximum number of epochs to train for')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Initial learning rate')
parser.add_argument('--cpu', action='store_true',
                    help='Whether to train on the CPU. If ommited, will train on a GPU')
parser.add_argument('--model', type=str, default='lenet',
                    choices=['lenet'])
parser.add_argument('--corrupted-test', action='store_true',
                    help='Whether to use a corrupted (shifted) testing set. If omitted, will use the standar one.')
parser.add_argument('--n', type=int, default=5,
                    help='Size of the ensemble to be trained.')
parser.add_argument('--validation-fraction', type=float, default=0.1,
                    help='Fraction of the training set to be held out for validation, in cases where a dedicated set is unavailable.')
parser.add_argument('--data-dir', type=str, default='/scratch/gp491/data',
                    help='Directory the relevant datasets can be found in')

args = parser.parse_args()

if __name__ == '__main__':

    wandb.init(project='mphil', entity='gintepe', dir=constants.LOGGING_DIR)
    wandb.config.update(args)
    

    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'

    train_loader, val_loader = get_train_and_val_loaders(
                                dataset_type=args.dataset_type,
                                data_dir=args.data_dir,
                                batch_size=args.batch_size,
                                val_fraction=args.validation_fraction,
                                num_workers=args.num_workers,
                                )
    
    if args.method == 'single':
        model = LeNet5().to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr,)
        criterion = nn.CrossEntropyLoss()

        train(model, train_loader, val_loader, criterion, optimizer, args.epochs, device=device)

    if args.method == 'ensemble':
        model = SimpleEnsemble(LeNet5, n=args.n).to(device)

        optimizers = [optim.Adam(m.parameters(), lr=args.lr,) for m in model.networks]
        # TODO fix the loss!
        criterion = nn.CrossEntropyLoss()
        train_simple_ensemble(model, train_loader, val_loader, criterion, optimizers, args.epochs, device=device)
