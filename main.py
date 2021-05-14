import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import datasets.mnist as mnist
import datasets.cifar10 as cifar10
import methods.mcdropout as mcd
import methods.ensemble as ens
import methods.models
import constants
import metrics

from methods.SingleNetwork import SingleNetwork
from methods.mcdropout.MCDropout import MCDropout
from methods.ensemble.Ensemble import Ensemble, NCEnsemble

def get_trainer(args, device):
    if args.method == 'single':
        trainer = SingleNetwork(args, device)

    if args.method == 'mcdrop':
        trainer = MCDropout(args, device)

    if args.method == 'ensemble':
        trainer = Ensemble(args, device)

    if args.method == 'ncensemble':
        trainer = NCEnsemble(args, device)

    return trainer

def test_mnist(trainer, args, metric_dict):
    if args.corrupted_test:
        if args.corruption=='rotation':
            for i in np.arange(0, 181, 15):
                print(f'\nShift: {i}\n')

                test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensity=i, corruption='rotation')

                acc, metric_res = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

                wandb.log({'Test/rotated accuracy': acc, 'shift': i})
                for name, val in metric_res.items():
                    wandb.log({f'Test/rotated {name}': val, 'shift': i})
        else:
            for i in np.arange(0, 29, 2):
                print(f'\nShift: {i}\n')

                test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensity=i, corruption='shift')

                acc, metric_res = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

                wandb.log({'Test/translated accuracy': acc, 'shift': i})
                for name, val in metric_res.items():
                    wandb.log({f'Test/translated {name}': val, 'shift': i})
    else:
        test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=False)
        acc, metric_res = trainer.test(test_loader=test_loader, metric_dict=metric_dict)
        print(f'Testing\nAccuracy: {acc}')

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

# data config
parser.add_argument('--data-dir', type=str, default='/scratch/gp491/data',
                    help='Directory the relevant datasets can be found in')
parser.add_argument('--dataset-type', type=str, default='mnist',
                    choices=['cifar10', 'mnist'], help='Dataset name')
parser.add_argument('--corrupted-test', action='store_true',
                    help='Whether to use a corrupted (shifted) testing set. If omitted, will use the standar one.')
parser.add_argument('--corruption', type=str, default='rotation',
                    choices=['rotation', 'shift'],
                    help='What kind of corruption to usefor MNIST testing.')
parser.add_argument('--validation-fraction', type=float, default=0.1,
                    help='Fraction of the training set to be held out for validation, in cases where a dedicated set is unavailable.')

# method config
parser.add_argument('--method', type=str, default='single',
                    choices=['single', 'ensemble', 'mcdrop', 'ncensemble'],
                    help='method to run')
parser.add_argument('--n', type=int, default=5,
                    help='Size of the ensemble to be trained.')
parser.add_argument('--model', type=str, default='lenet',
                    choices=['lenet', 'mlp', 'resnet'])
parser.add_argument('--reg-weight', type=float, default=0.5,
                    help='Scaling factor for custom loss regularisation, initial value.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate for models that use it')

# training config
parser.add_argument('--scheduled-lr', action='store_true',
                    help='if set, will use a learning rate scheduler, predefined for network type')
parser.add_argument('--batch-size', type=int, default=250,
                    help='Batch size to use in training')
parser.add_argument('--epochs', type=int, default=15,
                    help='Maximum number of epochs to train for')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Initial learning rate')
parser.add_argument('--cpu', action='store_true',
                    help='Whether to train on the CPU. If ommited, will train on a GPU')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of CPU cores to load data on')

args = parser.parse_args()

if __name__ == '__main__':

    wandb.init(project=f'mphil-{args.dataset_type}', entity='gintepe', dir=constants.LOGGING_DIR)
    wandb.config.update(args)

    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'

    train_loader, val_loader = get_train_and_val_loaders(
                                dataset_type=args.dataset_type,
                                data_dir=args.data_dir,
                                batch_size=args.batch_size,
                                val_fraction=args.validation_fraction,
                                num_workers=args.num_workers,
                                )

    trainer = get_trainer(args, device)
    trainer.train(train_loader, val_loader, epochs=args.epochs)
    
    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                    'ECE': metrics.wrap_ece(bins=20), 
                    'Brier': metrics.wrap_brier()}

    if args.dataset_type == 'mnist':
        test_mnist(trainer, args, metric_dict)



