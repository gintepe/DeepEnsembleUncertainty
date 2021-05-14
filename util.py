import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import datasets.mnist as mnist
import datasets.cifar10 as cifar10
import constants
import metrics
from configuration import Configuration

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

def test_mnist(trainer, args, metric_dict, wandb_log=True):
    if args.corrupted_test:
        for i in np.arange(0, 181, 15):
            print(f'\nShift: {i}\n')
            test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensity=i, corruption='rotation')
            acc, metric_res, _, _, _ = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

            if wandb_log:
                wandb.log({'Test/rotated accuracy': acc, 'shift': i})
                for name, val in metric_res.items():
                    wandb.log({f'Test/rotated {name}': val, 'shift': i})

        for i in np.arange(0, 29, 2):
            print(f'\nShift: {i}\n')
            test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensity=i, corruption='shift')
            acc, metric_res, _, _, _ = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

            if wandb_log:
                wandb.log({'Test/translated accuracy': acc, 'shift': i})
                for name, val in metric_res.items():
                    wandb.log({f'Test/translated {name}': val, 'shift': i})
    else:
        test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=False)
        acc, metric_res, _, _, _ = trainer.test(test_loader=test_loader, metric_dict=metric_dict)
        print(f'Testing\nAccuracy: {acc}')


def test_cifar(trainer, args, metric_dict, wandb_log=True):
    test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=False)
    acc, metric_res, _, _, _ = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

    if wandb_log and args.corrupted_test:
        wandb.log({'Test/corrupted accuracy': acc, 'intensity': 0})
        for name, val in metric_res.items():
            wandb.log({f'Test/corrupted {name}': val, 'intensity': 0})

    print(f'Testing\nAccuracy: {acc}')

    if args.corrupted_test:
        for i in range(1, 6):
            test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensities=[i])
            acc, metric_res, _, _, _ = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

            if wandb_log:
                wandb.log({'Test/corrupted accuracy': acc, 'intensity': i})
                for name, val in metric_res.items():
                    wandb.log({f'Test/corrupted {name}': val, 'intensity': i})

