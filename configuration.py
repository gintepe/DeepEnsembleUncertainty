import argparse
import json
import os
import pprint
import torch
import pathlib


class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser(description='Train a configurable ensemble on a given dataset.')

        # data config
        parser.add_argument('--data-dir', type=str, default='/scratch/gp491/data',
                            help='Directory the relevant datasets can be found in')
        parser.add_argument('--dataset-type', type=str, default='mnist',
                            choices=['cifar10', 'mnist'], help='Dataset name')
        parser.add_argument('--corrupted-test', action='store_true',
                            help='Whether to use a corrupted (shifted) testing set. If omitted, will use the standar one.')
        parser.add_argument('--validation-fraction', type=float, default=0.1,
                            help='Fraction of the training set to be held out for validation, in cases where a dedicated set is unavailable.')

        # method config
        parser.add_argument('--method', type=str, default='single',
                            choices=['single', 'ensemble', 'mcdrop', 'ncensemble'], help='method to run')
        parser.add_argument('--n', type=int, default=5, help='Size of the ensemble to be trained.')
        parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'mlp', 'resnet'],
                            help='Model architecture to be used')
        parser.add_argument('--reg-weight', type=float, default=0.5,
                            help='Scaling factor for custom loss regularisation, initial value.')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for models that use it')

        # training config
        parser.add_argument('--scheduled-lr', action='store_true',
                            help='if set, will use a learning rate scheduler, predefined for network type')
        parser.add_argument('--batch-size', type=int, default=250, help='Batch size to use in training')
        parser.add_argument('--epochs', type=int, default=15, help='Maximum number of epochs to train for')
        parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate')
        parser.add_argument('--cpu', action='store_true',
                            help='Whether to train on the CPU. If ommited, will train on a GPU')
        parser.add_argument('--checkpoint', action='store_true', help='Whether to save chekpoints')
        parser.add_argument('--num-workers', type=int, default=0, help='Number of CPU cores to load data on')

        args = parser.parse_args()
        return Configuration(vars(args))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, dir_path):
        """Dump configurations to a JSON file."""
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        with open(f'{dir_path}/args.json', 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)