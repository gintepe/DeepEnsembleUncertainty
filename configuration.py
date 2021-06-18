import argparse
import json
import os
import pprint
import torch
import pathlib

DEFAULT_DICT = {'data_dir': '/scratch/gp491/data', 'dataset_type': 'mnist', 'corrupted_test': False,
'validation_fraction': 0.1, 'method': 'single', 'n': 5, 'model': 'lenet', 'reg_weight': 0.5, 
'dropout': 0.5, 'scheduler': None, 'scheduler_step': 20, 'scheduler_rate': 0.1, 'batch_size': 250, 
'epochs': 15, 'lr': 0.003, 'weight_decay': 0, 'cpu': False, 'checkpoint': False, 'num_workers': 0, 
'reg_decay': 1, 'reg_min': 0, 'predict_gated': False, 'moe_type': 'dense', 'moe_gating': 'same'}


class Configuration(object):
    """Configuration for parameters exposed via the commandline."""

    def __init__(self, adict):
        # load defaults to avoid erros for missing values and backward compatability
        self.__dict__.update(DEFAULT_DICT)
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
                            choices=['cifar10', 'cifar100', 'mnist'], help='Dataset name')
        parser.add_argument('--corrupted-test', action='store_true',
                            help='Whether to use a corrupted (shifted) testing set. If omitted, will use the standar one.')
        parser.add_argument('--validation-fraction', type=float, default=0.1,
                            help='Fraction of the training set to be held out for validation, in cases where a dedicated set is unavailable.')

        # method config
        parser.add_argument('--method', type=str, default='single',
                            choices=['single', 'ensemble', 'mcdrop', 'ncensemble', 'ceensemble', 'moe'], help='method to run')
        parser.add_argument('--n', type=int, default=5, help='Size of the ensemble to be trained.')
        parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'mlp', 'resnet'],
                            help='Model architecture to be used')

        # training config
        parser.add_argument('--reg-weight', type=float, default=0.5,
                            help='Scaling factor for custom loss regularisation, initial value.')
        parser.add_argument('--reg-min', type=float, default=0.0,
                            help='Lower bound on the regularisation scaling constant.')
        parser.add_argument('--reg-decay', type=float, default=1, help='Exponential decay factor for regularisation weight')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for models that use it')
        parser.add_argument('--scheduler', type=str, choices=['step', 'exp', 'multistep'], default=None,
                            help='if set, will use a learning rate scheduler, as specified')
        parser.add_argument('--scheduler-step', type=int, default=20)
        parser.add_argument('--scheduler-rate', type=float, default=0.1)
        parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                            help='which optimizer to use. SGD will default to momentum of 0.9')
        parser.add_argument('--batch-size', type=int, default=250, help='Batch size to use in training')
        parser.add_argument('--epochs', type=int, default=15, help='Maximum number of epochs to train for')
        parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate')
        parser.add_argument('--weight-decay', type=float, default=0, help='Weight regularisation penalty')
        parser.add_argument('--cpu', action='store_true',
                            help='Whether to train on the CPU. If ommited, will train on a GPU')
        parser.add_argument('--checkpoint', action='store_true', help='Whether to save chekpoints')
        parser.add_argument('--num-workers', type=int, default=0, help='Number of CPU cores to load data on')

        # MoE specific config
        parser.add_argument('--predict-gated', action='store_true', help='Wether a MoE model should use gating or a simple mean in predictions')
        parser.add_argument('--moe-type', type=str, choices=['dense', 'fixed'], default='dense',
                            help='Type of a MoE model. Dense uses a gating network to determine weights for averaging, fixed is a dummy with fixed allocatons.')
        parser.add_argument('--moe-gating', type=str, choices=['same', 'simple'], default='same',
                            help='Type of a gating network to use in a MoE model. Same sets the network to have the same architecture as experts. Simple will be an arbitrary MLP of dimensions I like.')    

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