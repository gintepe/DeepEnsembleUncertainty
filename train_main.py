import argparse

parser = argparse.ArgumentParser(description='Train a configurable ensemble on a given dataset.')
parser.add_argument('--dataset_type', type=str, default='cifar10',
                    choices=['cifar10', 'mnist'], help='Dataset name')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of CPU cores to load data on')
parser.add_argument('--batch_size', type=int, default=250,
                    help='Batch size to use in training')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Initial learning rate')
parser.add_argument('--cpu', action='store_true',
                    help='Whether to train on the CPU. If ommited, will train on a GPU')
parser.add_argument('--model', type=str, default='test',
                    choices=['test'])

args = parser.parse_args()