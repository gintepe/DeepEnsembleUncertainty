import argparse
import wandb
import torch

import datasets.mnist as mnist
import datasets.cifar10 as cifar10
import constants
import metrics
from configuration import Configuration

from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, help='Path to relevant checkpoint of arguments')
    parser.add_argument('--model-path', type=str, help='Path to relevant model checkpoint')

    args = parser.parse_args()

    model_args = Configuration.from_json(args.args_path)
    trainer = get_trainer(model_args, device='cpu')
    trainer.load_checkpoint(args.model_path)

    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                'ECE': metrics.wrap_ece(bins=20), 
                'Brier': metrics.wrap_brier()}

    test_mnist(trainer, model_args, metric_dict, wandb_log=False)

