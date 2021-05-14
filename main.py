import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import constants
import metrics
from configuration import Configuration

from util import *


if __name__ == '__main__':

    args = Configuration.parse_cmd()

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

    if args.checkpoint:
        log_dir = f"{constants.CHECKPOINT_DIR}/{wandb.run.dir.split('/')[-2]}"
        trainer.checkpoint_dir = log_dir
        args.to_json(log_dir)

    trainer.train(train_loader, val_loader, epochs=args.epochs)

    
    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                    'ECE': metrics.wrap_ece(bins=20), 
                    'Brier': metrics.wrap_brier()}

    if args.dataset_type == 'mnist':
        test_mnist(trainer, args, metric_dict)
    if args.dataset_type == 'cifar10':
        test_cifar(trainer, args, metric_dict)



