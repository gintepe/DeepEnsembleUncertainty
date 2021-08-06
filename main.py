import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from pathlib import Path

import constants
import metrics
from configuration import Configuration

from util import *


if __name__ == '__main__':

    args = Configuration.parse_cmd()

    project = args.project if args.project is not None else f'mphil-{"mnist-moe" if args.dataset_type == "mnist" else args.dataset_type}'

    wandb.init(project=project, entity='gintepe', dir=constants.LOGGING_DIR)
    wandb.config.update(args)

    if args.seed is not None:
        seed_all(args.seed)

    if args.run_name is not None:
        wandb.run.name = args.run_name
        wandb.run.save()

    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'

    train_loader, val_loader = get_train_and_val_loaders(
                                dataset_type=args.dataset_type,
                                data_dir=args.data_dir,
                                batch_size=args.batch_size,
                                val_fraction=args.validation_fraction,
                                num_workers=args.num_workers,
                                seed=args.seed if args.seed is not None else 1
                                )

    trainer = get_trainer(args, device)

    if args.checkpoint:
        subdir = f'{args.log_subdir}/' if args.log_subdir is not None else ''
        log_dir = f"{constants.CHECKPOINT_DIR}/{subdir}{wandb.run.dir.split('/')[-2]}"
        trainer.checkpoint_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        args.to_json(log_dir)

    trainer.fit(train_loader, val_loader, epochs=args.epochs, early_stop_threshold=args.early_stop_tol)

    
    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                    'ECE': metrics.wrap_ece(bins=20), 
                    'Brier': metrics.wrap_brier()}

    test(trainer, args, metric_dict, wandb_log=True, save_dir=trainer.checkpoint_dir, val_loader=val_loader)

    # if args.dataset_type == 'mnist':
    #     test_mnist(trainer, args, metric_dict)
    # if 'cifar' in args.dataset_type:
    #     test_cifar(trainer, args, metric_dict)



