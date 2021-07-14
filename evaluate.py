import argparse
import wandb
import torch

import datasets.mnist as mnist
import datasets.cifar10 as cifar10
import constants
import metrics
from configuration import Configuration
from methods.moe.laplace_gating import apply_laplace_approximation

from util import *

def eval(args):

    model_args = Configuration.from_json(args.args_path)
    
    model_args.checkpoint=False
    model_args.predict_gated=not args.moe_mean
    model_args.gating_laplace=args.moe_laplace_gating
    model_args.orig_path = args.model_path
    model_args.entropy_threshold = args.entropy_threshold

    device='cuda' if args.cuda else 'cpu'

    trainer = get_trainer(model_args, device=device)
    trainer.load_checkpoint(args.model_path)
    
    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                'ECE': metrics.wrap_ece(bins=20), 
                'Brier': metrics.wrap_brier()}

    if args.log:
        project_name = f'mphil-{"mnist-moe" if model_args.dataset_type == "mnist" else model_args.dataset_type}{"-laplace" if args.moe_laplace_gating else ""}'
        wandb.init(project=project_name, entity='gintepe', dir=constants.LOGGING_DIR)
        # wandb.config.update(model_args)

    if args.moe_laplace_gating:
        train_loader, val_loader = get_train_and_val_loaders(
                                dataset_type=model_args.dataset_type,
                                data_dir=model_args.data_dir,
                                batch_size=model_args.batch_size,
                                val_fraction=model_args.validation_fraction,
                                num_workers=model_args.num_workers,
                                )

        trainer = apply_laplace_approximation(trainer, train_loader, val_loader, optimize_precision=args.fit_prec, entropy_threshold=0 if args.fit_et else args.entropy_threshold)
        print('precision used:')
        try:
            fit_prec = trainer.model.gating_network.net.prior_precision
            if args.fit_et:
                trainer.model.gating_network.set_entropy_threshold(val_loader, percentile=90)
                model_args.entropy_threshold = trainer.model.gating_network.entropy_threshold

        except:
            fit_prec = trainer.model.gating_network.prior_precision

        print(fit_prec)
        model_args.laplace_precision = fit_prec



    trainer.model.eval()
    trainer.model.to(device)
    print(trainer.device)

    if model_args.dataset_type == 'mnist':
        test_mnist(trainer, model_args, metric_dict)
    if 'cifar' in model_args.dataset_type:
        test_cifar(trainer, model_args, metric_dict)    

    if args.log:
        wandb.config.update(model_args)
        wandb.finish()

    if args.moe_laplace_gating and (args.laplace_precisions is not None):

        for prec in args.laplace_precisions:

            try:
                trainer.model.gating_network.net.prior_precision = prec
                if args.fit_et:
                    trainer.model.gating_network.set_entropy_threshold(val_loader, percentile=90)
                    model_args.entropy_threshold = trainer.model.gating_network.entropy_threshold

            except:
                trainer.model.gating_network.prior_precision = prec

            if args.log:
                model_args.laplace_precision = prec
                project_name = f'mphil-{"mnist-moe" if model_args.dataset_type == "mnist" else model_args.dataset_type}{"-laplace" if args.moe_laplace_gating else ""}'
                wandb.init(project=project_name, entity='gintepe', dir=constants.LOGGING_DIR)
                wandb.config.update(model_args)

            print('precision used:')
            print(prec)

            if model_args.dataset_type == 'mnist':
                test_mnist(trainer, model_args, metric_dict)
            if 'cifar' in model_args.dataset_type:
                test_cifar(trainer, model_args, metric_dict)

            if args.log:
                wandb.finish() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, help='Path to relevant checkpoint of arguments')
    parser.add_argument('--model-path', type=str, help='Path to relevant model checkpoint')
    parser.add_argument('--log', action='store_true', help='If present, log to wandb as a new run')
    parser.add_argument('--moe-mean', action='store_true', help='If present, use mean predictions for MoE, else use gated ones')
    parser.add_argument('--moe-laplace-gating', action='store_true', help='If present, Laplace approximatior for MoE gating')
    parser.add_argument('--laplace-precisions', nargs='+', action='extend', type=float, default=None)
    parser.add_argument('--entropy-threshold', type=float, default=None, help='if present, will use entropy-conditional gating')
    parser.add_argument('--fit-et', action='store_true', help='If present, use fitted entropy threshold')
    parser.add_argument('--fit-prec', action='store_true', help='If present, fit prior precision for first run on the validation set.')
    parser.add_argument('--cuda', action='store_true', help='use cuda or not. Currently buggy for cuda.')

    args = parser.parse_args()
    eval(args)