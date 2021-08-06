import argparse
import wandb
import torch

from pathlib import Path

import datasets.mnist as mnist
import datasets.cifar10 as cifar10
import constants
import metrics
from configuration import Configuration
from methods.moe.laplace_gating import apply_laplace_approximation

from util import *

def detailed_eval_save(args):
    model_args = Configuration.from_json(args.args_path)

    device='cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    # load the checkpoint and love to relevant device
    trainer = get_trainer(model_args, device=device)
    trainer.load_checkpoint(args.model_path)
    trainer.model.to(device)

    save_dir = '/'.join(args.args_path.split('/')[:-1]) if args.save_dir is None else f"{args.save_dir}/{args.args_path.split('/')[-2]}"

    print(f'Saving to {save_dir}')

    test_cifar_detailed(trainer, model_args, save_dir=save_dir)

def eval(args):

    model_args = Configuration.from_json(args.args_path)
    
    # update relevant arguments 
    model_args.checkpoint=False
    model_args.predict_gated=not args.moe_mean
    model_args.gating_laplace=args.moe_laplace_gating
    model_args.orig_path = args.model_path
    model_args.entropy_threshold = args.entropy_threshold

    device='cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    # load the checkpoint and love to relevant device
    trainer = get_trainer(model_args, device=device)
    trainer.load_checkpoint(args.model_path)
    trainer.model.to(device)
    
    # construct typical metric dict for testing
    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                'ECE': metrics.wrap_ece(bins=20), 
                'Brier': metrics.wrap_brier()}

    if args.moe_laplace_gating:
        # get data for LA fitting
        train_loader, val_loader = get_train_and_val_loaders(
                                dataset_type=model_args.dataset_type,
                                data_dir=model_args.data_dir,
                                batch_size=model_args.batch_size,
                                val_fraction=model_args.validation_fraction,
                                num_workers=model_args.num_workers,
                                )
        # fit the LA
        trainer = apply_laplace_approximation(
                    trainer, 
                    train_loader, 
                    val_loader, 
                    optimize_precision=args.fit_prec, 
                    entropy_threshold=0 if args.fit_et else args.entropy_threshold,
                    full_laplace=args.full_laplace,
                    )
        
        # for compatability due to architecture changes
        try:
            fit_prec = trainer.model.gating_network.net.prior_precision
            # if args.fit_et:
            #     trainer.model.gating_network.set_entropy_threshold(val_loader, percentile=90)
            #     model_args.entropy_threshold = trainer.model.gating_network.entropy_threshold
        except:
            fit_prec = trainer.model.gating_network.prior_precision

        print(trainer.model.gating_network.gate_by_entropy)
        # model_args.laplace_precision = fit_prec
        
        laplace_precisions = [fit_prec.cpu().item()] if (args.fit_prec or args.laplace_precisions is None) else [] 
        if args.laplace_precisions is not None:
            laplace_precisions += args.laplace_precisions

        trainer.model.eval()
        trainer.model.to(device)
        # evaluate as a fresh run for each precision value
        for prec in laplace_precisions:
            try:
                trainer.model.gating_network.net.prior_precision = prec
                if args.fit_et:
                    trainer.model.gating_network.set_entropy_threshold(val_loader, percentile=75, device=device)
                    model_args.entropy_threshold = trainer.model.gating_network.entropy_threshold
            except:
                trainer.model.gating_network.prior_precision = prec

            if args.log:
                model_args.laplace_precision = prec
                project_name = f'mphil-{"mnist-moe" if model_args.dataset_type == "mnist" else model_args.dataset_type}-laplace'
                wandb.init(project=project_name, entity='gintepe', dir=constants.LOGGING_DIR)
                wandb.config.update(model_args)

            print(f'Testing model.\nPrecision used: {prec}')

            if model_args.dataset_type == 'mnist':
                test_mnist(trainer, model_args, metric_dict)
            if 'cifar' in model_args.dataset_type:
                test_cifar(trainer, model_args, metric_dict)

            if args.log:
                wandb.finish() 

    # if there is no need to apply the laplace approximation, simply evaluate
    else:

        if args.log:
            project_name = f'mphil-{"mnist-moe" if model_args.dataset_type == "mnist" else model_args.dataset_type}' if model_args.project is None else model_args.project
            project_name = project_name if args.project is None else args.project
            if args.run_id is not None:
                wandb.init(id=args.run_id, project=project_name, entity='gintepe', dir=constants.LOGGING_DIR, resume='allow')
            else:
                wandb.init(project=project_name, entity='gintepe', dir=constants.LOGGING_DIR)

            if args.run_name is not None:
                wandb.run.name = args.run_name
                wandb.run.save()
            # wandb.config.update(model_args)

        trainer.model.eval()
        trainer.model.to(device)

        save_dir = '/'.join(args.args_path.split('/')[:-1]) if args.save_dir is None else f"{args.save_dir}/{args.args_path.split('/')[-2]}"

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        model_args.detailed_eval = True

        train_loader, val_loader = get_train_and_val_loaders(
                                dataset_type=model_args.dataset_type,
                                data_dir=model_args.data_dir,
                                batch_size=model_args.batch_size,
                                val_fraction=model_args.validation_fraction,
                                num_workers=model_args.num_workers,
                                seed=model_args.seed if model_args.seed is not None else 1
                                )

        test(trainer, model_args, metric_dict, wandb_log=True, save_dir=save_dir, val_loader=val_loader, moe_pred_toggle=args.toggle_moe_at)

        model_args.moe_toggle = args.toggle_moe_at

        # if model_args.dataset_type == 'mnist':
        #     test_mnist(trainer, model_args, metric_dict)
        # if 'cifar' in model_args.dataset_type:
        #     test_cifar(trainer, model_args, metric_dict)    

        if args.log:
            wandb.config.update(model_args)
            wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, help='Path to relevant checkpoint of arguments')
    parser.add_argument('--run-id', type=str, default=None, help='testing')
    parser.add_argument('--model-path', type=str, help='Path to relevant model checkpoint')
    parser.add_argument('--save-dir', type=str, default=None, help='Save directory for detailed evaluation')
    parser.add_argument('--project', type=str, default=None, help='Project to log to')
    parser.add_argument('--log', action='store_true', help='If present, log to wandb as a new run')
    parser.add_argument('--moe-mean', action='store_true', help='If present, use mean predictions for MoE, else use gated ones')
    parser.add_argument('--moe-laplace-gating', action='store_true', help='If present, use Laplace approximation for MoE gating')
    parser.add_argument('--full-laplace', action='store_true', 
        help='If present and --moe-laplace-gating is also set, use Laplace approximation for the full gating network')
    parser.add_argument('--laplace-precisions', nargs='+', action='extend', type=float, default=None,
        help='A list of LA precision values the network should be evaluated at')
    parser.add_argument('--entropy-threshold', type=float, default=None, help='If present, will use entropy-conditional gating')
    parser.add_argument('--fit-et', action='store_true', help='If present, use fitted entropy threshold')
    parser.add_argument('--fit-prec', action='store_true', help='If present, fit prior precision for first run on the validation set.')
    parser.add_argument('--cuda', action='store_true', help='If present, the model will be loaded and calculations performed on the GPU if possible')
    parser.add_argument('--detailed-eval', action='store_true', help='If present, will run evaluation on every cifar corruption type and save results in the logging directory')
    parser.add_argument('--run-name', type=str, default=None, help='name for the run in wandb')
    parser.add_argument('--toggle-moe-at', type=int, default=None, help='')

    args = parser.parse_args()

    # if args.detailed_cifar_eval:
    #     detailed_eval_save(args)
    # else:
    eval(args)