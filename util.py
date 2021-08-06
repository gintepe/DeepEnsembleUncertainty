import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle

import datasets.mnist as mnist
import datasets.cifar10 as cifar10
import constants
import metrics
from configuration import Configuration

from methods.SingleNetwork import SingleNetwork
from methods.mcdropout.MCDropout import MCDropout
from methods.ensemble.Ensemble import Ensemble, NCEnsemble, CEEnsemble
from methods.moe.MixtureOfExperts import SimpleMoE, TwoStepMoE


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def plot_calibration_hist(calibration_hist):
    values, bins = calibration_hist

    width = bins[1] - bins[0]

    plt.bar(bins[:-1] + width/2, values, width*0.95)
    plt.plot(bins, bins, 'r--')

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.show()


def plot_cosine_similarity(trainer, remove_diag=False):
    """
    For an ensemble model, will compute and visualise a matrix of cosine similarity
    between the weights of individual predictors.

    Parameters
    --------
    - trainer (methods.BaseTrainer): trainer for the desired ensemble
    - remove_diag (bool): wether to subtract 1 from the main diagonal (self-similarity entries)

    Returns
    --------
    - mat (np.ndarray): cosine similarity matrix
    """
    
    mat = get_cosine_similarities(trainer)
    
    if remove_diag:
        mat -= np.eye(mat.shape[0])

    fig, ax = plt.subplots()
    im = ax.imshow(mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(mat.shape[0]))
    ax.set_yticks(np.arange(mat.shape[1]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(np.arange(mat.shape[0]) + 1)
    ax.set_yticklabels(np.arange(mat.shape[1]) + 1) 

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom")

    ax.set_title("Cosine similarity of network weights")
    fig.tight_layout()
    plt.show()

    return mat

def plot_disagreement_mat(mat, remove_diag=False):
    """
    Visualise a given disagreement matrix

    Parameters
    --------
    - mat (np.ndarray): disagreement matrix
    - remove_diag (bool): wether to subtract 1 from the main diagonal (self-disagreement entries)
    """

    if remove_diag:
        mat -= np.eye(mat.shape[0])

    fig, ax = plt.subplots()
    im = ax.imshow(mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(mat.shape[0]))
    ax.set_yticks(np.arange(mat.shape[1]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(np.arange(mat.shape[0]) + 1)
    ax.set_yticklabels(np.arange(mat.shape[1]) + 1) 

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Pairwise disagreement', rotation=-90, va="bottom")

    ax.set_title("Pairwise disagreement of individual predictors")
    fig.tight_layout()
    plt.show()

def get_param_tensor(model):
    """ Flatten the parameters of a model into a single one-dimensional vector """
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    return params

def get_cosine_similarities(trainer):
    """
    For an ensemble model, will compute a matrix of cosine similarity between 
    the weights of individual predictors.

    Parameters
    --------
    - trainer (methods.BaseTrainer): trainer for the desired ensemble

    Returns
    --------
    - mat (np.ndarray): cosine similarity matrix
    """
    with torch.no_grad():
        param_list = []
        for net in trainer.model.networks:
            params = get_param_tensor(net).cpu().numpy()
            param_list.append(params)

        weights = np.stack(param_list, axis=0)
        mat = cosine_similarity(weights)
    return mat

def load_trainer(run_id, checkpoint_epoch, device='cpu'):
    """
    Load a checkpoint from a given epoch of a given run.

    Parameters
    --------
    - run_id (str): id of the run, as it appears in the logging directory
    - checkpoint_epoch (int): which epoch was the desired checkpoint recorded on
    - device (str): which device (cpu or gpu) to load the checkpointed model to.

    Returns
    --------
    - trainer (methods.BaseTrainer):
    - model_args (namespace):
    """
    checkpointed_args = f'/scratch/gp491/wandb/checkpoints/{run_id}/args.json'
    checkpointed_model = f'/scratch/gp491/wandb/checkpoints/{run_id}/epoch_{checkpoint_epoch}.pth'

    model_args = Configuration.from_json(checkpointed_args)
    trainer = get_trainer(model_args, device=device)
    trainer.load_checkpoint(checkpointed_model)

    return trainer, model_args

def get_trainer(args, device):
    """ Select a relevant trainer based on the value specified as an argument. """
    if args.method == 'single':
        trainer = SingleNetwork(args, device)

    if args.method == 'mcdrop':
        trainer = MCDropout(args, device)

    if args.method == 'ensemble':
        trainer = Ensemble(args, device)

    if args.method == 'ncensemble':
        trainer = NCEnsemble(args, device)

    if args.method == 'ceensemble':
        trainer = CEEnsemble(args, device)

    if args.method == 'moe':
        trainer = SimpleMoE(args, device)

    if args.method == 'moe2step':
        trainer = TwoStepMoE(args, device)

    return trainer

def get_train_and_val_loaders(dataset_type, data_dir, batch_size, val_fraction, num_workers, seed=1):
    """ Loads validation and training data for an appropriate dataset """
    if dataset_type == 'mnist':
        return mnist.get_mnist_train_valid_loader(data_dir, batch_size, random_seed=seed, 
                                                    valid_size=val_fraction, num_workers=num_workers)
    if dataset_type == 'cifar10':
        return cifar10.get_cifar_train_valid_loader(data_dir, batch_size, augment=True, 
                                                        random_seed=seed, valid_size=val_fraction,
                                                        num_workers=num_workers)
    if dataset_type == 'cifar100':
        return cifar10.get_cifar_train_valid_loader(data_dir, batch_size, augment=True, 
                                                        random_seed=seed, valid_size=val_fraction,
                                                        num_workers=num_workers, is_cifar10=False)

def test(trainer, args, metric_dict, wandb_log=True, save_dir=None, val_loader=None, moe_pred_toggle=None):
    if args.detailed_eval:
        if args.dataset_type == 'mnist':
            test_mnist(trainer, args, metric_dict, save_dir=save_dir, val_loader=val_loader, moe_pred_toggle=moe_pred_toggle)
        if 'cifar' in args.dataset_type:
            # for logging
            test_cifar(trainer, args, metric_dict, moe_pred_toggle=moe_pred_toggle)
            # for detailed eval
            test_cifar_detailed(trainer, args, save_dir=save_dir, val_loader=val_loader, moe_pred_toggle=moe_pred_toggle)
    else:
        if args.dataset_type == 'mnist':
            test_mnist(trainer, args, metric_dict)
        if 'cifar' in args.dataset_type:
            test_cifar(trainer, args, metric_dict)

def test_mnist(trainer, args, metric_dict, wandb_log=True, save_dir=None, val_loader=None, moe_pred_toggle=None):
    """
    Testing logic for the MNIST dataset.
    If specified in args, it will carry out full testing on
    differently rotated and translated test images.

    Parameters
    --------
    - trainer (methods.BaseTrainer): method logic wrapper
    - args (namespace): parsed command line arguments
    - metric dict (dictionary {name: function (prob, gt) -> float}):
    metrics to be evaluated at each testing run
    - wandb_log (bool): whether to log results to the weights 
    and biases logger  
    """

    results = {}
    rotations = list(np.arange(0, 181, 15))
    translations = list(np.arange(0, 29, 2))

    results['rotations'] = rotations
    results['translations'] = rotations

    if val_loader is not None:
        print('Full validation eval')
        acc, metric_res, stat_tracker = trainer.test(test_loader=val_loader, metric_dict=metric_dict)
        results['val_accuracy'] = acc
        for name, val in metric_res.items():
            results[f'val_{name}'] = val

        total = stat_tracker.total
        results['val_confidence'] = stat_tracker.cum_conf / total
        results['val_disagreement'] = stat_tracker.disagreements / total
        

    accuracies = np.zeros(len(rotations))
    eces = np.zeros_like(accuracies)
    nlls = np.zeros_like(accuracies)
    briers = np.zeros_like(accuracies)
    confidences = np.zeros_like(accuracies)
    disagreements = np.zeros_like(accuracies)

    if args.corrupted_test:
        for i, rotation in enumerate(rotations):

            if moe_pred_toggle is not None:
                if moe_pred_toggle <= i:
                    trainer.gated_predict = False
                else:
                    trainer.gated_predict=True
            
            print(f'\nShift: {rotation}\n')
            test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensity=rotation, corruption='rotation')
            acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

            total = stat_tracker.total

            accuracies[i] = acc
            eces[i] = metric_res['ECE']
            briers[i] = metric_res['Brier']
            nlls[i] = metric_res['NLL']
            confidences[i] = stat_tracker.cum_conf / total
            disagreements[i] = stat_tracker.disagreements / total

            if wandb_log:
                wandb.log({'Test/rotated accuracy': acc, 'shift': rotation})
                for name, val in metric_res.items():
                    wandb.log({f'Test/rotated {name}': val, 'shift': rotation})
                stat_tracker.log_statistics(prefix='Test/rotated', shift=rotation)

        if save_dir is not None:
            with open(f'{save_dir}/res.txt', 'wb') as f:
                pickle.dump(results, f)

            np.savez(
                f'{save_dir}/rotation.npz', 
                accuracies=accuracies, 
                eces=eces, 
                briers=briers, 
                nlls=nlls,
                confidences=confidences,
                disagreements=disagreements
                )
            
        accuracies = np.zeros(len(translations))
        eces = np.zeros_like(accuracies)
        nlls = np.zeros_like(accuracies)
        briers = np.zeros_like(accuracies)
        confidences = np.zeros_like(accuracies)
        disagreements = np.zeros_like(accuracies)

        for i, translation in enumerate(translations):

            if moe_pred_toggle is not None:
                if moe_pred_toggle <= i:
                    trainer.gated_predict = False
                else:
                    trainer.gated_predict=True

            print(f'\nShift: {translation}\n')
            test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensity=translation, corruption='shift')
            acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

            total = stat_tracker.total

            accuracies[i] = acc
            eces[i] = metric_res['ECE']
            briers[i] = metric_res['Brier']
            nlls[i] = metric_res['NLL']
            confidences[i] = stat_tracker.cum_conf / total
            disagreements[i] = stat_tracker.disagreements / total

            if wandb_log:
                wandb.log({'Test/translated accuracy': acc, 'shift': translation})
                for name, val in metric_res.items():
                    wandb.log({f'Test/translated {name}': val, 'shift': translation})
                stat_tracker.log_statistics(prefix='Test/translated', shift=translation)

        if save_dir is not None:
            np.savez(
                f'{save_dir}/translation.npz', 
                accuracies=accuracies, 
                eces=eces, 
                briers=briers, 
                nlls=nlls,
                confidences=confidences,
                disagreements=disagreements
                )
                
    else:
        test_loader = mnist.get_test_loader(args.data_dir, args.batch_size, corrupted=False)
        acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)
        print(f'Testing\nAccuracy: {acc}')


def test_cifar(trainer, args, metric_dict, wandb_log=True, save_dir=None, moe_pred_toggle=None):
    """
    Testing logic for the CIFAR datasets.
    If specified in args, testing will be carried out for different
    lovels of corruptions applied to the test set images.

    Parameters
    --------
    - trainer (methods.BaseTrainer): method logic wrapper
    - args (namespace): parsed command line arguments
    - metric dict (dictionary {name: function (prob, gt) -> float}):
    metrics to be evaluated at each testing run
    - wandb_log (bool): whether to log results to the weights 
    and biases logger  
    """

    is_cifar10 = args.dataset_type == 'cifar10'
    
    test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=False, is_cifar10=is_cifar10)
    acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

    if wandb_log and args.corrupted_test:
        wandb.log({'Test/corrupted accuracy': acc, 'intensity': 0})
        for name, val in metric_res.items():
            wandb.log({f'Test/corrupted {name}': val, 'intensity': 0})
        stat_tracker.log_statistics(prefix='Test/corrupted', shift=0, shift_name='intensity')
        

    print(f'Testing\nAccuracy: {acc}')

    if args.corrupted_test:
        for i in range(1, 6):

            if moe_pred_toggle is not None:
                if moe_pred_toggle <= i:
                    trainer.gated_predict = False
                else:
                    trainer.gated_predict=True

            test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensities=[i], is_cifar10=is_cifar10)
            acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

            if wandb_log:
                wandb.log({'Test/corrupted accuracy': acc, 'intensity': i})
                for name, val in metric_res.items():
                    wandb.log({f'Test/corrupted {name}': val, 'intensity': i})
                stat_tracker.log_statistics(prefix='Test/corrupted', shift=i, shift_name='intensity')
                

def load_res_mnist(directory):
    """
    Load the results as saved by the test_mnist function
    """
    res_dict_path = f'{directory}/res.txt'
    rot_np_path = f'{directory}/rotation.npz'
    tran_np_path = f'{directory}/translation.npz'

    with open(res_dict_path, 'rb') as f:
        res_dict = pickle.load(f)

    rot_np = np.load(rot_np_path)
    tran_np = np.load(tran_np_path)

    return res_dict, rot_np, tran_np

def load_res_cifar(directory):
    """
    Load the results as saved by the test_cifar_detailed function
    """
    res_dict_path = f'{directory}/res.txt'
    res_np_path = f'{directory}/res.npz'

    with open(res_dict_path, 'rb') as f:
        res_dict = pickle.load(f)

    res_np = np.load(res_np_path)

    return res_dict, res_np


def test_cifar_detailed(trainer, args, save_dir=None, val_loader=None, moe_pred_toggle=None):
    """
    Testing logic for the CIFAR datasets.
    Testing will be carried out for different types and levels of corruptions applied to the 
    test set images.

    Parameters
    --------
    - trainer (methods.BaseTrainer): method logic wrapper
    - args (namespace): parsed command line arguments
    - save_dir (str): directory to save the evaluation results in
    """

    results = {}

    is_cifar10 = args.dataset_type == 'cifar10'

    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                'ECE': metrics.wrap_ece(bins=20), 
                'Brier': metrics.wrap_brier()}
    
    test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=False, is_cifar10=is_cifar10)
    print('ID test eval')
    acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

    results['test_accuracy'] = acc
    for name, val in metric_res.items():
        results[f'test_{name}'] = val
    total=stat_tracker.total
    results['test_confidence'] = stat_tracker.cum_conf / total
    results['test_disagreement'] = stat_tracker.disagreements / total

    if val_loader is not None:
        print('Full validation eval')
        acc, metric_res, stat_tracker = trainer.test(test_loader=val_loader, metric_dict=metric_dict)
        results['val_accuracy'] = acc
        for name, val in metric_res.items():
            results[f'val_{name}'] = val
        total=stat_tracker.total
        results['val_confidence'] = stat_tracker.cum_conf / total
        results['val_disagreement'] = stat_tracker.disagreements / total

    corruptions = constants.CORRUPTIONS
    intensities = list(range(1,6))

    results['corruptions'] = corruptions
    results['intesites'] = intensities
    
    print(f'Testing\nAccuracy: {acc}')

    accuracies = np.zeros((len(corruptions), len(intensities)))
    eces = np.zeros_like(accuracies)
    nlls = np.zeros_like(accuracies)
    briers = np.zeros_like(accuracies)
    confidences = np.zeros_like(accuracies)
    disagreements = np.zeros_like(accuracies)

    for corruption_id, corruption in enumerate(corruptions):
        for intensity in range(1, 6):

            if moe_pred_toggle is not None:
                if moe_pred_toggle <= intensity:
                    trainer.gated_predict = False
                else:
                    trainer.gated_predict=True

            print(f'\nShift {corruption} level {intensity} eval')
            test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=True, intensities=[intensity], corruptions=[corruption], is_cifar10=is_cifar10)
            acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)
            
            total = stat_tracker.total

            accuracies[corruption_id, intensity-1] = acc
            eces[corruption_id, intensity-1] = metric_res['ECE']
            briers[corruption_id, intensity-1] = metric_res['Brier']
            nlls[corruption_id, intensity-1] = metric_res['NLL']
            confidences[corruption_id, intensity-1] = stat_tracker.cum_conf / total
            disagreements[corruption_id, intensity-1] = stat_tracker.disagreements / total

    if save_dir is not None:
        with open(f'{save_dir}/res.txt', 'wb') as f:
            pickle.dump(results, f)

        np.savez(
            f'{save_dir}/res.npz', 
            accuracies=accuracies, 
            eces=eces, 
            briers=briers, 
            nlls=nlls,
            confidences=confidences,
            disagreements=disagreements
            )
            



def cifar_regen_res_file(trainer, args, save_dir=None, val_loader=None):
    """

    Parameters
    --------
    - trainer (methods.BaseTrainer): method logic wrapper
    - args (namespace): parsed command line arguments
    - save_dir (str): directory to save the evaluation results in
    """

    results = {}

    is_cifar10 = args.dataset_type == 'cifar10'

    metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                'ECE': metrics.wrap_ece(bins=20), 
                'Brier': metrics.wrap_brier()}
    
    test_loader = cifar10.get_test_loader(args.data_dir, args.batch_size, corrupted=False, is_cifar10=is_cifar10)
    print('ID test eval')
    acc, metric_res, stat_tracker = trainer.test(test_loader=test_loader, metric_dict=metric_dict)

    results['test_accuracy'] = acc
    for name, val in metric_res.items():
        results[f'test_{name}'] = val

    total=stat_tracker.total
    results['test_confidence'] = stat_tracker.cum_conf / total
    results['test_disagreement'] = stat_tracker.disagreements / total

    if val_loader is not None:
        print('Full validation eval')
        acc, metric_res, stat_tracker = trainer.test(test_loader=val_loader, metric_dict=metric_dict)
        total=stat_tracker.total
        results['val_accuracy'] = acc
        for name, val in metric_res.items():
            results[f'val_{name}'] = val
        results['val_confidence'] = stat_tracker.cum_conf / total
        results['val_disagreement'] = stat_tracker.disagreements / total

    corruptions = constants.CORRUPTIONS
    intensities = list(range(1,6))

    results['corruptions'] = corruptions
    results['intesites'] = intensities        

    if save_dir is not None:
        with open(f'{save_dir}/res.txt', 'wb') as f:
            pickle.dump(results, f)    
            

