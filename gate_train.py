import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import wandb
import scipy
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from tqdm import tqdm

from metrics import basic_cross_entropy
from configuration import Configuration, DEFAULT_DICT
from util import *
from datasets import data_util

from methods.BaseTrainer import BaseTrainer
from methods.moe.models import SparseDispatcher
from methods.moe.gate_models import get_gating_network
from methods.moe.laplace_gating import get_adjusted_loader

class WrapperModel(nn.Module):
    def __init__(self, experts, gating):
        """
        Initialise the model.

        """
        super().__init__()
        self.experts = experts
        self.n = len(experts)
        for param in experts.parameters():
            param.requires_grad = False
        self.gating_network = gating

    def forward(self, x, labels=None):
        """
        
        """
        preds = [net(x) for net in self.experts]
        weights = self.gating_network(x)

        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)

        return weights, combined_pred, preds

class NoiseDataset(Dataset):
    def __init__(self, img_size, channels, n, size):
        super().__init__()
        self.size = size
        self.data = torch.randn(size, channels, img_size, img_size)
        self.label = torch.ones(n)/n

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.label

class UniformWrapperDataset(Dataset):
    def __init__(self, dataset, n):
        super().__init__()
        self.dataset = dataset
        self.label = torch.ones(n)/n

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample, _ = self.dataset[idx]
        return sample, self.label

class TrueCorruptedMNIST(Dataset):
    def __init__(self, n, size, transform, data_dir):
        super().__init__()
        self.label = torch.ones(n)/n
        self.size = size
        self.images, self.labels = data_util.load_mnist(f'{data_dir}/MNIST/raw', 'train')
        self.transform=transform

    def __len__(self):
        return(self.size)

    def __getitem__(self, idx):

        img = self.images[idx%self.images.shape[0]].reshape((28, 28, 1)).copy()

        rf = np.random.rand(1)
        rotations = np.arange(15, 181, 15)
        translations = np.arange(2, 27, 2)
        if rf > 0.5:
            if self.transform:
                img = self.transform(img)  
            img = TF.rotate(img, float(random.choice([-1, 1])*np.random.choice(rotations)))
        else:
            translation = np.random.choice(translations)
            translated_img = np.zeros_like(img)
            translated_img[:-1*translation, :, :] = img[translation:, :, :]
            translated_img[-1*translation:, :, :] = img[:translation, :, :]
            img = translated_img
            if self.transform:
                img = self.transform(img)  
        
        return img, self.label

def get_gt_shifted_loader(dataset_type, loader, n_experts, data_dir):
    print('Ground truth corruptions used for outlier training')
    if dataset_type == 'mnist':
        data = TrueCorruptedMNIST(n_experts, len(loader.dataset), loader.dataset.transform, data_dir)
    else:
        new_loader = cifar10.get_test_loader(data_dir, loader.batch_size, corrupted=True, intensities=[3,4], is_cifar10='00' not in dataset_type)
        data = UniformWrapperDataset(new_loader.dataset, n_experts)
    return DataLoader(data, loader.batch_size, shuffle=True)

def get_equivalent_noise_loader(loader, n_experts):
    print('Random noise used for outlier training')
    item = loader.dataset[0][0]
    img_size = item.shape[-1]
    channels = item.shape[0]
    size = len(loader.dataset)
    noise_data = NoiseDataset(img_size, channels, n_experts, size)

    return DataLoader(noise_data, loader.batch_size)


def gating_criterion(gating_out, preds, combined_pred, gt, gate_gt):
    return basic_cross_entropy(gating_out, gate_gt)

def run_train_epoch(model, criterion, optimizer, device, train_loader, ood_loader=None):
    correct = 0
    gate_oracle = 0
    total = 0
    ep_loss = 0

    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:

        if ood_loader is not None:
            iter_ood = iter(ood_loader)


        for X, y, y_gate in tepoch:
            
            # move data to relevant device
            X, y, y_gate = X.to(device), y.to(device), y_gate.to(device)

            # compute loss        
            gating_prediction, combined_prediction, individual_predictions = model(X)
            loss = criterion(gating_prediction, individual_predictions, combined_prediction, y, y_gate)

            if ood_loader is not None:
                ood_X, ood_y = next(iter_ood)
                ood_X, ood_y = ood_X.to(device), ood_y.to(device)
                gp_ood, cp_ood, ips_ood = model(ood_X)
                
                ood_loss = metrics.softXEnt(gp_ood, ood_y)
                loss += ood_loss

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            tepoch.set_postfix(loss=loss)

            ep_loss += loss * X.shape[0]

            _, predicted = torch.max(combined_prediction, 1)
            correct += (predicted == y).sum().item()


            _, predicted = torch.max(gating_prediction, 1)
            gate_oracle += (predicted == y_gate).sum().item()
            total += X.shape[0]
    
    return correct, total, ep_loss, gate_oracle

def validate(model, criterion, device, val_loader):
    correct = 0
    gate_oracle = 0
    total = 0
    ep_loss = 0

    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            for X, y, y_gate in tepoch:
                
                # move data to relevant device
                X, y, y_gate = X.to(device), y.to(device), y_gate.to(device)

                # compute loss        
                gating_prediction, combined_prediction, individual_predictions = model(X)
                loss = criterion(gating_prediction, individual_predictions, combined_prediction, y, y_gate)

                loss = loss.item()
                tepoch.set_postfix(loss=loss)

                ep_loss += loss * X.shape[0]
                _, predicted = torch.max(combined_prediction, 1)
                correct += (predicted == y).sum().item()


                _, predicted = torch.max(gating_prediction, 1)
                gate_oracle += (predicted == y_gate).sum().item()
                total += X.shape[0]
        
    return correct, total, ep_loss, gate_oracle


def fit_gating(
        subnets, 
        gating_network, 
        train_loader, 
        valid_loader, 
        lr, 
        weight_decay,
        criterion,
        device, 
        epochs, 
        log=False,
        ood_loader=None
        ):

    model = WrapperModel(subnets, gating_network)
    optimizer = torch.optim.Adam(model.gating_network.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    if log:
        wandb.watch(model)

    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}')

        correct, total, ep_loss, gate_oracle = run_train_epoch(model, criterion, optimizer, device, train_loader, ood_loader)

        print(f'\nTraining\n--------------\nEnsemble accuracy {correct/total}\nGate oracle accuracy {gate_oracle/total}\nLoss {ep_loss/total}')

        if log:
            wandb.log({'Training/Ensemble Accuracy': correct/total, 'epoch':epoch})
            wandb.log({'Training/Gating Accuracy': gate_oracle/total, 'epoch':epoch})
            wandb.log({'Training/Loss': ep_loss/total, 'epoch':epoch})

        correct, total, ep_loss, gate_oracle = run_train_epoch(model, criterion, optimizer, device, valid_loader)
        
        print(f'\nValidation\n--------------\nEnsemble accuracy {correct/total}\nGate oracle accuracy {gate_oracle/total}\nLoss {ep_loss/total}')

        if log:
            wandb.log({'Validation/Ensemble Accuracy': correct/total, 'epoch':epoch})
            wandb.log({'Validation/Gating Accuracy': gate_oracle/total, 'epoch':epoch})
            wandb.log({'Validation/Loss': ep_loss/total, 'epoch':epoch})

    return model.experts, model.gating_network
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, help='Path to relevant checkpoint of arguments')
    parser.add_argument('--model-path', type=str, help='Path to relevant model checkpoint')
    parser.add_argument('--cuda', action='store_true', help='If present, the model will be loaded and calculations performed on the GPU if possible')
    parser.add_argument('--log', action='store_true', help='If present, log to wandb as a new run')
    parser.add_argument('--g-type', type=str, choices=['same', 'simple', 'mcd_simple', 'mcdc_simple', 'mcd_lenet', 'conv'], default='same',
                            help='Type of a gating network to use in a MoE model.')    
    parser.add_argument('--g-epochs', type=int, default=15, help='Maximum number of epochs to train for')
    parser.add_argument('--g-batch_size', type=int, default=128, help='Batch size for gating training')
    parser.add_argument('--g-lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--g-weight-decay', type=float, default=0, help='Weight regularisation penalty')
    parser.add_argument('--g-training-mode', type=str, choices=['gate', 'ensemble', 'sum'], default='gate',)
    parser.add_argument('--g-outliers', type=str, choices=['noise', 'gt'], default=None,)
    parser.add_argument('--eval-orig', action='store_true', help='If present, an evaluation is run for the original checkpoint')
    
    args = parser.parse_args()
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    model_args = Configuration.from_json(args.args_path)
    trainer = get_trainer(model_args, device)
    trainer.load_checkpoint(args.model_path)

    if args.log:
        project_name = f'mphil-{model_args.dataset_type}-gate-train'
        wandb.init(project=project_name, entity='gintepe', dir=constants.LOGGING_DIR)
        wandb.config.update(args)

    if args.eval_orig:
        metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                    'ECE': metrics.wrap_ece(bins=20), 
                    'Brier': metrics.wrap_brier()}

        if model_args.dataset_type == 'mnist':
            test_mnist(trainer, model_args, metric_dict, wandb_log=args.log)
        if 'cifar' in model_args.dataset_type:
            test_cifar(trainer, model_args, metric_dict, wandb_log=args.log)

    else:
        train_loader, valid_loader = get_train_and_val_loaders(
                                        model_args.dataset_type, 
                                        model_args.data_dir, 
                                        batch_size=args.g_batch_size, 
                                        val_fraction=0.1, 
                                        num_workers=0
                                    )

        # lazy way to handle both regular ensembeles and MoEs
        try:
            models = trainer.model.experts
        except:
            models = trainer.model.networks

        gate_train_loader = get_adjusted_loader(models, train_loader, return_original=True, device=device)
        gate_valid_loader = get_adjusted_loader(models, valid_loader, return_original=True, device=device)

        n = len(models)
        data_feat = 28*28 if model_args.dataset_type == 'mnist' else 32*32
        
        gating_network = get_gating_network(trainer.get_model_class(model_args), args.g_type, data_feat, n)

        if args.g_training_mode == 'ensemble':
            criterion = metrics.ensemble_criterion
        elif args.g_training_mode == 'sum':
            criterion = metrics.loss_sum_criterion
        else:
            criterion = gating_criterion

        ood_loader=None
        if args.g_outliers is not None:
            if args.g_outliers == 'noise':
                ood_loader = get_equivalent_noise_loader(train_loader, n)
            elif args.g_outliers == 'gt':
                ood_loader = get_gt_shifted_loader(model_args.dataset_type, train_loader, n, model_args.data_dir)
        
        # print(len(train_loader.dataset), len(ood_loader.dataset))

        networks, gate = fit_gating(
            models,
            gating_network,
            gate_train_loader,
            gate_valid_loader,
            lr=args.g_lr,
            weight_decay=args.g_weight_decay,
            criterion=criterion,
            device=device,
            epochs=args.g_epochs,
            log=args.log,
            ood_loader=ood_loader
        )
        
        dummy_args = model_args
        dummy_args.orginal_method = model_args.method
        dummy_args.orig_path = args.model_path
        dummy_args.moe_gating = 'simple'
        dummy_args.method = 'moe'
        dummy_args.moe_type = 'dense'
        dummy_args.predict_gated = True
        dummy_args.n = n
        dummy_args.cpu = device == 'cpu'

        if args.log:
            wandb.config.update(dummy_args)

        dummy_trainer = get_trainer(dummy_args, device)

        dummy_trainer.model.gating_network = gate
        dummy_trainer.model.experts = networks

        metric_dict = {'NLL': lambda p, g: metrics.basic_cross_entropy(p, g).item(), 
                        'ECE': metrics.wrap_ece(bins=20), 
                        'Brier': metrics.wrap_brier()}

        if dummy_args.dataset_type == 'mnist':
            test_mnist(dummy_trainer, dummy_args, metric_dict, wandb_log=args.log)
        if 'cifar' in dummy_args.dataset_type:
            test_cifar(dummy_trainer, dummy_args, metric_dict, wandb_log=args.log)


