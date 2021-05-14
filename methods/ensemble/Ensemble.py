from methods.BaseTrainer import BaseTrainer
from methods.ensemble.models import *
from metrics import basic_cross_entropy

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

class Ensemble(BaseTrainer):
    def __init__(self, args, device):
        print(f'Initialising an ensemble of {args.n} networks')
        criterion = nn.CrossEntropyLoss()
        self.n = args.n
        super().__init__(args, criterion, device)
        self.optimizer = [optim.Adam(m.parameters(), lr=args.lr,) for m in self.model.networks]
        if args.scheduled_lr:
            self.use_scheduler()

    def get_model(self, args):
        model_class = self.get_model_class(args)
        return SimpleEnsemble(model_class, n=self.n)

    def predict_val(self, x):
        return self.model(x)[0]
    
    def predict_test(self, x):
        return self.model(x)[0]

    def use_scheduler(self):
        schedulers = []
        for i in range(self.n):
            schedulers.append(optim.lr_scheduler.StepLR(self.optimizer[i], 20, gamma = 0.1))
        self.scheduler = schedulers

    def train(self,
         train_loader,
         val_loader,
         epochs,
         log=True,):

        self.model.to(self.device)

        batches = 0
        
        if log:
            wandb.watch(self.model)
        
        for epoch in range(1, epochs + 1):
            self.model.train()

            print(f'Epoch {epoch}')
            correct = 0
            total = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                for X, y in tepoch:
                    
                    X, y = X.to(self.device), y.to(self.device)
                    [opt.zero_grad() for opt in self.optimizer]
                    
                    pred, y_hats = self.model(X)

                    losses = [self.criterion(y_hat, y) for y_hat in y_hats]

                    loss = 0
                    for i in range(len(losses)):
                        losses[i].backward()
                        self.optimizer[i].step()

                        loss += losses[i].item()

                    tepoch.set_postfix(loss=loss/len(losses))

                    batches += 1

                    _, predicted = torch.max(pred, 1)
                    correct += (predicted == y).sum().item()
                    total += X.shape[0]

                    if log:
                        wandb.log({'Training/loss': loss/len(losses), 'batch': batches})

            val_loss, val_acc = self.validate(val_loader, val_criterion=basic_cross_entropy)

            if log:
                self.log_info(correct/total, val_loss, val_acc, batches, epoch)

            if self.scheduler is not None:
                for schd in self.scheduler:
                    schd.step()
        
        if self.checkpoint_dir is not None:
            self.save_checkpoint(epoch, val_loss)


class NCEnsemble(BaseTrainer):
    def __init__(self, args, device):
        print(f'Initialising a negative-correlation normalised ensemble of {args.n} networks')
        criterion = nc_joint_regularised_cross_entropy
        self.n = args.n
        self.l = args.reg_weight
        super().__init__(args, criterion, device)
        self.optimizer = [optim.Adam(m.parameters(), lr=args.lr,) for m in self.model.networks]
        if args.scheduled_lr:
            self.use_scheduler()

    def get_model(self, args):
        model_class = self.get_model_class(args)
        return SimpleEnsemble(model_class, n=self.n)

    def predict_val(self, x):
        return self.model(x)[0]
    
    def predict_test(self, x):
        return self.model(x)[0]

    def use_scheduler(self):
        schedulers = []
        for i in range(self.n):
            schedulers.append(optim.lr_scheduler.StepLR(self.optimizer[i], 20, gamma = 0.1))
            self.scheduler = schedulers

    def train(self,
         train_loader,
         val_loader,
         epochs,
         log=True,):

        self.model.to(self.device)

        batches = 0
        
        if log:
            wandb.watch(self.model)
        
        for epoch in range(1, epochs + 1):
            self.model.train()

            print(f'Epoch {epoch}')
            correct = 0
            total = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                for X, y in tepoch:
                    
                    X, y = X.to(self.device), y.to(self.device)
                    [opt.zero_grad() for opt in self.optimizer]
                    
                    pred, y_hats = self.model(X)

                    losses = self.criterion(pred.detach(), y_hats, y, self.l)

                    loss = 0
                    for i in range(len(losses)):
                        losses[i].backward()
                        self.optimizer[i].step()

                        loss += losses[i].item()

                    tepoch.set_postfix(loss=loss/len(losses))

                    batches += 1

                    _, predicted = torch.max(pred, 1)
                    correct += (predicted == y).sum().item()
                    total += X.shape[0]

                    if log:
                        wandb.log({'Training/loss': loss/len(losses), 'batch': batches})

            val_loss, val_acc = self.validate(val_loader, val_criterion=basic_cross_entropy)

            if self.scheduler is not None:
                for schd in self.scheduler:
                    schd.step()

            if log:
                self.log_info(correct/total, val_loss, val_acc, batches, epoch)

        if self.checkpoint_dir is not None:
            self.save_checkpoint(epoch, val_loss)

def nc_joint_regularised_cross_entropy(mean_probs, pred_logits, ground_truth, l):
    prob_predictions = nn.functional.softmax(torch.stack(pred_logits.copy(), dim=1).detach(), dim=-1)
    sum_factor = (1 - torch.eye(len(pred_logits))).to(mean_probs.device).detach()
    sums = torch.matmul(sum_factor, (prob_predictions - mean_probs.unsqueeze(1)).detach())
    
    losses = []
    for i, pred in enumerate(pred_logits):
        cn = torch.nn.functional.cross_entropy(pred, ground_truth)

        reg = torch.mean(torch.sum((prob_predictions[:, i, :] - mean_probs) * sums[:, i, :], dim=-1))

        losses.append(cn + l*reg)

    return losses