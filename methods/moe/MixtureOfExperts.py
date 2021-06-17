from methods.BaseTrainer import BaseTrainer
from methods.moe.models import *
from metrics import basic_cross_entropy

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# TODO choices to be made: whether to average outputs or probabilities if using a fully end-to-end setup
# for now doing a simple one with averaged probs
class SimpleMoE(BaseTrainer):
    def __init__(self, args, device):

        criterion = basic_cross_entropy
        self.n = args.n
        super().__init__(args, criterion, device)
        self.val_criterion = basic_cross_entropy

    def get_model(self, args):
        """ Retrieves and initialises the relevant model as specified by args. """
        model_class = self.get_model_class(args)
        if args.dataset_type == 'cifar100':
            return DenseBasicMoE(model_class, n=self.n, num_classes=100)
        else:
            return DenseBasicMoE(model_class, n=self.n)
    
    def predict_val(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a validation step.
        """
        return self.model(x)#[0]
    
    def predict_test(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a testing step.
        """
        return self.model(x)

    # actually I think for this one (end-to-end, implicit, simple weighted avg gating) there does not need to be any special steps
    # it can just happily fit into the framework of the single network training with an appropriately implmented network. 
    # turns out this is needed but only due to always returning preds
    def train(self, train_loader, batches, log=True):  
        """
        Training step logic for a single epoch.

        Parameters
        -------
        - train_loader (torch.utils.data.DataLoader): iterator for the training data.
        - batches (int): number of batches seen so far.
        - log (bool): whether to use the weights and biases logger.

        Returns
        -------
        - correct (int): number of correct predictions observed.
        - total (int): number of datapoints observed.
        - batches (int): updated count of batches observed.
        """

        correct = 0
        total = 0

        self.model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                
                # move data to relevant device
                X, y = X.to(self.device), y.to(self.device)

                # compute loss        
                y_hat, preds = self.model(X)
                loss = self.criterion(y_hat, y)
                
                # backpropogate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                tepoch.set_postfix(loss=loss)
                batches += 1
                _, predicted = torch.max(y_hat, 1)
                correct += (predicted == y).sum().item()
                total += X.shape[0]

                if log:
                    wandb.log({'Training/loss': loss, 'batch': batches})
        
        return correct, total, batches    