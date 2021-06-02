import torch 
import wandb
import json
import pathlib
import numpy as np
import scipy.stats
import torch.optim as optim

from abc import abstractmethod
from tqdm import tqdm
from collections import defaultdict

import constants
from methods.models import *
from metrics import compute_accuracies_at_confidences

class BaseTrainer():
    """
    Base class encompassing the training loop for most methods
    """
    def __init__(self, args, criterion, device):
        """
        Initialise the trainer

        Parameters
        ------
        - args (namespace): parsed command line argumetns
        - criterion (function x, y -> scalar): training criterion to optimize
        - device (torch.device or str): device to perform the calculations on
        """

        self.device = device
        self.model = self.get_model(args)
        self.criterion = criterion
        self.val_criterion = criterion
        self.optimizer = self.get_optimizer(args)
        self.schedulers = self.get_schedulers(args)
        self.checkpoint_dir = None

    @abstractmethod
    def get_model(self, args):
        """ Retrieves and initialises the relevant model as specified by args. """
        raise NotImplementedError("Abstract method without implementation provided")

    def get_optimizer(self, args):
        if args.optimizer == 'adam':
            opt = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # opt = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            print('SGD optimizer')
            opt = optim.SGD(self.model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
        return opt
    
    # TODO: find a nicer way to incorporate schedulers
    def get_schedulers(self, args):
        """
        Initialise a learning rate scheduler as specified by args.
        """

        if args.scheduler == 'step':
            return [torch.optim.lr_scheduler.StepLR(self.optimizer, args.scheduler_step, gamma = args.scheduler_rate)]
            # return [torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[90, 135], gamma=args.scheduler_rate)]
        elif args.scheduler == 'exp':
            return [torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.scheduler_rate) ]
        elif args.scheduler == 'multistep':
            print('using multistep scheduler')
            return [torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[90, 135], gamma=args.scheduler_rate)]
        else:
            return []


    # TODO the following should probably always be the same 
    @abstractmethod
    def predict_val(self, x):
        """ Method to retrieve predictions for input x during a validation step """
        raise NotImplementedError("Abstract method without implementation provided")

    def save_checkpoint(self, epoch, loss):
        """
        Save a checkpoint for evaluation or continued training.
        The checkpointed data will be save in self.checkpoint_dir.
        In the case of networks with a list of optimizers, optimizer 
        state is omitted.

        Parameters
        ------
        - epoch (int): current epoch
        - loss (float): last training loss value
        """
        if isinstance(self.optimizer, list):
            # TODO: figure out if I can save the list of optimizers nicely
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'val_loss': loss
                }, f'{self.checkpoint_dir}/epoch_{epoch}.pth')
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': loss
                }, f'{self.checkpoint_dir}/epoch_{epoch}.pth')

    def load_checkpoint(self, checkpoint_path):
        """ A model weights and if possible optimizer state from the given checkpoint_path """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not isinstance(self.optimizer, list):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @abstractmethod
    def predict_test(self, x):
        """
        Compute prediction for the testing stage.
        Expected output: probabilities 
        """
        raise NotImplementedError("Abstract method without implementation provided")

    
    def get_model_class(self, args):
        """ Retrieve the model class corresponding to the specification in args """
        if args.model == 'lenet':
            return LeNet5
        if args.model == 'mlp':
            return MLP
        if args.model == 'resnet':
            return ResNet
        else:
            raise ValueError('invalid network type')

    # TODO: doesn't need to be an instance method
    def log_info(self, train_acc, val_loss, val_acc, val_conf, batches, epoch):
        """ 
        Log the given information to the weights and biasses logger. 

        Parameters
        -----
        - train_acc (float): training accuracy for the epoch
        - val_loss (flaot): mean validation loss for the epoch
        - vall_acc (float): validation accuracy for the epoch
        - vall_conf (float): mean validation confidence for the epoch
        - batches (int): total number of training batches seen
        - epoch (int): total number of epochs elapsed
        """
        wandb.log({'Training/accuracy': train_acc, 'batch': batches, 'epoch': epoch})
        wandb.log({'Validation/loss': val_loss, 'batch': batches, 'epoch': epoch})
        wandb.log({'Validation/accuracy': val_acc, 'batch': batches, 'epoch': epoch})
        wandb.log({'Validation/confidence': val_conf, 'batch': batches, 'epoch': epoch})

    def validate(self, val_loader):  
        """
        Validation step logic

        Parameters
        -------
        - val_loader (torch.utils.data.DataLoader): iterator for the validation data

        Returns
        -------
        - validation_loss (flaot): mean loss per sample
        - validation_accuracy (float): overall accuracy for the dataset
        - validation_confidence (float): mean prediction confidence for the dataset
        """  
        
        print('\nValidating')
        
        cum_loss = 0
        total = 0
        correct = 0
        cum_conf = 0

        self.model.eval()
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for X, y in tepoch:
                    
                    X, y = X.to(self.device), y.to(self.device)

                    y_hat = self.predict_val(X)

                    loss = self.val_criterion(y_hat, y)

                    loss = loss.item()
                    tepoch.set_postfix(loss=loss)

                    cum_loss += loss * X.size(0)
                    total += X.size(0)

                    confidence, predicted = torch.max(y_hat, 1)
                    correct += (predicted == y).sum().item()
                    cum_conf += confidence.sum().item()

            validation_loss = cum_loss/total
            validation_accuracy = correct/total
            validation_confidence = cum_conf/total

            print(f'Validation loss: {validation_loss}; accuracy: {validation_accuracy}\n')
                
        return validation_loss, validation_accuracy, validation_confidence

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
                y_hat = self.model(X)
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

    def fit(self,
            train_loader,
            val_loader,
            epochs,
            log=True,):

        """
        Training loop logic.

        Parameters
        -------
        - train_loader (torch.utils.data.DataLoader): iterator for the training data.
        - val_loader (torch.utils.data.DataLoader): iterator for the validation data.
        - epochs (int): maximum number of epochs to train for.
        - log (bool): whether to use the weights and biases logger.
        """
        print(self.optimizer)
        # print(self.schedulers[0])
        batches = 0

        self.model.to(self.device)
        
        if log:
            wandb.watch(self.model)
        
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}')
            
            correct, total, batches = self.train(train_loader, batches, log)
            val_loss, val_acc, val_conf = self.validate(val_loader)

            if log:
                self.log_info(correct/total, val_loss, val_acc, val_conf, batches, epoch)

            for sched in self.schedulers:
                sched.step()

        if self.checkpoint_dir is not None:
            self.save_checkpoint(epoch, val_loss)

    def test(self, test_loader, metric_dict, confidence_thresholds=None, entropy_bins=None):  
        """
        Testing/evaluation logic.
        Returns and computes some extra information to allow for further data visualisaton.
        For only customisable behaviour, discard the final 3 returned items and last 2 params.

        Parameters
        -------
        - test_loader (torch.utils.data.DataLoader): iterator for the test data.
        - metric_dict (dictionary {name: function (prob, gt) -> float}): metrics to be evaluated 
          at each testing step. Must be possible to aggregate via mean.
        - confidence_thresholds (np.ndarray): if thresholded accuracies and counts are needed, this 
          parameter should contain a list of increasing thresholds in the range [0, 1].
        - entropy_bins (np.ndarray): if binned entrypy counts are needed, this parameter should 
          contain a list of bin boundaries in a desired range.

        Returns
        -------
        - test_accuracy (float): Accuracy of the predictions accross the testing set.
        - metric_accumulators (dictionary {name: float}): mean values of the metrics given.
        - thresholded_accuracy (np.ndarray): an array containing accuracies of predictions with 
          confidence over a corresponding threshold in the cinfidence_thresholds parameter.
          None if the latter not supplied.
        - thresholded_counts (np.ndarray): an array containing counts of predictions with 
          confidence over a corresponding threshold in the cinfidence_thresholds parameter.
          None if the latter not supplied.
        - binned_entropy_counts (nd.array): histogram values for prediction entropy, corresponding 
          to bin edges supplied as the entropy_bans parameter. None if the latter not supplied. 
        """

        print('\nTesting')
        cum_loss = 0
        total = 0
        correct = 0

        thresholded_counts, thresholded_accuracy, binned_entropy_counts = None, None, None
        if confidence_thresholds is not None:
            thresholded_counts = np.zeros_like(confidence_thresholds)
            thresholded_accuracy = np.zeros_like(confidence_thresholds)
        if entropy_bins is not None:
            binned_entropy_counts = np.zeros(entropy_bins.shape[0] - 1)

        self.model.to(self.device)  
        self.model.eval()

        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as tepoch:
                metric_accumulators = defaultdict(int)
                for X, y in tepoch:
                    
                    X, y = X.to(self.device), y.to(self.device)

                    y_hat = self.predict_test(X)

                    for name, metric in metric_dict.items():
                        metric_val = metric(y_hat, y)
                        # assumes all metrics are mean-reduced
                        metric_accumulators[name] += metric_val * X.size(0)

                    total += X.size(0)

                    _, predicted = torch.max(y_hat, 1)
                    correct += (predicted == y).sum().item()

                    if confidence_thresholds is not None:
                        t_acc, t_count = compute_accuracies_at_confidences(y.cpu().numpy(), y_hat.cpu().numpy(), confidence_thresholds)
                        thresholded_accuracy += np.multiply(t_acc, t_count)
                        thresholded_counts += t_count
                    
                    if entropy_bins is not None:
                        t_entropy = scipy.stats.entropy(y_hat.cpu().numpy(), axis=1)
                        binned_entropy_counts += np.histogram(t_entropy, entropy_bins)[0]
            
            if confidence_thresholds is not None:
                thresholded_accuracy = thresholded_accuracy / thresholded_counts

            test_accuracy = correct/total
            
            print(f'Results: \nAccuracy: {test_accuracy}')
            for name, val in metric_accumulators.items():
                metric_accumulators[name] = val/total
                print(f'{name}: {metric_accumulators[name]}')
                
        return test_accuracy, metric_accumulators, thresholded_accuracy, thresholded_counts, binned_entropy_counts
