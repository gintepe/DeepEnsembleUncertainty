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
from metrics import compute_accuracies_at_confidences, disagreement_and_correctness, bin_predictions_and_accuracies_multiclass

#TODO this is super non regression-friendly
# maybe it would be a good idea to adjust somehow to make it more flexible 

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
            opt = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        return opt
    
    def get_schedulers(self, args):
        """
        Initialise a learning rate scheduler as specified by args.
        """

        scheduler = self.get_scheduler(self.optimizer, args.scheduler, args.scheduler_step, args.scheduler_rate)

        if scheduler is not None:
            return [scheduler]
        else:
            return []

    def get_scheduler(self, optimizer, sched_type, step, rate):
        if sched_type == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step, gamma = rate)
            # return [torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[90, 135], gamma=args.scheduler_rate)]
        elif sched_type == 'exp':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, rate) 
        elif sched_type == 'multistep':
            print('using multistep scheduler')
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=rate)
        elif sched_type == 'multistep-ext':
            print('using extended multistep scheduler')
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180], gamma=rate)
        elif sched_type == 'multistep-adam':
            print('using extended, paper copied, multistep scheduler')
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160, 180], gamma=rate)
        else:
            return None


    # TODO the following should probably always be the same 
    @abstractmethod
    def predict_val(self, x):
        """ 
        Method to retrieve predictions for input x during a validation step
        Should always return (overall prediction, individual predictions) with the latter
        None if a method does not use ensembling 
        """
        raise NotImplementedError("Abstract method without implementation provided")

    @abstractmethod
    def predict_test(self, x):
        """
        Compute prediction for the testing stage.
        Expected output: probabilities 
        Should always return (overall prediction, individual predictions) with the latter
        None if a method does not use ensembling
        """
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
    def log_info(self, train_acc, val_loss, val_acc, val_conf, val_avg_acc, val_avg_dis, batches, epoch):
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
        if val_avg_acc is not None:
            wandb.log({'Validation/subnet_accuracy': val_avg_acc, 'batch': batches, 'epoch': epoch})
            wandb.log({'Validation/pairwise_disagreement': val_avg_dis, 'batch': batches, 'epoch': epoch})

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
        avg_correct = 0
        disagreed = 0

        self.model.eval()
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for X, y in tepoch:
                    
                    X, y = X.to(self.device), y.to(self.device)

                    y_hat, preds = self.predict_val(X)

                    loss = self.val_criterion(y_hat, y)

                    loss = loss.item()
                    tepoch.set_postfix(loss=loss)

                    cum_loss += loss * X.size(0)
                    total += X.size(0)

                    confidence, predicted = torch.max(y_hat, 1)
                    correct += (predicted == y).sum().item()
                    cum_conf += confidence.sum().item()

                    dis, avgc, _, _ = disagreement_and_correctness(preds, y)
                    avg_correct += avgc
                    disagreed += dis

            validation_loss = cum_loss/total
            validation_accuracy = correct/total
            validation_confidence = cum_conf/total

            validation_subnet_accuracy = None if preds is None else avg_correct/total
            validation_avg_disagreement = None if preds is None else disagreed/total

            print(f'Validation loss: {validation_loss}; accuracy: {validation_accuracy}\n')
                
        return validation_loss, validation_accuracy, validation_confidence, validation_subnet_accuracy, validation_avg_disagreement

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
            early_stop_threshold=None,
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

        # we will call this large enough
        min_val_loss = 1e7
        no_val_improvement = 0

        self.model.to(self.device)
        
        # if log:
        #     wandb.watch(self.model)
        
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}')
            
            correct, total, batches = self.train(train_loader, batches, log)
            val_loss, val_acc, val_conf, val_avg_acc, val_dis = self.validate(val_loader)

            # logic for early stopping
            if early_stop_threshold is not None:
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    no_val_improvement = 0
                else:
                    no_val_improvement += 1

                if no_val_improvement > early_stop_threshold:
                    break

            if log:
                self.log_info(correct/total, val_loss, val_acc, val_conf, val_avg_acc, val_dis, batches, epoch)

            for sched in self.schedulers:
                sched.step()

        if self.checkpoint_dir is not None:
            self.save_checkpoint(epoch, val_loss)

    def test(
          self, 
          test_loader, 
          metric_dict, 
          confidence_thresholds=None, 
          entropy_bins=None, 
          track_full_disagreements=False,
          calibration_hist_bins=None,
          ):  
        """
        Testing/evaluation logic.
        Returns and computes some extra information to allow for further data visualisaton.
        For only customisable behaviour, discard the final 3 returned items and last 2 params.
        Parameters 3 and further are primarily used for checkpoint evaluation and plotting.

        Parameters
        -------
        - test_loader (torch.utils.data.DataLoader): iterator for the test data.
        - metric_dict (dictionary {name: function (prob, gt) -> float}): metrics to be evaluated 
          at each testing step. Must be possible to aggregate via mean.
        - confidence_thresholds (np.ndarray): if thresholded accuracies and counts are needed, this 
          parameter should contain a list of increasing thresholds in the range [0, 1].
        - entropy_bins (np.ndarray): if binned entrypy counts are needed, this parameter should 
          contain a list of bin boundaries in a desired range.
        - track_full_disagreements (bool): wether to compute and track a full disagreement matrix
        - calibration_hist_bins (int):

        Returns
        -------
        - test_accuracy (float): Accuracy of the predictions accross the testing set.
        - metric_accumulators (dictionary {name: float}): mean values of the metrics given.
          For ensembling based methods/ones making multiple predictions, average pair disagreement
          and average component accuracy are added to the required metrics returned here. For all 
          models, an everage confidence metric is added.
        - thresholded_accuracy (np.ndarray): an array containing accuracies of predictions with 
          confidence over a corresponding threshold in the confidence_thresholds parameter.
          None if the latter not supplied.
        - thresholded_counts (np.ndarray): an array containing counts of predictions with 
          confidence over a corresponding threshold in the confidence_thresholds parameter.
          None if the latter not supplied.
        - binned_entropy_counts (nd.array): histogram values for prediction entropy, corresponding 
          to bin edges supplied as the entropy_bans parameter. None if the latter not supplied. 
        - disagreement_mat (nd.array): For ensemble methods, an n by n matrix containing the
          percentage of samples given pairs of classifiers disagree on. 
        - calibration_hist (Tuple(np.ndarray)): calibration histogram bins and average accuracy values
        """

        print('\nTesting')

        stat_tracker = StatisticsTracker(self.n, confidence_thresholds, entropy_bins, track_full_disagreements, calibration_hist_bins)

        self.model.to(self.device)  
        self.model.eval()

        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as tepoch:
                metric_accumulators = defaultdict(int)
                for X, y in tepoch:
                    
                    X, y = X.to(self.device), y.to(self.device)

                    y_hat, preds = self.predict_test(X)

                    for name, metric in metric_dict.items():
                        metric_val = metric(y_hat, y)
                        # assumes all metrics are mean-reduced
                        metric_accumulators[name] += metric_val * X.size(0)

                    stat_tracker.update(y_hat, preds, y)

            correct = stat_tracker.correct
            total = stat_tracker.total

            test_accuracy = correct/total
            print(f'Results: \nAccuracy: {test_accuracy}')
            for name, val in metric_accumulators.items():
                metric_accumulators[name] = val/total
                print(f'{name}: {metric_accumulators[name]}')

        return test_accuracy, metric_accumulators, stat_tracker

class StatisticsTracker():
    def __init__(self, n, 
        confidence_thresholds=None, 
        entropy_bins=None, 
        track_full_disagreements=False,
        calibration_hist_bins=None,
        ):

        self.is_multi_pred = False

        self.track_calibration_histogram = calibration_hist_bins is not None
        self.init_calibration_hist(calibration_hist_bins)
        
        self.track_thresholded_confidence = confidence_thresholds is not None
        self.init_counts_acc_by_confidence(confidence_thresholds)

        self.track_binned_entropy = entropy_bins is not None
        self.init_binned_entropies(entropy_bins)

        self.track_full_disagreements = track_full_disagreements
        if self.track_full_disagreements:
            self.disagreement_mat = np.zeros((n, n))
            
        self.total = 0
        self.correct = 0

        self.avg_corr = 0
        self.cum_conf = 0
        self.disagreements = 0
        self.subnet_correct = np.zeros(n)

    def update(self, y_hat, preds, y):
        self.is_multi_pred = preds is not None

        self.total += y.shape[0]
        
        confidence, predicted = torch.max(y_hat, 1)
        self.correct += (predicted == y).sum().item()
        self.cum_conf += confidence.sum().item()

        dis, avgc, dis_mat, subc = disagreement_and_correctness(preds, y)
        self.disagreements += dis
        self.avg_corr += avgc
        self.subnet_correct += subc
        if self.track_full_disagreements:
            self.disagreement_mat += dis_mat

        self.update_binned_entropies(y_hat)
        self.update_calibration_hist(y_hat, y)
        self.update_counts_acc_by_confidence(y_hat, y)

    def log_statistics(self, prefix, shift, shift_name='shift'):
        wandb.log({f'{prefix} confidence': self.cum_conf / self.total, shift_name: shift})
    
        if self.is_multi_pred:
            wandb.log({f'{prefix} disagreement': self.disagreements / self.total, shift_name: shift})
            wandb.log({f'{prefix} component accuracy': self.avg_corr / self.total, shift_name: shift})
            for i in range(min(self.subnet_correct.shape[0], 5)):
                wandb.log({f'{prefix} subnet {i} accuracy': self.subnet_correct[i] / self.total, shift_name: shift})

    def get_disagreement_mat(self):
        return self.disagreement_mat / self.total

    def init_binned_entropies(self, entropy_bins):
        if self.track_binned_entropy:
            self.entropy_bins = entropy_bins
            self.binned_entropy_counts = np.zeros(entropy_bins.shape[0] - 1)

    def update_binned_entropies(self, y):
        if self.track_binned_entropy:
            t_entropy = scipy.stats.entropy(y.cpu().numpy(), axis=1)
            self.binned_entropy_counts += np.histogram(t_entropy, self.entropy_bins)[0]

    def get_binned_entropies(self):
        if self.track_binned_entropy:
            return self.binned_entropy_counts

    def init_calibration_hist(self, n_bins):
        if self.track_calibration_histogram:
            self.calibration_hist_bins = n_bins
            self.calibration_hist_vals = np.zeros(n_bins)
            self.calibration_hist_counts = np.zeros(n_bins)
            self.calibration_bins = None

    def update_calibration_hist(self, y_hat, y):
        if self.track_calibration_histogram:
            self.calibration_bins, accuracies, counts = bin_predictions_and_accuracies_multiclass(y_hat.cpu().numpy(), y.cpu().numpy(), self.calibration_hist_bins)
            self.calibration_hist_vals += np.multiply(accuracies, counts)
            self.calibration_hist_counts += counts

    def get_calibration_hist(self):
        if self.track_calibration_histogram:
            non_zero_counts = np.where(self.calibration_hist_counts > 0)
            calibration_hist_vals = self.calibration_hist_vals.copy()
            calibration_hist_vals[non_zero_counts] = self.calibration_hist_vals[non_zero_counts] / self.calibration_hist_counts[non_zero_counts]
            return (calibration_hist_vals, self.calibration_bins)

    def init_counts_acc_by_confidence(self, confidence_thresholds):
        if self.track_thresholded_confidence:
            self.confidence_thresholds = confidence_thresholds
            self.thresholded_counts = np.zeros_like(confidence_thresholds)
            self.thresholded_accuracy = np.zeros_like(confidence_thresholds)
    
    def update_counts_acc_by_confidence(self, y_hat, y):
        if self.track_thresholded_confidence:
            t_acc, t_count = compute_accuracies_at_confidences(y.cpu().numpy(), y_hat.cpu().numpy(), self.confidence_thresholds)
            self.thresholded_accuracy += np.multiply(t_acc, t_count)
            self.thresholded_counts += t_count

    def get_counts_acc_by_confidence(self):
        if self.track_thresholded_confidence:
            thresholded_accuracy = self.thresholded_accuracy / self.thresholded_counts
            return thresholded_accuracy, self.thresholded_counts