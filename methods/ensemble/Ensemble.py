from methods.BaseTrainer import BaseTrainer
from methods.ensemble.models import *
from metrics import basic_cross_entropy

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

class BaseEnsemble(BaseTrainer):
    """
    Base class for deep ensembles and their variations
    """
    def __init__(self, args, criterion, device):
        """
        Initialise the trainer and network.
        
        Parameters
        --------
        - args (namespace): parsed command line arguments.
        - criterion (function mx, x, y -> List[scalar]): enssemble training criterion to optimize
        - device (torch.device or str): device to perform calculations on.
        """
        
        super().__init__(args, criterion, device)
        
    def get_optimizer(self, args):
        """
        Overrides the base class's implementation to retrieve
        a list of optimizers rather than a single one.
        """
        if args.optimizer == 'adam':
            return [optim.Adam(m.parameters(), lr=args.lr,) for m in self.model.networks]
        else:
            print('SGD optimizer')
            return list([optim.SGD(m.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9) for m in self.model.networks])

    def get_model(self, args):
        """
        Implements base class's abstract method.
        Retrieves and intialises an ensemble of relevant models.
        """
        model_class = self.get_model_class(args)
        if args.dataset_type == 'cifar100':
            return SimpleEnsemble(model_class, n=self.n, num_classes=100)
        else:
            return SimpleEnsemble(model_class, n=self.n)

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
        return self.model(x)#c[0]

    def get_schedulers(self, args):
        """
        Overrides the base class's implementation to retrieve
        a list of schedulers rather than a single one.
        """
        if args.scheduler is None:
            return []
        
        schedulers = []
        for i in range(self.n):
            if args.scheduler == 'step':
                sched = optim.lr_scheduler.StepLR(self.optimizer[i], args.scheduler_step, gamma = args.scheduler_rate)
            elif args.scheduler == 'exp':
                sched = optim.lr_scheduler.ExponentialLR(self.optimizer[i], args.scheduler_rate) 
            else:
                sched = torch.optim.lr_scheduler.MultiStepLR(self.optimizer[i], milestones=[90, 135], gamma=args.scheduler_rate)
            schedulers.append(sched)
        return schedulers

    def train(self,
         train_loader,
         batches,
         log=True,):
        """
        Overrides the base class's implementation to provide an ensemble-specific training step.

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

        self.model.train()
        correct = 0
        total = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                
                X, y = X.to(self.device), y.to(self.device)
                
                pred, y_hats = self.model(X)

                losses = self.criterion(pred.detach(), y_hats, y)

                loss = 0
                for i in range(len(losses)):
                    self.optimizer[i].zero_grad()
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

        return correct, total, batches


class Ensemble(BaseEnsemble):
    """ Class for training simple deep ensembles. """

    def __init__(self, args, device):
        """
        Initialise the trainer and network.
        
        Parameters
        --------
        - args (namespace): parsed command line arguments.
        - device (torch.device or str): device to perform calculations on.
        """

        print(f'Initialising an ensemble of {args.n} networks')
        # criterion = nn.CrossEntropyLoss()
        criterion = self.ensemble_cn
        self.n = args.n
        super().__init__(args, criterion, device)
        self.val_criterion = basic_cross_entropy

    def ensemble_cn(self, mean_probs, pred_logits, ground_truth):
        """
        Cross-entropy wrapper for ensembles, for compatability with the BaseEnsemble training step. 
        Applies the cross-entropy loss to each subnetwork individually and returns the values as a list.

        Parameters
        -------
        - maen_probs (torch.Tensor): the mean of the converted to probabilities predictions.
        - pred_logits (List[torch.Tensor]): network output for each ensemble member.
        - ground_truth (torch.Tensor): ground truth labels, regular int encoding.

        Returns
        -------
        - losses (List[torch.Tensor]): loss values for each network.
        """
        losses = [torch.nn.functional.cross_entropy(pred, ground_truth) for pred in pred_logits]
        return losses


class NCEnsemble(BaseEnsemble):
    """
    Class for training negative correlation regularised [1] deep ensembles.

    References
    --------
    [1] : Shui, Changjian, et al. "Diversity regularization in deep ensembles." 
          arXiv preprint arXiv:1802.07881 (2018).
    """
    def __init__(self, args, device):
        """
        Initialise the trainer and network.
        
        Parameters
        --------
        - args (namespace): parsed command line arguments.
        - device (torch.device or str): device to perform calculations on.
        """

        print(f'Initialising a negative-correlation normalised ensemble of {args.n} networks')
        
        self.n = args.n
        self.l = args.reg_weight
        self.min_l = args.reg_min
        self.decay = args.reg_decay
        
        criterion = self.nc_joint_regularised_cross_entropy
        super().__init__(args, criterion, device)
        
        self.val_criterion = basic_cross_entropy

    def train(self,
         train_loader,
         batches,
         log=True,):
        """
        Expands the base class's implementation to utilise regularisation scaling parameter updates.

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

        correct, total, batches = super().train(train_loader, batches, log)

        if self.l > self.min_l:
            self.l = self.l * self.decay

        return correct, total, batches
    
    def nc_joint_regularised_cross_entropy(self, mean_probs, pred_logits, ground_truth):
        """
        Computes a list of negative correlation regularised (as per [1]) losses.

        Parameters
        -------
        - maen_probs (torch.Tensor): the mean of the converted to probabilities predictions.
        - pred_logits (List[torch.Tensor]): network output for each ensemble member.
        - ground_truth (torch.Tensor): ground truth labels, regular int encoding.

        Returns
        -------
        - losses (List[torch.Tensor]): loss values for each network.

        References
        -------
        [1] : Shui, Changjian, et al. "Diversity regularization in deep ensembles." 
            arXiv preprint arXiv:1802.07881 (2018).
        """

        prob_predictions = nn.functional.softmax(torch.stack(pred_logits.copy(), dim=1).detach(), dim=-1)
        sum_factor = (1 - torch.eye(len(pred_logits))).to(mean_probs.device).detach()
        sums = torch.matmul(sum_factor, (prob_predictions - mean_probs.unsqueeze(1)).detach())

        # ======= additional logging =======
        cn_acc, reg_acc = 0, 0
        # ======= additional logging =======

        losses = []
        for i, pred in enumerate(pred_logits):
            cn = torch.nn.functional.cross_entropy(pred, ground_truth)

            reg = torch.mean(torch.sum((nn.functional.softmax(pred, dim=-1) - mean_probs) * sums[:, i, :], dim=-1))
            # reg = torch.mean(torch.sum((nn.functional.softmax(pred, dim=-1) - mean_probs) * (mean_probs - nn.functional.softmax(pred, dim=-1)), dim=-1))

            losses.append(cn + self.l*reg)
            
        # ======= additional logging =======S
            cn_acc += cn.item()
            reg_acc += self.l*reg.item()
        
        wandb.log({'Training/crossentropy': cn_acc/len(pred_logits)})
        wandb.log({'Training/regularizer': reg_acc/len(pred_logits)})
        # ======= additional logging =======

        return losses


class CEEnsemble(BaseEnsemble):
    """
    Class for training pairwise cross-entropy regularised deep ensembles. 
    The loss function is used is partially adapted from [2], with terms decoupled
    and used in a traditional, rather than efficient, ensemble setting.

    References
    --------
    [2] : Opitz, Michael, Horst Possegger, and Horst Bischof. "Efficient model averaging for deep neural networks." 
            Asian Conference on Computer Vision. Springer, Cham, 2016.
    """
    def __init__(self, args, device):
        """
        Initialise the trainer and network.
        
        Parameters
        --------
        - args (namespace): parsed command line arguments.
        - device (torch.device or str): device to perform calculations on.
        """

        print(f'Initialising a pairwise cross-entropy regularised ensemble of {args.n} networks')
        
        self.n = args.n
        self.l = args.reg_weight
        self.min_l = args.reg_min
        self.decay = args.reg_decay
        
        criterion = self.ce_joint_regularised_cross_entropy
        super().__init__(args, criterion, device)
        
        self.val_criterion = basic_cross_entropy

    def train(self,
         train_loader,
         batches,
         log=True,):
        """
        Expands the base class's implementation to utilise regularisation scaling parameter updates.

        Parameters
        -------
        - train_loader (torch.utils.data.DataLoader): iterator for the training data.
        - batches (int): number of batches seen so far.

        Returns
        -------
        - correct (int): number of correct predictions observed.
        - total (int): number of datapoints observed.
        - batches (int): updated count of batches observed.
        """

        correct, total, batches = super().train(train_loader, batches, log)

        if self.l > self.min_l:
            self.l = self.l * self.decay

        return correct, total, batches

    def ce_joint_regularised_cross_entropy(self, mean_probs, pred_logits, ground_truth):
        """
        Computes a list of pairwise cross-entropy regularised (as per [2]) losses.
        The version is disentangled, to compute regularisation for each ensemble predictor individually,
        with the term comprised of sum of cross entropy values for the network in question, with each other network,
        computed as an average per-pair.

        Parameters
        -------
        - maen_probs (torch.Tensor): the mean of the converted to probabilities predictions.
        - pred_logits (List[torch.Tensor]): network output for each ensemble member.
        - ground_truth (torch.Tensor): ground truth labels, regular int encoding.

        Returns
        -------
        - losses (List[torch.Tensor]): loss values for each network.

        References
        -------
        [2] : Opitz, Michael, Horst Possegger, and Horst Bischof. "Efficient model averaging for deep neural networks." 
            Asian Conference on Computer Vision. Springer, Cham, 2016.
        """

        # prob_predictions = nn.functional.softmax(torch.stack(pred_logits.copy(), dim=1), dim=-1)

        # ======= additional logging =======
        cn_acc, reg_acc = 0, 0
        # ======= additional logging =======

        losses = []
        for i, pred in enumerate(pred_logits):
            cn = torch.nn.functional.cross_entropy(pred, ground_truth)

            reg = 0
            for j, pred_extra in enumerate(pred_logits):
                if not i == j:
                    # reg += torch.sum((nn.functional.softmax(pred, dim=-1)) * nn.functional.log_softmax(pred_extra, dim=-1).detach() )/(len(pred_logits)*len(pred_logits)-1)
                    reg += torch.mean(torch.sum((nn.functional.softmax(pred, dim=-1)) * nn.functional.log_softmax(pred_extra, dim=-1).detach(), dim=-1))/(len(pred_logits)*len(pred_logits)-1)
            
            losses.append(cn + self.l*reg)
            
        # ======= additional logging =======S
            cn_acc += cn.item()
            reg_acc += self.l*reg.item()

        wandb.log({'Training/crossentropy': cn_acc/len(pred_logits)})
        wandb.log({'Training/regularizer': reg_acc/len(pred_logits)})
        # ======= additional logging =======

        return losses