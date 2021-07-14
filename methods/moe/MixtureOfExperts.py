from methods.BaseTrainer import BaseTrainer
from methods.moe.models import *
from metrics import basic_cross_entropy

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# TODO not all implemented models are currently compatible with the train function

# TODO choices to be made: whether to average outputs or probabilities if using a fully end-to-end setup
# for now doing a simple one with averaged probs
class SimpleMoE(BaseTrainer):
    """
    Class implementing training and optimization for the mixture of experts paradigm 
    """
    def __init__(self, args, device):
        """
        Initialise the trainer and network.
        
        Parameters
        --------
        - args (namespace): parsed command line arguments.
        - device (torch.device or str): device to perform calculations on.
        """

        criterion = basic_cross_entropy
        self.n = args.n
        self.gated_predict = args.predict_gated
        self.data_features = 32*32*3 if 'cifar' in args.dataset_type else 28*28
        super().__init__(args, criterion, device)
        self.val_criterion = basic_cross_entropy
        self.load_loss_coeff = args.reg_weight

    def get_model(self, args):
        """ Retrieves and initialises the relevant model as specified by args. """
        model_class = self.get_model_class(args)
        
        if args.moe_type == 'fixed':
            moe_class = DenseFixedMoE
        elif args.moe_type == 'sparse':
            moe_class = SparseMoE
        else:
            moe_class = DenseBasicMoE

        if args.dataset_type == 'cifar100':
            return moe_class(model_class, gate_type=args.moe_gating, data_feat=self.data_features, 
                             n=self.n, k=args.moe_topk, dropout_p=args.dropout, num_classes=100)
        else:
            return moe_class(model_class, gate_type=args.moe_gating, data_feat=self.data_features,
                             n=self.n, k=args.moe_topk, dropout_p=args.dropout)
    
    def predict_val(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a validation step.

        Depending on the specification provided when initialising the model,
        can adjust the prediction to use a non-gated, traditional ensemble approach.
        """
        combined_pred, preds = self.model.forward_dense(x)
        if self.gated_predict:
            return combined_pred, preds
        else:
            return torch.mean(nn.functional.softmax(torch.stack(preds, dim=0), dim=-1), dim=0), preds
    
    def predict_test(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a testing step.
        """
        combined_pred, preds = self.model.forward_dense(x)
        if self.gated_predict:
            return combined_pred, preds
        else:
            return torch.mean(nn.functional.softmax(torch.stack(preds, dim=0), dim=-1), dim=0), preds

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

        loads = np.zeros(self.n)
        loads_by_label = list([np.zeros(self.n) for i in range(10)])

        self.model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                
                # move data to relevant device
                X, y = X.to(self.device), y.to(self.device)

                # compute loss        
                y_hat, preds, batch_loads, batch_loads_by_label, load_loss = self.model(X, labels=y, loss_coef=self.load_loss_coeff)
                loss = self.criterion(y_hat, y) + load_loss
                
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

                loads += batch_loads

                if len(batch_loads_by_label) > 0:
                    for i in range(10):
                        loads_by_label[i] += batch_loads_by_label[i]
                if log:
                    wandb.log({'Training/loss': loss, 'batch': batches})
        
        if log:
            wandb.log({"loads": wandb.Histogram(np_histogram=(loads, np.linspace(0, self.n, self.n + 1)))})
        
        if sum([np.sum(load_counts) for load_counts in loads_by_label]) > 0:
            for i in range(10):
                if log:
                    wandb.log({f"loads class {i}": wandb.Histogram(np_histogram=(loads_by_label[i], np.linspace(0, self.n, self.n + 1)))})
                print(f'Loads for the epoch label {i}: {loads_by_label[i]}')
        
        print(f'Loads for the epoch: {loads}')

        return correct, total, batches    