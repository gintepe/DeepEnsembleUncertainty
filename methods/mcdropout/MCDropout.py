from methods.BaseTrainer import BaseTrainer
from methods.mcdropout.models import *

from metrics import basic_cross_entropy

import torch.nn as nn
import torch.optim as optim


class MCDropout(BaseTrainer):
    """ Class for training networks using MC dropout. """
    def __init__(self, args, device):
        """
        Initialise the trainer and network.
        
        Parameters
        --------
        - args (namespace): parsed command line arguments.
        - device (torch.device or str): device to perform calculations on.
        """

        criterion = nn.CrossEntropyLoss()
        super().__init__(args, criterion, device)
        
        self.val_criterion = basic_cross_entropy
        self.n = args.n
    
    def get_model(self, args):
        """
        Implements base class's abstract method.
        Retrieves and intialises a relevant model.
        """
        model_class = self.get_model_class(args)
        return model_class(dropout_p=args.dropout)

    def get_model_class(self, args):
        """
        Overrides the base class's method to use 
        models with dropout
        """
        if args.model == 'lenet':
            return LeNet5MCDropout
        if args.model == 'mlp':
            return MLPMCDropout
        if args.model == 'resnet':
            return ResNetMCDropout
        else:
            raise ValueError('invalid network type')

    def predict_val(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a validation step.
        """
        # return self.model(x)
        return self.model.mc_predict(x, self.n)[0]

    def predict_test(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a testing step.
        """
        return self.model.mc_predict(x, self.n)[0]