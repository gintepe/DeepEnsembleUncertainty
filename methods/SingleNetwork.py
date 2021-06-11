from methods.BaseTrainer import BaseTrainer 

import torch
import torch.nn as nn
import torch.optim as optim

from metrics import basic_cross_entropy


class SingleNetwork(BaseTrainer):
    """ Class for training simple traditional network implementations. """
    
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

    def get_model(self, args):
        """
        Implements base class's abstract method.
        Retrieves and intialises a relevant model.
        """
        model_class = self.get_model_class(args)

        #TODO: maybe model classes should just take args for intialisation??
        if args.dataset_type == 'cifar100':
            return model_class(num_classes=100)
        else:
            return model_class()

    def predict_val(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a validation step.
        """
        # return self.model(x)
        return torch.nn.functional.softmax(self.model(x), dim=-1), None

    def predict_test(self, x):
        """
        Implements base class's abstract method.
        Predict for x during a testing step.
        """
        return torch.nn.functional.softmax(self.model(x), dim=-1), None
