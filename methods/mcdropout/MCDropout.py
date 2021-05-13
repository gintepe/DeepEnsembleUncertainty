from methods.BaseTrainer import BaseTrainer
from methods.mcdropout.models import *

import torch.nn as nn
import torch.optim as optim


class MCDropout(BaseTrainer):
    def __init__(self, args, device):
        criterion = nn.CrossEntropyLoss()
        super().__init__(args, criterion, device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,)
        self.n = args.n

    def get_model(self, args):
        model_class = self.get_model_class(args)
        return model_class(dropout_p=0.5)

    def get_model_class(self, args):
        if args.model == 'lenet':
            return LeNet5MCDropout
        if args.model == 'mlp':
            return MLPMCDropout
        if args.model == 'resnet':
            return ResNetMCDropout
        else:
            raise ValueError('invalid network type')

    def predict_val(self, x):
        return self.model(x)

    def predict_test(self, x):
        return self.model.mc_predict(x, self.n)[0]