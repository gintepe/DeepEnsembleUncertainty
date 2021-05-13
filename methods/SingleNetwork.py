from methods.BaseTrainer import BaseTrainer 

import torch
import torch.nn as nn
import torch.optim as optim


class SingleNetwork(BaseTrainer):
    def __init__(self, args, device):
        criterion = nn.CrossEntropyLoss()
        super().__init__(args, criterion, device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,)

    def get_model(self, args):
        model_class = self.get_model_class(args)
        return model_class()

    def predict_val(self, x):
        return self.model(x)

    def predict_test(self, x):
        return torch.nn.functional.softmax(self.model(x), dim=-1)
