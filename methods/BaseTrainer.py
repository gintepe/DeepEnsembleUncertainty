import torch 
import wandb


from abc import abstractmethod
from tqdm import tqdm
from collections import defaultdict

from methods.models import *

class BaseTrainer():
    def __init__(self, args, criterion, device):
        self.device = device
        self.model = self.get_model(args).to(self.device)
        self.criterion = criterion
        # every child class should set this
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def get_model(self, args):
        raise NotImplementedError("Abstract method without implementation provided")

    # TODO the following should probably always be the same 
    @abstractmethod
    def predict_val(self, x):
        raise NotImplementedError("Abstract method without implementation provided")

    @abstractmethod
    def predict_test(self, x):
        """
        Compute prediction for the testing stage.
        Expected output: probabilities 
        """
        raise NotImplementedError("Abstract method without implementation provided")

    # TODO: find a nicer way to incorporate schedulers
    def use_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma = 0.1)
    
    def get_model_class(self, args):
        if args.model == 'lenet':
            return LeNet5
        if args.model == 'mlp':
            return MLP
        if args.model == 'resnet':
            return ResNet
        else:
            raise ValueError('invalid network type')

    def log_info(self, train_acc, val_loss, val_acc, batches, epoch):
        wandb.log({'Training/accuracy': train_acc, 'batch': batches, 'epoch': epoch})
        wandb.log({'Validation/loss': val_loss, 'batch': batches, 'epoch': epoch})
        wandb.log({'Validation/accuracy': val_acc, 'batch': batches, 'epoch': epoch})

    def validate(self, val_loader, val_criterion):    
        
        print('\nValidating')
        
        cum_loss = 0
        total = 0
        correct = 0

        self.model.eval()
        with tqdm(val_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                
                X, y = X.to(self.device), y.to(self.device)

                y_hat = self.predict_val(X)

                loss = val_criterion(y_hat, y)

                loss = loss.item()
                tepoch.set_postfix(loss=loss)

                cum_loss += loss * X.size(0)
                total += X.size(0)

                _, predicted = torch.max(y_hat, 1)
                correct += (predicted == y).sum().item()

        print(f'Validation loss: {cum_loss/total}; accuracy: {correct/total}\n')
                
        return cum_loss / total, correct / total

    def train(self,
            train_loader,
            val_loader,
            epochs,
            log=True,):

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
                    self.optimizer.zero_grad()
                    y_hat = self.model(X)

                    loss = self.criterion(y_hat, y)

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

            val_loss, val_acc = self.validate(val_loader, val_criterion=self.criterion)

            if log:
                self.log_info(correct/total, val_loss, val_acc, batches, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

    def test(self, test_loader, metric_dict):    
        
        print('\nTesting')
        
        cum_loss = 0
        total = 0
        correct = 0

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

            print(f'Results: \nAccuracy: {correct/total}')
            for name, val in metric_accumulators.items():
                metric_accumulators[name] = val/total
                print(f'{name}: {metric_accumulators[name]}')
                
        return correct / total, metric_accumulators
