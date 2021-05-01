import torch 
import wandb

from tqdm import tqdm
from collections import defaultdict
# basic training loop for a single network

def log_info(train_acc, val_loss, val_acc, batches, epoch):
    wandb.log({'Training accuracy': train_acc, 'batch': batches, 'epoch': epoch})
    wandb.log({'Validation loss': val_loss, 'batch': batches, 'epoch': epoch})
    wandb.log({'Validation accuracy': val_acc, 'batch': batches, 'epoch': epoch})

def validate(model, val_loader, criterion, device, pred=lambda m, x: m(x)):    
    
    print('\nValidating')
    
    cum_loss = 0
    total = 0
    correct = 0

    model.eval()
    with tqdm(val_loader, unit="batch") as tepoch:
        for X, y in tepoch:
            
            X, y = X.to(device), y.to(device)

            y_hat = pred(model, X)

            loss = criterion(y_hat, y)

            loss = loss.item()
            tepoch.set_postfix(loss=loss)

            cum_loss += loss * X.size(0)
            total += X.size(0)

            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == y).sum().item()

    print(f'Validation loss: {cum_loss/total}; accuracy: {correct/total}\n')
            
    return cum_loss / total, correct / total

def train(model,
         train_loader,
         val_loader,
         criterion,
         optimizer,
         epochs,
         log=True,
         device='cpu'):

    batches = 0
    
    if log:
        wandb.watch(model)
    
    for epoch in range(1, epochs + 1):
        model.train()

        print(f'Epoch {epoch}')
        correct = 0
        total = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for X, y in tepoch:
                
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = model(X)

                loss = criterion(y_hat, y)

                loss.backward()
                optimizer.step()

                loss = loss.item()
                tepoch.set_postfix(loss=loss)

                batches += 1

                _, predicted = torch.max(y_hat, 1)
                correct += (predicted == y).sum().item()
                total += X.shape[0]

                if log:
                    wandb.log({'Training loss': loss, 'batch': batches})

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if log:
            log_info(correct/total, val_loss, val_acc, batches, epoch)

def test(model, test_loader, metric_dict, device, is_single_output=True, pred=lambda m, x: m(x)):    
    
    print('\nTesting')
    
    cum_loss = 0
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            metric_accumulators = defaultdict(int)
            for X, y in tepoch:
                
                X, y = X.to(device), y.to(device)

                if is_single_output:
                    y_hat = torch.nn.functional.softmax(pred(model, X), dim=-1)
                else:
                    y_hat = pred(model, X)

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
