import torch 
import wandb

from tqdm import tqdm
from collections import defaultdict
# basic training loop for a single network

def test(model, test_loader, metric_dict, device, is_single_output=True):    
    
    print('\nTesting')
    
    cum_loss = 0
    total = 0
    correct = 0

    model.eval()
    with tqdm(test_loader, unit="batch") as tepoch:
        metric_accumulators = defaultdict(int)
        for X, y in tepoch:
            
            X, y = X.to(device), y.to(device)

            if is_single_output:
                y_hat = model(X)
            else:
                # assumes that for multi-output setups overall prediction will be first output
                y_hat = model(X)[0]

            for name, metric in metric_dict.items():
                metric_val = metric(y_hat, y)
                # assumes all metrics are mean-reduced
                metric_dict[name] += metric_val * X.size(0)

            total += X.size(0)

            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == y).sum().item()

    print(f'Results: \nAccuracy: {correct/total}\n')
    for name, val in metric_accumulators:
        metric_accumulators[name] = val/total
        print(f'{name}: {val')
            
    return correct / total, metric_accumulators