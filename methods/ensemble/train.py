import torch 
import wandb

from tqdm import tqdm
from methods.general_loops import *
from metrics import basic_cross_entropy

#TODO: parameter naming convention!
def train(model,
         train_loader,
         val_loader,
         criterion,
         optimizers,
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
                [optimizer.zero_grad() for optimizer in optimizers]
                
                pred, y_hats = model(X)

                losses = [criterion(y_hat, y) for y_hat in y_hats]

                loss = 0
                for i in range(len(losses)):
                    losses[i].backward()
                    optimizers[i].step()

                    loss += losses[i].item()

                tepoch.set_postfix(loss=loss/len(losses))

                batches += 1

                _, predicted = torch.max(pred, 1)
                correct += (predicted == y).sum().item()
                total += X.shape[0]

                if log:
                    wandb.log({'(Ensemble) Mean Training loss': loss/len(losses), 'batch': batches})

        val_loss, val_acc = validate(model, val_loader, basic_cross_entropy, device, pred=lambda m, x: m(x)[0])

        if log:
            log_info(correct/total, val_loss, val_acc, batches, epoch)
