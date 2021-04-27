import torch 
import wandb

from tqdm import tqdm
# basic training loop for a single network

def validate(model, val_loader, criterion, device):    
    cum_loss = 0
    els = 0
    correct = 0
    
    with tqdm(val_loader, unit="batch") as tepoch:
        model.eval()
        for X, y in tepoch:
            
            X, y = X.to(device), y.to(device)

            y_hat = model(X)

            loss = criterion(y_hat, y)

            loss = loss.item()
            tepoch.set_postfix(loss=loss)

            cum_loss += loss * X.size(0)
            els += X.size(0)

            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == y).sum().item()
            
    return cum_loss / els, correct / els

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

                if log:
                    wandb.log({'train_loss': loss, 'batch': batches})

        print('Validating')
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Validation loss: {val_loss}')

        if log:
            wandb.log({'train_accuracy': correct/len(train_loader.dataset), 'batch': batches, 'epoch': epoch})
            wandb.log({'val_loss': val_loss, 'batch': batches, 'epoch': epoch})
            wandb.log({'val_accuracy': val_acc, 'batch': batches, 'epoch': epoch})



