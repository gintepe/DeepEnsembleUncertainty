import torch 
import wandb

from tqdm import tqdm
# basic training loop for a single network

def log_info(train_acc, val_loss, val_acc, batches, epoch):
    wandb.log({'Training accuracy': train_acc, 'batch': batches, 'epoch': epoch})
    wandb.log({'Validation loss': val_loss, 'batch': batches, 'epoch': epoch})
    wandb.log({'Validation accuracy': val_acc, 'batch': batches, 'epoch': epoch})

def validate(model, val_loader, criterion, device, is_single_output=True):    
    
    print('\nValidating')
    
    cum_loss = 0
    total = 0
    correct = 0

    model.eval()
    with tqdm(val_loader, unit="batch") as tepoch:
        for X, y in tepoch:
            
            X, y = X.to(device), y.to(device)

            if is_single_output:
                y_hat = model(X)
            else:
                # assumes that for multi-output setups overall prediction will be first output
                y_hat = model(X)[0]

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

    
def train_simple_ensemble(model,
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

        val_loss, val_acc = validate(model, val_loader, basic_cross_entropy, device, is_single_output=False)

        if log:
            log_info(correct/total, val_loss, val_acc, batches, epoch)

def basic_cross_entropy(probs, gt):
    nll = torch.nn.NLLLoss()
    return nll(torch.log(probs), gt)