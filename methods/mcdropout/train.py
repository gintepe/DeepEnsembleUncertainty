import methods.general_loops

def train(model,
         train_loader,
         val_loader,
         criterion,
         optimizer,
         epochs,
         log=True,
         device='cpu'):
    return methods.general_loops.train(model,
         train_loader,
         val_loader,
         criterion,
         optimizer,
         epochs,
         log=log,
         device=device)