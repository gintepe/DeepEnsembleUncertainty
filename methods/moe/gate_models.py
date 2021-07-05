import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import wandb
from methods.mcdropout.models import MCDropout, LeNet5MCDropout

MC_SAMPLES=30

class MCDLeNetGate(nn.Module):
    def __init__(self, dropout_p, out_features):
        super().__init__()
        self.gating_network = LeNet5MCDropout(dropout_p, out_features)

    def forward(self, x):
        if self.training:
            return self.gating_network(x)
        else:
            out, preds = self.gating_network.mc_predict(x, n_samples=MC_SAMPLES)
            return out


class MCDC_FC(nn.Module):
    """
    FC layer capable of using MC DropConnect
    Adapted from the tensor flow implementation in https://github.com/hula-ai/mc_dropconnect
    as provided alongside [1]

    References
    ------------
    [1]: Mobiny, Aryan, et al. "Dropconnect is effective in modeling uncertainty of 
         bayesian deep networks." Scientific reports 11.1 (2021): 1-14.
    """
    def __init__(self, in_feat, out_feat, p):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat= out_feat
        self.p = p
        W = torch.empty((in_feat, out_feat), requires_grad=True)
        bias = torch.empty(out_feat, requires_grad=True)

        torch.nn.init.xavier_normal_(W)
        torch.nn.init.normal_(bias)

        self.W = torch.nn.Parameter(W)
        self.bias = torch.nn.Parameter(bias)


    def forward(self, x):

        W = torch.nn.functional.dropout(self.W, self.p, training=True) * self.p
        bias = torch.nn.functional.dropout(self.bias, self.p, training=True) * self.p
        
        x = torch.matmul(x, (W))
        x = x + bias

        return x


class SimpleGate(nn.Module):
    """
    A class implementing a simple MLP with a single hidden layer and ReLU activation.
    Typically to be used as a simplified gating network alternative.
    """
    def __init__(self, in_feat, out_feat):
        """
        Initialises the network with a customised nummber of input and output features
        Parameters
        --------
        - in_feat (int): number of input features.
        - out_feat (int): number of output features.
        """
        super().__init__()
        self.in_feat = in_feat
        self.gating_network = nn.Sequential(
                                    nn.Linear(in_feat, 100), 
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(), 
                                    nn.Linear(100, out_feat)
                                )
        self.gating_network.apply(init_weights)
    
    def forward(self, x):
        """
        Compute gating probability logits for x, typically to be used to
        compute probabilities over individual experts.
        """
        x = x.reshape(x.shape[0], self.in_feat)
        out = self.gating_network(x)
        
        return out

class SimpleMCDropGate(nn.Module):
    """
    A class implementing a simple MLP with a single hidden layer and ReLU activation, using MC Dropout.
    Typically to be used as a simplified gating network alternative.
    """
    def __init__(self, in_feat, out_feat, p):
        """
        Initialises the network with a customised nummber of input and output features
        Parameters
        --------
        - in_feat (int): number of input features.
        - out_feat (int): number of output features.
        """
        super().__init__()
        print(f'mc drop gate with p = {p}')
        self.in_feat = in_feat
        self.gating_network = nn.Sequential(
                                    # MCDropout(p=p),
                                    nn.Linear(in_feat, 100), 
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(), 
                                    #TODO remove the magic from the numbers
                                    MCDropout(p=p),
                                    nn.Linear(100, out_feat)
                                )
        self.gating_network.apply(init_weights)
    
    def forward(self, x):
        """
        Compute gating probability logits for x, typically to be used to
        compute probabilities over individual experts.
        """
        if self.training:

            x = x.reshape(x.shape[0], self.in_feat)
            out = self.gating_network(x)
            
            return out

        else:
            # print('sampling')
            x = x.reshape(x.shape[0], self.in_feat)
            preds = [self.gating_network(x) for i in range(MC_SAMPLES)]
            # print(f'sampled {len(preds)}')
            combined_pred = torch.mean(torch.stack(preds.copy(), dim=0), dim=0)
            # print(combined_pred - preds[10])
            return combined_pred


class SimpleMCDropConnectGate(nn.Module):
    """
    A class implementing a simple MLP with a single hidden layer and ReLU activation, using MC DropConnect.
    Typically to be used as a simplified gating network alternative.
    """
    def __init__(self, in_feat, out_feat, p):
        """
        Initialises the network with a customised nummber of input and output features
        Parameters
        --------
        - in_feat (int): number of input features.
        - out_feat (int): number of output features.
        """
        super().__init__()
        print(f'mc drop-connect gate with p = {p}')
        self.in_feat = in_feat
        self.gating_network = nn.Sequential(
                                    MCDC_FC(in_feat, 100, p), 
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(), 
                                    MCDC_FC(100, out_feat, p)
                                )
        self.gating_network.apply(init_weights)
    
    def forward(self, x):
        """
        Compute gating probability logits for x, typically to be used to
        compute probabilities over individual experts.
        """
        if self.training:

            x = x.reshape(x.shape[0], self.in_feat)
            out = self.gating_network(x)
            
            return out

        else:
            # print('sampling')
            x = x.reshape(x.shape[0], self.in_feat)
            preds = [self.gating_network(x) for i in range(MC_SAMPLES)]
            # print(f'sampled {len(preds)}')
            combined_pred = torch.mean(torch.stack(preds.copy(), dim=0), dim=0)
            # print(combined_pred - preds[10])
            return combined_pred


# could amend this to be "or Conv layer" and have it the same as resnet init
# could also be outsourced to the util file
def init_weights(m):
    """
    Custom weight initialisation for module.
    Parameters
    ----------
    - m (nn.Module): module to be initialised.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        # nn.init.kaiming_normal(m.weight.data)
        # nn.init.zeros_(m.weight.data)
        nn.init.normal_(m.bias.data)

def get_gating_network(network_class, gate_type, data_feat, n, dropout_p=0.1):
    if gate_type == 'same':
        return network_class(num_classes=n)
    if gate_type == 'mcd_simple':
        return SimpleMCDropGate(in_feat=data_feat, out_feat=n, p=dropout_p)
    if gate_type == 'mcdc_simple':
        return SimpleMCDropConnectGate(in_feat=data_feat, out_feat=n, p=dropout_p)
    if gate_type == 'mcd_lenet':
        return MCDLeNetGate(dropout_p=dropout_p, out_features=n)
    else: 
        return SimpleGate(in_feat=data_feat, out_feat=n)