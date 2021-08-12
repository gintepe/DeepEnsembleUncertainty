import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import wandb
from methods.mcdropout.models import MCDropout, LeNet5MCDropout, ResNetMCDropout
import scipy

MC_SAMPLES=50

class MCDLeNetGate(nn.Module):
    """
    A class wrapping the LeNet5MCDropout class to be used as a gating network.
    """
    def __init__(self, dropout_p, out_feat):
        super().__init__()
        self.gating_network = LeNet5MCDropout(dropout_p, out_feat)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.training:
            return self.softmax(self.gating_network(x))
        else:
            out, preds = self.gating_network.mc_predict(x, n_samples=MC_SAMPLES)
            return out

class MCDResNetGate(nn.Module):
    """
    A class wrapping the LeNet5MCDropout class to be used as a gating network.
    """
    def __init__(self, dropout_p, out_feat):
        super().__init__()
        self.gating_network = ResNetMCDropout(dropout_p, num_classes=out_feat)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.training:
            return self.softmax(self.gating_network(x))
        else:
            out, preds = self.gating_network.mc_predict(x, n_samples=MC_SAMPLES)
            return out


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
        print('Using a simple gate')
        self.in_feat = in_feat
        self.gating_network = nn.Sequential(
                                    nn.Linear(in_feat, 100), 
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(), 
                                    nn.Linear(100, out_feat),
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

class SimpleConvGate(nn.Module):
    """
    A class implementing a simple convolutional network.
    Typically to be used as a simplified gating network alternative.
    """
    def __init__(self, img_size=28, out_feat=5):
        """
        Initialises the network with a customised nummber of input and output features
        Parameters
        --------
        - in_feat (int): number of input features.
        - out_feat (int): number of output features.
        """
        super().__init__()
        print('Using a simple convolutional gate')
        self.is_mnist = img_size == 28
        self.feature_extractor = nn.Sequential(
                                    nn.Conv2d(1 if self.is_mnist else 3, 16, 3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(16, 32, 3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 32, 3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                )
        if self.is_mnist:
            self.classifier = nn.Sequential(nn.BatchNorm1d(32),
                                        nn.Linear(32, out_feat)
                                        )
        else:
            self.classifier = nn.Sequential(nn.BatchNorm1d(128),
                                        nn.Linear(128, out_feat)
                                        )
                
    
    def forward(self, x):
        """
        Compute gating probability logits for x, typically to be used to
        compute probabilities over individual experts.
        """
        x = self.feature_extractor(x)
        x = x.reshape(x.shape[0], -1)
        out = self.classifier(x)
        
        return out

class SimpleConvMCDGate(nn.Module):
    """
    A class implementing a simple convolutional network with dropout after every layer.
    Typically to be used as a simplified gating network alternative.
    """
    def __init__(self, img_size=28, out_feat=5, p=0.1):
        """
        Initialises the network with a customised nummber of input and output features
        Parameters
        --------
        - in_feat (int): number of input features.
        - out_feat (int): number of output features.
        """
        super().__init__()
        print('Using a simple convolutional gate')
        self.is_mnist = img_size == 28
        self.feature_extractor = nn.Sequential(
                                    nn.Conv2d(1 if self.is_mnist else 3, 16, 3),
                                    MCDropout(p=p),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(16, 32, 3),
                                    MCDropout(p=p),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 32, 3),
                                    MCDropout(p=p),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                )
        if self.is_mnist:
            self.classifier = nn.Sequential(nn.BatchNorm1d(32),
                                        nn.Linear(32, out_feat),
                                        nn.Softmax()
                                        )
        else:
            self.classifier = nn.Sequential(nn.BatchNorm1d(128),
                                        nn.Linear(128, out_feat),
                                        nn.Softmax()
                                        )
    
    def base_forward(self, x):
        x = self.feature_extractor(x)
        x = x.reshape(x.shape[0], -1)
        out = self.classifier(x)
    
        return out

    def mc_forward(self, x):
        preds = [self.base_forward(x) for i in range(MC_SAMPLES)]
        combined_pred = torch.mean(torch.stack(preds.copy(), dim=0), dim=0)
        return combined_pred

    def forward(self, x):
        """
        Compute gating probability logits for x, typically to be used to
        compute probabilities over individual experts.
        """
        if self.training:
            return self.base_forward(x)
        else:
            return self.mc_forward(x)

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
        - p (float): the probability with which dropout should be applied
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
                                    nn.Linear(100, out_feat),
                                    nn.Softmax(dim=-1)
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
            x = x.reshape(x.shape[0], self.in_feat)
            preds = [self.gating_network(x) for i in range(MC_SAMPLES)]
            combined_pred = torch.mean(torch.stack(preds.copy(), dim=0), dim=0)
            return combined_pred

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
        """
        Initialises the layer with a customised nummber of input and output features
        and drop-connect probability.

        Parameters
        --------
        - in_feat (int): number of input features.
        - out_feat (int): number of output features.
        - p (float): the probability with which connections should be dropped
        """

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
        - p (float): the probability with which connections should be dropped
        """
        super().__init__()
        print(f'mc drop-connect gate with p = {p}')
        self.in_feat = in_feat
        self.gating_network = nn.Sequential(
                                    MCDC_FC(in_feat, 100, p), 
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(), 
                                    MCDC_FC(100, out_feat, p),
                                    nn.Softmax(dim=-1)
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


class GateWrapper(nn.Module):
    """
    Wrapper class for some gating network types, ensuring the following:
    * compatability with Laplace approximation as well as the MoE model implementation
      which requires probabilites to be returned from the gating network.
    * possibility to use gating in a conditional fashion based on the output entropy levels

    The current set-up for entropy conditional gating uses top-1 gating for low entropy samples
    and dense gating for high-entropy samples.
    """
    def __init__(self, net, normalise=True, gate_by_entropy=False, entropy_threshold=0.5, is_laplace=False):
        """
        Initialises the wrapper and sets custom behaviours
        
        Parameters
        --------
        - net (torch.nn.Module): network to wrap
        - normalise (bool): whether the outputs of net are logits and should be passed thorugh a softmax
        - gate_by_entropy (bool): whether to use entropy-conditional gating
        - entropy_threshold (float): Threshold to use in entropy conditional gating.
        - is_laplace (bool): whether the gating model has a Laplace approximation applied
        """
        super().__init__()
        self.net = net
        self.is_laplace = is_laplace
        self.gate_by_entropy=gate_by_entropy
        self.entropy_threshold=entropy_threshold
        self.normalise = normalise
        self.softmax = nn.Softmax(dim=-1)
        if self.gate_by_entropy:
            print(f'Gating outputs of entropy of {self.entropy_threshold} replaced by uniform')

    def set_entropy_threshold(self, loader, percentile=95, device='cpu'):
        """
        Use a given dataset to set an entropy threshold for entropy-conditional gating.
        The threshold is set to consider the given percentile of the given dataset as "low entropy".

        Parameters
        ----------
        - loader (torch.utils.data.DataLoader): iterator for the in distribution data
        - precentile (int): between 0 and 100. The proportion of the ID dataset to use to determine the threshold.
        - device (str): cpu or cuda, where to perform the computations
        """

        self.to(device)
        entropies = []
        orig_training_state = self.training
        self.training = True
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = self.forward(x)
            batch_entropies = scipy.stats.entropy(out.cpu().numpy(), axis=1)
            entropies.append(batch_entropies)
        entropies = np.concatenate(entropies).flatten()
        self.entropy_threshold = np.percentile(entropies, percentile)

    
    def forward(self, x):
        
        if self.is_laplace:
            out = self.net(x, link_approx='probit')
        else:
            out = self.net(x)
        
        if self.normalise:
            out = self.softmax(out)

        if self.gate_by_entropy and not self.training:
            # uniform for high entropy, gated for low:
            # entropy = scipy.stats.entropy(out.cpu().numpy(), axis=1)
            # high_entropy_idxs = np.where(entropy > self.entropy_threshold)
            # out[high_entropy_idxs] = torch.ones(out.shape[1]) / out.shape[1]

            # gated for high entropy, top-1 for low
            entropy = scipy.stats.entropy(out.cpu().numpy(), axis=1)
            low_entropy_idxs = np.where(entropy < self.entropy_threshold)
            top_k_vals, top_k_indices = out.topk(1, dim=-1)
            zeros = torch.zeros_like(out)
            topk_weights = zeros.scatter(1, top_k_indices, 1)
            out[low_entropy_idxs] = topk_weights[low_entropy_idxs]

        return out


def get_gating_network(network_class, gate_type, data_feat, n, dropout_p=0.1):
    """
    Selects and initialises an appropriate gating network as specified by the parameters.

    Parameters
    ---------
    - network_class (type): initialisable network class, only used when gate_type is 'same'
    - gate_type (str): the kind of gating network desired. Options are 'same' (same as supplied network_class), 
      'mcd_simple' (MLP with 1 hidden layer and MC Dropout), 'mcdc_simple' (MLP with 1 hidden layer and MC Drop-Connect),
      'mcd_lenet' (LeNet5 with MC Dropout), 'conv' (small 3-convolutional-layer network), 'simple' (default, MLP with 1 hidden layer).
    - data-feat (int): number of features of the datapoints
    - n (int): number of experts to be gated over
    - dropout_p (float): for methods using dropout, its probability

    Returns
    ---------
    - gating_network (torch.nn.Module): initialised gating network
    """
    if gate_type == 'same':
        return GateWrapper(network_class(num_classes=n))
    if gate_type == 'mcd_simple':
        return SimpleMCDropGate(in_feat=data_feat, out_feat=n, p=dropout_p)
    if gate_type == 'mcdc_simple':
        return SimpleMCDropConnectGate(in_feat=data_feat, out_feat=n, p=dropout_p)
    if gate_type == 'mcd_lenet':
        return MCDLeNetGate(dropout_p=dropout_p, out_feat=n)
    if gate_type == 'mcd_resnet':
        return MCDResNetGate(dropout_p=dropout_p, out_feat=n)
    if gate_type == 'mcd_conv':
        return SimpleConvMCDGate(img_size = int(np.sqrt(data_feat)), out_feat=n, p=dropout_p)
    if gate_type == 'conv':
        return GateWrapper(SimpleConvGate(img_size = int(np.sqrt(data_feat)), out_feat=n))
    else: 
        return GateWrapper(SimpleGate(in_feat=data_feat, out_feat=n))