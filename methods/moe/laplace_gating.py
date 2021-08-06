from laplace import Laplace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from time import time

from methods.moe.gate_models import GateWrapper

from methods.moe.models import SparseDispatcher

class DenseLaplaceWrapper(nn.Module):
    """
    Class implementing a naiive approach for mixture of experts, with forward passes performed on all expert networks
    and gating applied afterwards. 
    """
    def __init__(self, original_model, la_gating, gate_by_entropy=False, entropy_threshold=0.3):
        """
        Initialise the model.

        Parameters
        ----------
        - original_model (nn.Module): full original MoE model
        - la_gating (Laplace.BaseLaplace): gating network with the Laplace Approximation applied
        - gate_by_entropy (bool): whether to use entropy-conditional gating
        - entropy_threshold (float): only relevant if the parmeter above is set to True. Threshold
          to use in entropy conditional gating.  

        """
        super().__init__()
        self.experts = original_model.experts
        self.gating_network = GateWrapper(
                                la_gating, 
                                normalise=False, 
                                gate_by_entropy=gate_by_entropy, 
                                entropy_threshold=entropy_threshold, 
                                is_laplace=True
                            )

    def forward(self, x, labels=None):
        """
        Compute combined and individual predictions for x.
        
        Parameters
        ---------
        - x (torch.Tensor): input data
        - labels (torch.Tensor): ground truth labels, for compatability with other approaches.

        Returns
        ---------
        - combined_pred (torch.Tensor): combined predixtion of the mixture of experts.
        - preds (list[torch.Tensor]): predictions of the individual experts.
        - part_sizes (np.ndarray): number of samples in batch x each individual expert 
          is assigned a non-zero weight for.
        - part_sizes_by_label (list[np.ndarray]): for compatability. 
          Number of samples in batch x each individual expert 
          is assigned a non-zero weight for, organised by ground truth label. Compatability is limited 
          to 10 labels, for a mixture of experts classifying samples into more categories, only the 
          first 10 will be considered.
        """
        preds = [net(x) for net in self.experts]
        weights = self.gating_network(x)

        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)
        
        weight_mask = weights > 0.1
        part_sizes = weight_mask.sum(0).cpu().numpy()
        disp = SparseDispatcher(self.n, weight_mask, labels)


        return combined_pred, preds, part_sizes, disp.part_sizes_by_label(), 0

    def forward_dense(self, x):
        """
        Compute combined and individual predictions for x.
        """
        preds = [net(x) for net in self.experts]
        weights = self.gating_network(x)

        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)
        return combined_pred, preds


def apply_laplace_approximation(trainer, train_loader, val_loader, optimize_precision=True, entropy_threshold=None, full_laplace=False):
    """
    Apply the Laplace approximation to the supplied trainer's gating network

    Parameters
    ----------
    - trainer (methods.BaseTrainer): original trainer with a trained model
    - train_loader (torch.utils.data.DataLoader): iterator for the original training data
    - val_loader (torch.utils.data.DataLoader): iterator for the original validation data
    - optimize_precision (bool): whether to optimise the prior precision or use the default value of 1
    - entropy_threshold (float): Threshold to use in entropy conditional gating. If not None, entropy-conditional
      gating is always used.
    - full-laplace (bool): whether to compute the LA for the full network or only the last layer

    Returns
    ----------
    - trainer (methods.BaseTrainer): trainer with a model which now uses a gating network with a LA
    """
    st = time()
    model = laplacify_gating(
                trainer.model, 
                train_loader, 
                val_loader, 
                optimize_precision, 
                gate_by_entropy=entropy_threshold is not None, 
                entropy_threshold=entropy_threshold,
                device=trainer.device,
                full_laplace=full_laplace,
            )

    print(f'LA time taken {time() - st}')
    trainer.model = model
    return trainer

def laplacify_gating(
        model, 
        train_loader, 
        val_loader, 
        optimize_precision=True, 
        gate_by_entropy=False, 
        entropy_threshold=0, 
        device='cpu', 
        full_laplace=False
    ):

    """
    Apply the Laplace approximation to the gating subnetwork of a given network. 
    Assumed to be a MoE model.

    Parameters
    ----------
    - model (torch.nn.Module): original trained model
    - train_loader (torch.utils.data.DataLoader): iterator for the original training data
    - val_loader (torch.utils.data.DataLoader): iterator for the original validation data
    - optimize_precision (bool): whether to optimise the prior precision or use the default value of 1
    - gate_by_entropy (bool): whether to use entropy-conditional gating
    - entropy_threshold (float): Threshold to use in entropy conditional gating.
    - device (str): cpu or cuda, where to perform the computations
    - full-laplace (bool): whether to compute the LA for the full network or only the last layer

    Returns
    ----------
    - wrapped_model (nn.Module): a MoE model which now uses a gating network with a LA
    """

    print('Applying Laplace approximation to the gating network')

    # back-compatability protection, as the structure of models was changed slightly
    try:
        gate = model.gating_network.net
    except:
        gate = model.gating_network

    if full_laplace:
        la_gate = Laplace(gate, 'classification',
                subset_of_weights='all', 
                hessian_structure='full')
    else:
        la_gate = Laplace(gate, 'classification',
                 subset_of_weights='last_layer', 
                 hessian_structure='full')
    
    gate_train_loader = get_adjusted_loader(model.experts, train_loader, device=device)

    print('fitting')
    t1 = time()
    la_gate.fit(gate_train_loader)
    t2 = time()
    print(f'Fitting time: {t2 - t1}')

    if optimize_precision:
        gate_val_loader = get_adjusted_loader(model.experts, val_loader, device=device)
        t1 = time()
        print(f'Val label construction time: {t1 - t2}')

        print('Optimising prior precision')
        la_gate.optimize_prior_precision(method='CV', val_loader=gate_val_loader)
        print(f'optimization time {time() - t1}')

    print('wrapping up a MoE model')
    wrapped_model = DenseLaplaceWrapper(model, la_gate, gate_by_entropy, entropy_threshold)
    return wrapped_model

def get_adjusted_loader(model_list, loader, device='cpu', return_original=False):
    """
    Transforms an orginal dataloader used for training by changing the labels to be useful for the gting network in isolation.
    
    Parameters
    ----------
    - model_list (nn.ModuleList): original trained MoE model experts
    - laoder (torch.util.data.DataLoader): original dataloader 
    - device (str): cpu or cuda, where to perform the computations.
    """
    wrapped_dataset = WrapperDataset(loader.dataset, model_list, device=device, return_orig=return_original)
    # assumes a sampler is used
    return DataLoader(wrapped_dataset, loader.batch_size, sampler=loader.sampler)

def get_soft_label(model_list, sample, gt):
    # compute individual negated losses for the experts
    # nagation allows higher values to signify more desireble options, compatable with softmax
    neg_losses = -1 * torch.Tensor([F.cross_entropy(net(sample.unsqueeze(0)), gt) for net in model_list])
    soft_label = F.softmax(neg_losses, dim=-1)

    return soft_label

def get_label(model_list, sample, gt):
    """
    Computes a proxy label to be used for a gating network of the given MoE model by choosing the subnetwork 
    with the lowest individual loss as the ground truth. Assumes a cross-entropy loss can be used for the experts.
    """
    neg_losses = torch.stack([F.cross_entropy(net(sample), gt, reduction='none') for net in model_list], dim=1)
    val, label = torch.min(neg_losses, dim=-1)

    return label

class WrapperDataset(Dataset):
    def __init__(self, orig_dataset, models, device='cpu', return_orig=False):
        """
        Set up a wrapper dataset that will supply the same samples as the original, but with adjusted labels.

        Parameters
        ----------
        - orig_dataset (torch.util.data.Dataset): original dataset
        - models (nn.ModuleList): original trained MoE model experts
        - la_gating (Laplace.BaseLaplace): gating network with the Laplace Approximation applied
        - device (str): cpu or cuda, where to perform the computations.
        """
        self.original = orig_dataset
        self.models = models
        self.models.to(device)
        self.new_labels = self.get_all_labels(device)
        self.return_original = return_orig

    def get_all_labels(self, device='cpu'):
        """ Compute all the adjusted labels based on the model's outputs """
        loader = DataLoader(self.original, batch_size=3000, shuffle=False)
        label_list = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                labels = get_label(self.models, X, y)
                label_list.append(labels)

        return torch.cat(label_list)

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):

        sample, original_label = self.original[idx]
        new_label = self.new_labels[idx]
        
        if self.return_original:
            return sample, original_label, new_label

        return sample, new_label