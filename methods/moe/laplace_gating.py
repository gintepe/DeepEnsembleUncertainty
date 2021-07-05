from laplace import Laplace

import torch
import torch.nn as nnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def laplacify_gating(model, train_loader, val_loader):

    print('Applying Laplace approximation to the gating network')

    la_gate = Laplace(model.gating_network, 'classification',
             subset_of_weights='last_layer', 
             hessian_structure='full')

    gate_train_loader = get_adjusted_loader(model, train_loader)

    print('fitting')

    la_gate.fit(gate_train_loader)

    gate_val_loader = get_adjusted_loader(model, val_loader)

    print('Optimising prior precision')
    la_gate.optimize_prior_precision(method='CV', val_loader=gate_val_loader)

    print('re-setting the gate')
    # this is actually not possible as the structure requires this to be a nn.Module, and 
    # the LA wrapped options are not.
    # likely need to wrap this in a new model set-up
    model.gating_network = la_gate

    return model

def get_adjusted_loader(moe_model, loader):
    wrapped_dataset = WrapperDataset(loader.dataset, moe_model)
    # assumes a sampler is used
    return DataLoader(wrapped_dataset, loader.batch_size, sampler=loader.sampler)

# quite inefficient, would be better to do it in a batched way
# TODO THE LAPLACE LIB DOES NOT SUPPORT SOFT LABELS
# or more like the torch framework does not support soft labels for some reason....
def get_soft_label(moe_model, sample, gt):
    # compute individual negated losses for the experts
    # nagation allows higher values to signify more desireble options, compatable with softmax
    # print(sample.shape)
    # print(gt.shape)
    # print(gt)
    neg_losses = -1 * torch.Tensor([F.cross_entropy(net(sample.unsqueeze(0)), gt) for net in moe_model.experts])
    soft_label = F.softmax(neg_losses, dim=-1)

    return soft_label

def get_label(moe_model, sample, gt):
    # just pick the expert with lowest loss... sigh....
    neg_losses = torch.Tensor([F.cross_entropy(net(sample.unsqueeze(0)), gt) for net in moe_model.experts])
    # print(neg_losses)
    val, label = torch.min(neg_losses, dim=-1)

    return label

class WrapperDataset(Dataset):
    def __init__(self, orig_dataset, model):
        """
        Set up a wrapper dataset that will supply the same samples as the original, but with adjusted labels
        """
        self.original = orig_dataset
        self.model = model

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):

        sample, original_label = self.original[idx]
        new_label = get_label(self.model, sample, torch.Tensor([original_label]).long())
        
        return sample, new_label