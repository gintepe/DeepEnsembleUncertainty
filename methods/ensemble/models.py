import torch
import torch.nn as nn

class SimpleEnsemble(nn.Module):
    """ Class to wrap several networks into an ensemble """
    def __init__(self, network_class=None, networks=None, n=5, **kwargs):
        """
        Initialise an ensemble of identical networks with independent initialisations.
        Either a network_class or networks have to be supplied.

        Parameters
        --------
        - network_class (type): type of network to use in the ensemble
        - networks (List[torch.nn.Module]): list of networks to wrap into the ensemble.
        - n (int): number of networks to intialise if no list is supplied
        - kwargs: parameters to pass to individual network initialisers.
        """
        super(SimpleEnsemble, self).__init__()

        assert(not (network_class is None) == (networks is None))

        if networks:
            self.networks = nn.ModuleList(networks)
        else:
            self.networks = nn.ModuleList([network_class(**kwargs) for i in range(n)])

        self.n = len(self.networks)

    def forward(self, x):
        """
        Assumes a classification problem using a (variation of) cross-entropy loss.
        Computes predictions for batch input x.

        Returns 
        --------
        - combined_pred (torch.Tensor): mean predicted probabilities per class
        - preds (List[torch.Tensor]): individual network predictions as logits
        """
        preds = [net(x) for net in self.networks]

        torch.mean(nn.functional.softmax(torch.stack(preds.copy(), dim=0), dim=-1), dim=0)
        combined_pred = torch.mean(nn.functional.softmax(torch.stack(preds.copy(), dim=0), dim=-1), dim=0)
        return combined_pred, preds
