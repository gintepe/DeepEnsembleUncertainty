import torch
import torch.nn as nn

class SimpleEnsemble(nn.Module):
    def __init__(self, network_class=None, networks=None, n=5, **kwargs):
        super(SimpleEnsemble, self).__init__()

        assert(not (network_class is None) == (networks is None))

        if networks:
            self.networks = nn.ModuleList(networks)
        else:
            self.networks = nn.ModuleList([network_class(**kwargs) for i in range(n)])

        self.n = len(self.networks)

    def forward(self, x):
        preds = [net(x) for net in self.networks]
        
        # output actual mean of probabilities
        combined_pred = torch.mean(nn.functional.softmax(torch.stack(preds.copy(), dim=0), dim=-1), dim=0)
        return combined_pred, preds