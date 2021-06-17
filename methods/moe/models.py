import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DenseBasicMoE(nn.Module):
    def __init__(self, network_class, n=5, **kwargs):
        super().__init__()
        self.experts = nn.ModuleList([network_class(**kwargs) for i in range(n)])
        #TODO this is not necessarily the best appproach, but for now sort of works since it takes the same input
        # overall it might make sense to have this be a simple MLP
        self.gating_network = network_class(num_classes=n)

    def forward(self, x):
        preds = [net(x) for net in self.experts]
        weights = nn.functional.softmax(self.gating_network(x), dim=-1)
        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)
        
        return combined_pred, preds


class DenseFixedMoE(nn.Module):
    def __init__(self, network_class, n=5, **kwargs):
        super().__init__()
        self.experts = nn.ModuleList([network_class(**kwargs) for i in range(n)])
        self.gating_network = network_class(num_classes=n)
        for param in self.gating_network.parameters():
            param.requires_grad = False

    def forward(self, x):
        preds = [net(x) for net in self.experts]
        weights = nn.functional.softmax(self.gating_network(x), dim=-1)
        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)
        
        return combined_pred, preds