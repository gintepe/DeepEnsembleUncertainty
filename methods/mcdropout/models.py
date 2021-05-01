import torch 
import torch.nn as nn

# Identical behavior to dropout, but always applied
class MCDropout(nn.Module):
    def __init__(self, p):
        super(MCDropout, self).__init__()
        self.p = p
    def forward(self, x):
        return nn.functional.dropout(x, self.p, training=True)

class LeNet5MCDropout(nn.Module):
    def __init__(self, dropout_p, init_weights = True):
        super(LeNet5MCDropout, self).__init__()
        self.conv = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            MCDropout(dropout_p),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
                        
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            MCDropout(dropout_p),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
                        
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            MCDropout(dropout_p),
            nn.Tanh(),)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            MCDropout(dropout_p),
            nn.Tanh(),

            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def mc_predict(self, x, n_samples):
        preds = [self.forward(x) for i in range(n_samples)]
        combined_pred = torch.mean(nn.functional.softmax(torch.stack(preds.copy(), dim=0), dim=-1), dim=0)

        return combined_pred, preds