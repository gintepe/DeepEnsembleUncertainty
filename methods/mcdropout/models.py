import torch 
import torch.nn as nn
import torch.nn.functional as F

from methods.models import _weights_init, LambdaLayer

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

class MLPMCDropout(nn.Module):
    """
    Simple multi-layer perceptron
    Consistent with nets used in "Simple and Scalable Predictive 
    Uncertainty Estimation using Deep Ensembles"
    """
    def __init__(self, dropout_p, init_weights = True, size_in=1*28*28, size_out=10, n_hidden_layer=3, n_hidden_units=200):
        super(MLPMCDropout, self).__init__()
        hidden_layers = []
        for i in range(1, n_hidden_layer):
            hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            hidden_layers.append(nn.BatchNorm1d(num_features=n_hidden_units))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(MCDropout(p=dropout_p))

        self.net = nn.Sequential(nn.Linear(size_in, n_hidden_units),
                                nn.BatchNorm1d(num_features=n_hidden_units),
                                nn.ReLU(),
                                MCDropout(p=dropout_p),
                                *hidden_layers,
                                nn.Linear(n_hidden_units, size_out),)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.net(x)
        return x

    def mc_predict(self, x, n_samples):
        preds = [self.forward(x) for i in range(n_samples)]
        combined_pred = torch.mean(nn.functional.softmax(torch.stack(preds.copy(), dim=0), dim=-1), dim=0)

        return combined_pred, preds

    
class ResidualBlockMCDropout(nn.Module):
    def __init__(self, droupout_p, in_channels, out_channels, stride, option='A'):
        super(ResidualBlockMCDropout, self).__init__()
        self.dropout = MCDropout(p=droupout_p)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], 
                                                (0, 0, 0, 0, out_channels//4, out_channels//4), 
                                                "constant", 
                                                0)
                                            )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channels)
                )

    def forward(self, x):
        out = self.dropout(self.block1(x))
        out = self.block2(out)
        out += self.shortcut(x)
        out = self.dropout(F.relu(out))
        return out

class ResNetMCDropout(nn.Module):
    """
    ResNet for CIFAR 10, as described in the paper for the specific use-case
    Can be customised, but will default to ResNet20
    """
    def __init__(self, dropout_p, num_blocks=3, num_classes=10):
        super(ResNetMCDropout, self).__init__()

        self.in_channels=16
        self.num_blocks = num_blocks
        self.dropout_p = dropout_p

        self.conv1 =  nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            MCDropout(p=self.dropout_p)
            )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, out_channels, stride):
        strides = [stride] + [1]*(self.num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlockMCDropout(self.dropout_p, self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def mc_predict(self, x, n_samples):
        preds = [self.forward(x) for i in range(n_samples)]
        combined_pred = torch.mean(nn.functional.softmax(torch.stack(preds.copy(), dim=0), dim=-1), dim=0)

        return combined_pred, preds

