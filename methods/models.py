import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LeNet5(nn.Module):
    # for the 28 by 28 mnist
    """
    Standard implementation of the LeNet5 network.
    Intended for use with the MNIST dataset
    """
    def __init__(self, init_weights = True):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
                        
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
                        
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),

            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# TODO there should be some consistency in batchnorm use, 
# I don't know what the arguments are
# for using it before activation
class MLP(nn.Module):
    """
    Simple multi-layer perceptron
    Consistent with nets used in "Simple and Scalable Predictive 
    Uncertainty Estimation using Deep Ensembles"
    """
    def __init__(self, init_weights = True, size_in=1*28*28, size_out=10, n_hidden_layer=3, n_hidden_units=200):
        super(MLP, self).__init__()
        hidden_layers = []
        for i in range(1, n_hidden_layer):
            hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            hidden_layers.append(nn.BatchNorm1d(num_features=n_hidden_units))
            hidden_layers.append(nn.ReLU())

        self.net = nn.Sequential(nn.Linear(size_in, n_hidden_units),
                                nn.BatchNorm1d(num_features=n_hidden_units),
                                nn.ReLU(),
                                *hidden_layers,
                                nn.Linear(n_hidden_units, size_out),)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.net(x)
        return x

# based on PyTorch source code + repo by Yerlan Idelbayev
# https://github.com/akamaster/pytorch_resnet_cifar10

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, option='A'):
        super(ResidualBlock, self).__init__()
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
        out = self.block1(x)
        out = self.block2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet for CIFAR 10, as described in the paper for the specific use-case
    Can be customised, but will default to ResNet20
    """
    def __init__(self, num_blocks=3, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels=16
        self.num_blocks = num_blocks

        self.conv1 =  nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
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
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
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
