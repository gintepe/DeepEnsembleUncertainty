#!/bin/bash

# the first argument should be the directory to dowload data to
DIR=$1

# make relevant directories
mkdir -p $DIR

cd $DIR

# get CIFAR10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz

wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar

python3  << END
import numpy as np
import pickle
file = 'cifar-10-batches-py/test_batch'
with open(file, 'rb') as fo:
  dict = pickle.load(fo, encoding='bytes')
labels = dict[b'labels']
np.save('CIFAR-10-C/test_labels.npy', np.array(labels))
END


# get CIFAR100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz

wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-100-C.tar

python3  << END
import numpy as np
import pickle
file = 'cifar-100-python/test'
with open(file, 'rb') as fo:
  dict = pickle.load(fo, encoding='bytes')
labels = dict[b'fine_labels']
np.save('CIFAR-100-C/test_labels.npy', np.array(labels))
END

# get MNIST
python3  << END
import torchvision
m1 = torchvision.datasets.MNIST(".", train=True, download=True)
m2 = torchvision.datasets.MNIST(".", train=True, download=True)
END
