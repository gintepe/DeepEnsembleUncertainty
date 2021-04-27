import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF


import data_util 
import constants

def get_mnist_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    normalize = transforms.Normalize(constants.MNIST_MEAN, constants.MNIST_STD )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    
    train_transform = valid_transform

    train_loader, valid_loader = data_util.get_train_valid_loader(data_dir, batch_size, 
                                    random_seed, train_transform, valid_transform, datasets.MNIST, 
                                    valid_size, shuffle, num_workers, pin_memory)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    corrupted=False,
                    intensity=10,):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - corrupted: wether to return original test images or rotate them.
    - intensity: rotation angle.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize =transforms.Normalize((0.1307,), (0.3081,) )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if corrupted:
        dataset = RotatedMNISTTest(f'{data_dir}/MNIST/raw', intensity, transform=transform)
    else:
        dataset = datasets.MNIST(
            root=data_dir, train=False,
            download=False, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class RotatedMNISTTest(Dataset):

    def __init__(self, data_dir, rotation, transform=None):
        self.images, self.labels = data_util.load_mnist(data_dir, 't10k')
        self.transform=transform
        self.rotation = rotation

    def __len__(self):
        return(self.images.shape[0])

    def __getitem__(self, idx):

        img = TF.rotate(self.images, np.random.choice([-1, 1])*self.rotation)

        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]


def get_MNIST_loaders(data_dir, 
                        batch_size,
                        random_seed,
                        valid_size=0.1,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False,
                        corrupted_test=False,):
    
    train_loader, valid_loader = get_mnist_train_valid_loader(data_dir, batch_size,
                                    random_seed, valid_size, shuffle, num_workers, pin_memory)
    test_loader = get_test_loader(data_dir, batch_size, shuffle, num_workers, pin_memory,
                                    corrupted_test,)

    return train_loader, valid_loader, test_loader
