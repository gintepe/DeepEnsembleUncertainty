import torch
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF


import datasets.data_util as data_util 
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
    
    Parameters
    ------
    - data_dir (str): path directory to the dataset.
    - batch_size (int): how many samples per batch to load.
    - augment (bool): whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed (int): fix seed for reproducibility.
    - valid_size (float): fraction of the training set used for
      the validation set. Should be in the range [0, 1].
    - shuffle (bool): whether to shuffle the train/validation indices.
    - num_workers (int): number of subprocesses to use when loading the dataset.
    - pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU

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
                    intensity=10,
                    corruption='rotation'):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    
    Parameters
    ------
    - data_dir (str): path directory to the dataset.
    - batch_size (int): how many samples per batch to load.
    - shuffle (bool): whether to shuffle the dataset after every epoch.
    - num_workers (int): number of subprocesses to use when loading the dataset.
    - pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - corrupted (bool): whether to return original test images or rotate them.
    - intensity (int): rotation angle in degrees or shift in pixels.
    - corruption (str): type of corruption, rotation or translation
    
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(constants.MNIST_MEAN, constants.MNIST_STD)

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if corrupted:
        if corruption == 'rotation':
            dataset = RotatedMNISTTest(f'{data_dir}/MNIST/raw', intensity, transform=transform)
        else:
            dataset = TranslatedMNISTTest(f'{data_dir}/MNIST/raw', intensity, transform=transform)
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
    """Class to wrap the MNIST test set with applied rotations"""

    def __init__(self, data_dir, rotation, transform=None):
        """
        Initialise the dataset and load data

        Parameters
        ------
        - data_dir (str): path to the dataset directory.
        - rotation (int): rotation angle in degrees.
        - transform (torchvision.transform): transformation 
        to apply to images prior to rotation.
        """
        self.images, self.labels = data_util.load_mnist(data_dir, 't10k')
        self.transform=transform
        self.rotation=rotation

    def __len__(self):
        return(self.images.shape[0])

    def __getitem__(self, idx):

        img = self.images[idx].reshape((28, 28, 1)).copy()

        if self.transform:
            img = self.transform(img)

        img = TF.rotate(img, float(random.choice([-1, 1])*self.rotation))
        
        return img, int(self.labels[idx])


class TranslatedMNISTTest(Dataset):
    """Class to wrap the MNIST test set with applied cyclic translation"""

    def __init__(self, data_dir, translation, transform=None):
        """
        Initialise the dataset and load data

        Parameters
        ------
        - data_dir (str): path to the dataset directory.
        - translation (int): translation shift in pixels.
        - transform (torchvision.transform): transformation 
        to apply to images prior to translaiton.
        """
        self.images, self.labels = data_util.load_mnist(data_dir, 't10k')
        self.transform=transform
        self.translation=translation

    def __len__(self):
        return(self.images.shape[0])

    def __getitem__(self, idx):

        img = self.images[idx].reshape((28, 28, 1)).copy()
        
        if self.translation > 0:
            translated_img = np.zeros_like(img)
            translated_img[:, :-1*self.translation, :] = img[:, self.translation:, :]
            translated_img[:, -1*self.translation:, :] = img[:, :self.translation, :]
        else:
            translated_img = img

        if self.transform:
            translated_img = self.transform(translated_img)

        
        return translated_img, int(self.labels[idx])


def get_MNIST_loaders(data_dir, 
                        batch_size,
                        random_seed,
                        valid_size=0.1,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False,
                        corrupted_test=False,
                        intensity=10,
                        corruption='rotation'):

    """
    Utility function for loading and returning training, validation and testing
    multi-process iterators over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    
    Parameters
    ------
    - data_dir (str): path directory to the dataset.
    - batch_size (int): how many samples per batch to load.
    - shuffle (bool): whether to shuffle the dataset after every epoch.
    - num_workers (int): number of subprocesses to use when loading the dataset.
    - pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - corrupted_test (bool): whether to return original test images or rotate them.
    - intensity (int): rotation angle in degrees or shift in pixels.
    - corruption (str): type of corruption, rotation or translation
    
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - test_loader: test set iterator.
    """
    
    train_loader, valid_loader = get_mnist_train_valid_loader(data_dir, batch_size,
                                    random_seed, valid_size, shuffle, num_workers, pin_memory)
    test_loader = get_test_loader(data_dir, batch_size, shuffle, num_workers, pin_memory,
                                    corrupted_test, intensity, corruption)

    return train_loader, valid_loader, test_loader
