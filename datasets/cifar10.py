import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import datasets.data_util as data_util 

import constants

def get_cifar_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           is_cifar10=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
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
      True if using GPU.
    - is_cifar10 (bool): when true, will load cifar10, otherwise will load cifar100. 
    
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    normalize = transforms.Normalize( mean=constants.CIFAR_MEAN, std=constants.CIFAR_STD,)


    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = valid_transform

    dataset = datasets.CIFAR10 if is_cifar10 else datasets.CIFAR100

    train_loader, valid_loader = data_util.get_train_valid_loader(data_dir, batch_size, 
                                    random_seed, train_transform, valid_transform, dataset, 
                                    valid_size, shuffle, num_workers, pin_memory)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    corrupted=False,
                    corruptions=constants.CORRUPTIONS, 
                    intensities=range(1,6),
                    is_cifar10=True,):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    
    Parameters
    ------
    - data_dir (str): path directory to the dataset.
    - batch_size (int): how many samples per batch to load.
    - shuffle (bool): whether to shuffle the dataset after every epoch.
    - num_workers (int): number of subprocesses to use when loading the dataset.
    - pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - corrupted (bool): whether to load the regular test dataset or the curropted version.
    - corruptions (List[str]): if corrupted, which corruptions to load
    - intensities (List[int]): the intensities to load (1 to 5) for the specfied corruptions
    - is_cifar10 (bool): when true, will load cifar10, otherwise will load cifar100. 
    
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize( mean=constants.CIFAR_MEAN, std=constants.CIFAR_STD,)

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if corrupted:
        sub_data_dir = f'{data_dir}/CIFAR-10-C' if is_cifar10 else f'{data_dir}/CIFAR-100-C'
        dataset = CorruptedCifarTest(sub_data_dir, corruptions, 
                                        intensities, transform=transform)
    elif is_cifar10:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=False, transform=transform,
        )
    else:
        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=False, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class CorruptedCifarTest(Dataset):
    """ Represent the corrupted CIFAR10 dataset alongside correct labels"""

    def __init__(self, data_dir, corruptions=constants.CORRUPTIONS, intensities=range(1,6), transform=None):
        """
        Initialise the dataset and load data to memory

        Parameters
        ------
        - data_dir (str): path directory to the dataset.
        - corruptions (List[str]): if corrupted, which corruptions to load
        - intensities (List[int]): the intensities to load (1 to 5) for the specfied corruptions
        """
        self.load_corrupted_cifar(data_dir, corruptions, intensities)
        self.transform=transform
    
    def __len__(self):
        return(self.corrupted_images.shape[0])

    def __getitem__(self, idx):

        img = self.corrupted_images[idx]
        label = self.labels[idx % constants.CIFAR10_TEST_N]

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def load_corrupted_cifar(self, data_dir, corruptions, intensities):
        """
        Load data with relevant corruptions to memory

        Parameters
        ------
        - data_dir (str): path directory to the dataset.
        - corruptions (List[str]): if corrupted, which corruptions to load
        - intensities (List[int]): the intensities to load (1 to 5) for the specfied corruptions
        """

        self.labels = np.load(f'{data_dir}/test_labels.npy')

        sets = []
        for corruption in corruptions:
            imgs = np.load(f'{data_dir}/{corruption}.npy')
            for intensity in intensities:
                sets.append(imgs[(intensity-1)*constants.CIFAR10_TEST_N : (intensity)*constants.CIFAR10_TEST_N])

        self.corrupted_images = np.concatenate(sets, axis=0)


def get_CIFAR10_loaders(data_dir, 
                        batch_size,
                        augment,
                        random_seed,
                        valid_size=0.1,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False,
                        corrupted_test=False,
                        corruptions=constants.CORRUPTIONS, 
                        intensities=range(1,6),
                        is_cifar10=True):

    """
    Utility function for loading and returning training, validation and testing
    multi-process iterators over the CIFAR-10 dataset.
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
      True if using GPU.
    - corrupted_test (bool): whether to load the regular test dataset or the curropted version.
    - corruptions (List[str]): if corrupted, which corruptions to load
    - intensities (List[int]): the intensities to load (1 to 5) for the specfied corruptions
    - is_cifar10 (bool): when true, will load cifar10, otherwise will load cifar100. 
    
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - test_loader: test set iterator.
    """
    
    train_loader, valid_loader = get_cifar_train_valid_loader(data_dir, batch_size, augment,
                                    random_seed, valid_size, shuffle, num_workers, pin_memory, is_cifar10)
    test_loader = get_test_loader(data_dir, batch_size, shuffle, num_workers, pin_memory,
                                    corrupted_test, corruptions, intensities, is_cifar10)

    return train_loader, valid_loader, test_loader
