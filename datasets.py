import torch
import numpy as np

from utils import plot_images
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import constants

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

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

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=False, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=False, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    corrupted=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if corrupted:
        dataset = CorruptedCifar10Test(f'{data_dir}/CIFAR-10-C', transform=transform)
    else:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=False, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class CorruptedCifar10Test(Dataset)

    def __init__(self, data_dir, corruptions=constants.CORRUPTIONS, intensities=range(1,6), transform=None):
        load_corrupted_cifar(data_dir, corruptions, intensities)
        self.transform=transform
    
    def __len__(self):
        return(self.corrupted_images.shape[0])

    def __getitem__(self, idx):

        img = self.corrupted_images[idx]
        label = self.labels[idx % constants.CIFAR10_TEST_N]

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def load_corrupted_cifar(self, data_dir, corruptions=constants.CORRUPTIONS, intensities=range(1,6)):
        self.labels = np.load(f'{data_dir}/test_labels.npy')

        sets = []
        for corruption in corruptions:
            imgs = np.load(f'{data_dir}/{corruption}.npy')
            for intensity in intensities:
                sets.append(imgs[(intensity-1)*constants.CIFAR10_TEST_N : (intensity)*constants.CIFAR10_TEST_N])

        self.corrupted_images = np.permute(np.concatenate(sets, axis=0), (0, 3, 1, 2)

def get_CIFAR10_loaders(data_dir, 
                        batch_size,
                        augment,
                        random_seed,
                        valid_size=0.1,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False,
                        corruptions=constants.CORRUPTIONS, 
                        intensities=range(1,6)):
    
    train_loader, valid_loader = get_train_valid_loader(data_dir, batch_size, augment,
                                    random_seed, valid_size, shuffle, num_workers, pin_memory)
