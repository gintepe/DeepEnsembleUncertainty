import torch
import numpy as np
import os
import gzip

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt

import constants

# def show_images(images, intensity, nmax=64,):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xticks([]); ax.set_yticks([])
#     plt.imshow(f'batch_ex_shift_{intensity}', make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           train_transform,
                           valid_transform,
                           dataset,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False):
    """
    Utility function for loading and returning train and validation
    multi-process iterators over a trochvision image dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    
    Parameters
    ------
    - data_dir (str): path directory to the dataset.
    - batch_size (int): how many samples per batch to load.
    - random_seed (int): fix seed for reproducibility.
    - valid_transform (torchvision.transform): transformations to apply to the validation set
    - train_transform (torchvision.transform): transformations to apply to the training set
    - valid_size (float): fraction of the training set used for the validation set. 
    Should be in the range [0, 1].
    - shuffle (bool): whether to shuffle the train/validation indices.
    - num_workers (int): number of subprocesses to use when loading the dataset.
    - pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load the dataset
    train_dataset = dataset(
        root=data_dir, train=True,
        download=False, transform=train_transform,
    )

    valid_dataset = dataset(
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


def load_mnist(path, kind='train'):

    """
    Load MNIST data form raw files

    Parameters
    ------
    - path (str): path to the MNIST data folder
    - kind (str): type of data to load. train for training, 
    t10k for testing

    Resturns
    ------
    - imgaes (np.ndarray): Array of flattened image data, 
    n_images x pixels
    - labels (np.ndarray): Array of integer labels 
    """

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

