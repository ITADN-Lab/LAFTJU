import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from cutout import Cutout

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# number of images to load per batch
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

def read_dataset(batch_size=16,valid_size=0.2,num_workers=0,pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
    #   transforms.RandomCrop(32, padding=4),  # pad 0 on all sides first, then randomly crop image to 32x32
        transforms.RandomHorizontalFlip(),  # flip image with 50% probability
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # mean and std used for normalization of each R, G, B channel
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])


    # Convert data to torch.FloatTensor and normalize.
    train_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False,
                                download=True, transform=transform_test)
        

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # Sample elements without replacement according to the given index list
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader