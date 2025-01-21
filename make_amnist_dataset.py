from utils import CURETSRDataset, l2normalize, standardization, DatasetSplit
import torchvision.transforms as transforms
import torch
import argparse
import pickle
import random
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data.dataset import Dataset
import os
from torch.utils.data import DataLoader
import torch.utils.data as data
from utils import FastMNIST, AmbiguousMNIST, DatasetSplit
    
def main(round):

    mnist = FastMNIST('data', train=True, download=True)
    mnist_test_dataset = FastMNIST('data', train=False, download=True)

    amnist = AmbiguousMNIST()
    
    mnist_num_shards, mnist_num_imgs = 200, 300
    amnist_num_shards, amnist_num_imgs = 400, 300

    mnist_idx_shard, amnist_idx_shard = [i for i in range(mnist_num_shards)], [i for i in range(amnist_num_shards)]
    mnist_dict_users, amnist_dict_users = {i:np.array([]) for i in range(5)}, {i:np.array([]) for i in range(5)}
    mnist_idxs = np.arange(mnist_num_shards*mnist_num_imgs)
    amnist_idxs = np.arange(amnist_num_shards*amnist_num_imgs)
    mnist_labels = mnist.targets.numpy()
    amnist_labels = amnist.labels.numpy()

    # sort labels
    mnist_idxs_labels = np.vstack((mnist_idxs, mnist_labels))
    mnist_idxs_labels = mnist_idxs_labels[:, mnist_idxs_labels[1,:].argsort()]
    mnist_idxs = mnist_idxs_labels[0,:]

    amnist_idxs_labels = np.vstack((amnist_idxs, amnist_labels))
    amnist_idxs_labels = amnist_idxs_labels[:, amnist_idxs_labels[1,:].argsort()]
    amnist_idxs = amnist_idxs_labels[0,:]

    a = [3, 3, 3, 3, 3]
    b = [1, 1, 1, 1, 1]
    # divide and assign a,b shards/client for both mnist and amnist
    for i in range(5):
        mnist_rand_set = set(np.random.choice(mnist_idx_shard, a[i], replace=False))
        mnist_idx_shard = list(set(mnist_idx_shard) - mnist_rand_set)
        amnist_rand_set = set(np.random.choice(amnist_idx_shard, b[i], replace=False))
        amnist_idx_shard = list(set(amnist_idx_shard) - amnist_rand_set)
        for mrand in mnist_rand_set:
            mnist_dict_users[i] = np.concatenate((mnist_dict_users[i], mnist_idxs[mrand*mnist_num_imgs:(mrand+1)*mnist_num_imgs]), axis=0)
        for amrand in amnist_rand_set:
            amnist_dict_users[i] = np.concatenate((amnist_dict_users[i], amnist_idxs[amrand*amnist_num_imgs:(amrand+1)*amnist_num_imgs]), axis=0)
    
    all_test = []
    train_loaders = []
    test_loaders = []

    for i in range(5):
        random.shuffle(mnist_dict_users[i])
        random.shuffle(amnist_dict_users[i])

        mnist_idx_train = mnist_dict_users[i][:int(0.8*len(mnist_dict_users[i]))]
        mnist_idx_test = mnist_dict_users[i][int(0.8*len(mnist_dict_users[i])):]
        amnist_idx_train = amnist_dict_users[i][:int(0.8*len(amnist_dict_users[i]))]
        amnist_idx_test = amnist_dict_users[i][int(0.8*len(amnist_dict_users[i])):]

        mnist_train = DatasetSplit(mnist, mnist_idx_train)
        
        mnist_test = DatasetSplit(mnist, mnist_idx_test)
        amnist_train = DatasetSplit(amnist, amnist_idx_train)
        amnist_test = DatasetSplit(amnist, amnist_idx_test)

        dirty_mnist_train = data.ConcatDataset([mnist_train, amnist_train])
        dirty_mnist_test = data.ConcatDataset([mnist_test, amnist_test])
        all_test.append(dirty_mnist_test)
        trainloader = DataLoader(dirty_mnist_train, batch_size=128, shuffle=True)
        testloader = DataLoader(dirty_mnist_test, batch_size=128, shuffle=False)
        train_loaders.append(trainloader)
        test_loaders.append(testloader)
    
    all_dirty_mnist_test = data.ConcatDataset(all_test)
    testloader = DataLoader(all_dirty_mnist_test, batch_size=128, shuffle=False)

    testloader = DataLoader(mnist_test_dataset, batch_size=128, shuffle=False)
    
    all_clean_test = torch.utils.data.ConcatDataset(all_test)
    # testloader = DataLoader(all_clean_and_dirty_test, batch_size=128, shuffle=False)

    testloader = DataLoader(all_clean_test, batch_size=128, shuffle=False)

    print(round, len(train_loaders), len(test_loaders))
    print(train_loaders)

    with open(f'/home/ancarey/distjust/data_files/final/normal_train_dirtymnist_{round}.pkl', 'wb') as f:  
            pickle.dump(train_loaders, f)

    with open(f'/home/ancarey/distjust/data_files/final/normal_test_curetsr_{round}.pkl', 'wb') as f:  
            pickle.dump(test_loaders, f)

    with open(f'/home/ancarey/distjust/data_files/final/normal_testloader_curetsr_{round}.pkl', 'wb') as f:  
            pickle.dump(testloader, f)

if __name__=='__main__':
    for i in range(3):
         main(i)

