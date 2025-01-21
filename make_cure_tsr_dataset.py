from utils import CURETSRDataset, l2normalize, standardization, DatasetSplit
import torchvision.transforms as transforms
import torch
import argparse
import pickle
import random
import numpy as np
import os
from torch.utils.data import DataLoader

def main(round):
    data = '/home/ancarey/distjust/data/'
    batch_size = 128

    clean_dir = os.path.join(data, 'CURE-TSR/Real_Train/ChallengeFree')
    dirty_dir = os.path.join(data, 'CURE-TSR/Real_Train/LensBlur-5')

    clean = CURETSRDataset(clean_dir, transforms.Compose([transforms.Resize([28, 28]), transforms.ToTensor(), l2normalize, standardization]))
    dirty = CURETSRDataset(dirty_dir, transforms.Compose([transforms.Resize([28, 28]), transforms.ToTensor(), l2normalize, standardization]))
    
    print(len(clean), len(dirty))
    num_shards, img_per_shard = 258, 60

    clean_idx_shard = [i for i in range(num_shards)]
    clean_dict_users = {i:np.array([]) for i in range(5)}
    clean_idxs = np.arange(num_shards*img_per_shard)
    clean_labels = np.array(clean.targets)[:15480]

    dirty_idx_shard = [i for i in range(num_shards)]
    dirty_dict_users = {i:np.array([]) for i in range(5)}
    dirty_idxs = np.arange(num_shards*img_per_shard)
    dirty_labels = np.array(dirty.targets)[:15480]
    

    clean_idxs_labels = np.vstack((clean_idxs, clean_labels))
    clean_idxs_labels = clean_idxs_labels[:, clean_idxs_labels[1,:].argsort()]
    clean_idxs = clean_idxs_labels[0,:]

    dirty_idxs_labels = np.vstack((dirty_idxs, dirty_labels))
    dirty_idxs_labels = dirty_idxs_labels[:, dirty_idxs_labels[1,:].argsort()]
    dirty_idxs = dirty_idxs_labels[0,:]

    a = [19, 15, 10, 5, 1]
    b = [1, 5, 10, 15, 19]

    train_loaders = []
    test_loaders = []
    all_test = []

    for i in range(5):
            clean_rand_set = set(np.random.choice(clean_idx_shard, a[i], replace=False))
            clean_idx_shard = list(set(clean_idx_shard) - clean_rand_set)
            
            dirty_rand_set = set(np.random.choice(dirty_idx_shard, b[i], replace=False))
            dirty_idx_shard = list(set(dirty_idx_shard) - dirty_rand_set)

            for mrand in clean_rand_set:
                clean_dict_users[i] = np.concatenate((clean_dict_users[i], clean_idxs[mrand*img_per_shard:(mrand+1)*img_per_shard]), axis=0)

            for mrand in dirty_rand_set:
                dirty_dict_users[i] = np.concatenate((dirty_dict_users[i], dirty_idxs[mrand*img_per_shard:(mrand+1)*img_per_shard]), axis=0)


    for i in range(5):
        random.shuffle(clean_dict_users[i])
        random.shuffle(dirty_dict_users[i])

        clean_idx_train = clean_dict_users[i][:int(0.8*len(clean_dict_users[i]))]
        clean_idx_test = clean_dict_users[i][int(0.8*len(clean_dict_users[i])):]
        dirty_idx_train = dirty_dict_users[i][:int(0.8*len(dirty_dict_users[i]))]
        dirty_idx_test = dirty_dict_users[i][int(0.8*len(dirty_dict_users[i])):]
       

        clean_train = DatasetSplit(clean, clean_idx_train)
        dirty_train = DatasetSplit(dirty, dirty_idx_train)
        # train = clean_train
        train = torch.utils.data.ConcatDataset([clean_train, dirty_train])
        trainloader = DataLoader(train, batch_size=128, shuffle=True)
        train_loaders.append(trainloader)
       
        clean_test = DatasetSplit(clean, clean_idx_test)
        dirty_test = DatasetSplit(dirty, dirty_idx_test)
        test = torch.utils.data.ConcatDataset([clean_test, dirty_test])
        all_test.append(clean_test)
        testloader = DataLoader(test, batch_size=128, shuffle=False)
        test_loaders.append(testloader)
    
    all_clean_test = torch.utils.data.ConcatDataset(all_test)
    # testloader = DataLoader(all_clean_and_dirty_test, batch_size=128, shuffle=False)

    testloader = DataLoader(all_clean_test, batch_size=128, shuffle=False)

    print(round, len(train_loaders), len(test_loaders))
    print(train_loaders)
    print(y)
    with open(f'/home/ancarey/distjust/data_files/final/train_curetsr_{round}.pkl', 'wb') as f:  
            pickle.dump(train_loaders, f)

    with open(f'/home/ancarey/distjust/data_files/final/test_curetsr_{round}.pkl', 'wb') as f:  
            pickle.dump(test_loaders, f)

    with open(f'/home/ancarey/distjust/data_files/final/testloader_curetsr_{round}.pkl', 'wb') as f:  
            pickle.dump(testloader, f)

if __name__=='__main__':
    for i in range(3):
         main(i)

