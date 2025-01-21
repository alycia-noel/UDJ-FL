import random
import os
import copy
import numpy as np
from copy import deepcopy
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_dirichlet
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image

class DatasetSplit(Dataset):
    """
    An abstract dataset class wrapped around Pytorch dataset class
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.labels = [self.dataset[int(i)][1] for i in idxs]

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class FastMNIST(MNIST):
    def __init__(self, root, train, download):
        super().__init__(root, train, download)
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(.1307).div_(0.3081)
    
    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        return img, label
        
class AmbiguousMNIST(Dataset):
    def __init__(self):
        self.data = torch.load("data/amnist/amnist_samples.pt")
        self.labels = torch.load("data/amnist/amnist_labels.pt")

        self.data = self.data.sub_(0.1307).div_(0.3081)
        self.data = self.data.expand(-1, self.labels.shape[1], 28, 28).reshape(-1, 1, 28, 28)
        self.labels = self.labels.reshape(-1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        return img, label
    
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_dataset(traindir):
    imgs = []
    targets = []
    for fname in sorted(os.listdir(traindir)):
        target = int(fname[3:5]) - 1
        path = os.path.join(traindir, fname)
        # item = (path, target)
        imgs.append(path)
        targets.append(target)
    return imgs, targets

def _is_tensor_image(img):
    return torch.is_tensor(img), img.ndimension() == 3

def l2normalize(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    tensor = tensor.mul(255)
    norm_tensor = tensor/torch.norm(tensor)
    return norm_tensor

def standardization(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    for t in tensor:
        t.sub_(t.mean()).div_(t.std())
        
    return tensor

 
class CURETSRDataset(Dataset):
    def __init__(self, traindir, transform=None, target_transform =None, loader = pil_loader):
        self.traindir = traindir
        self.imgs, self.targets = make_dataset(traindir)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index], self.targets[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


        
def get_dataset(args):
    """
    Returns train and test dataset and a user group which is a dict where
    the keys are the user index and the values are the corresponding data
    for each of those users. 
    """

    if args.dataset == 'mnist':
        mnist_dataset = FastMNIST('data', train=True, download=True)
        mnist_test_dataset = FastMNIST('data', train=False, download=True)
        # mnist_dataset = data.ConcatDataset([mnist_train_dataset, mnist_test_dataset])

        amnist_dataset = AmbiguousMNIST()

        # sample training data amongst users
        if args.iid:
            mnist_dict_users, amnist_dict_users = mnist_iid(mnist_dataset, amnist_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(mnist_dataset, args.num_users)
            else:
                mnist_dict_users, amnist_dict_users = mnist_noniid(mnist_dataset, amnist_dataset, args.num_users)
    
    
    all_test = []
    train_loaders = []
    test_loaders = []
    
    # Create dataloaders 
    for i in range(args.num_users):
        random.shuffle(mnist_dict_users[i])
        random.shuffle(amnist_dict_users[i])

        mnist_idx_train = mnist_dict_users[i][:int(0.8*len(mnist_dict_users[i]))]
        mnist_idx_test = mnist_dict_users[i][int(0.8*len(mnist_dict_users[i])):]
        amnist_idx_train = amnist_dict_users[i][:int(0.8*len(amnist_dict_users[i]))]
        amnist_idx_test = amnist_dict_users[i][int(0.8*len(amnist_dict_users[i])):]

        mnist_train = DatasetSplit(mnist_dataset, mnist_idx_train)
        
        # temp_dict = {i:0 for i in range(10)}
        # for l in mnist_train.labels:
        #     temp_dict[l.item()] += 1
        
        # print(temp_dict)
        
        mnist_test = DatasetSplit(mnist_dataset, mnist_idx_test)
        amnist_train = DatasetSplit(amnist_dataset, amnist_idx_train)
        amnist_test = DatasetSplit(amnist_dataset, amnist_idx_test)

        dirty_mnist_train = data.ConcatDataset([mnist_train, amnist_train])
        dirty_mnist_test = data.ConcatDataset([mnist_test, amnist_test])
        all_test.append(dirty_mnist_test)
        trainloader = DataLoader(dirty_mnist_train, batch_size=args.local_bs, shuffle=True)
        testloader = DataLoader(dirty_mnist_test, batch_size=128, shuffle=False)
        train_loaders.append(trainloader)
        test_loaders.append(testloader)
    
    all_dirty_mnist_test = data.ConcatDataset(all_test)
    testloader = DataLoader(all_dirty_mnist_test, batch_size=128, shuffle=False)

    testloader = DataLoader(mnist_test_dataset, batch_size=128, shuffle=False)
    return train_loaders, test_loaders, testloader, mnist_dataset[0][0].shape

def average_weights(args, beta, w, amount_data, previous_weights, aus):
  
    if args.round == 0:
        au_scores = [0.0609,0.123,0.15,0.1679,0.27239]
    elif args.round == 1:
        au_scores = [0.052,0.0917,0.16018,0.2616,0.2115]
    elif args.round == 2:
        au_scores = [0.09327,0.09599,0.1461,0.2204,0.2775]
        
    w_avg = deepcopy(w[0])

    pow_au = [a ** beta for a in au_scores]
    weights = [a/sum(pow_au) for a in pow_au]

    if args.which_agg == 'fedavg':
        weights = [a/sum(amount_data) for a in amount_data] # -- FedAvg

    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += w[i][key]*weights[i]
        # w_avg[key] = torch.div(w_avg[key], len(weights))
    return w_avg, weights

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : SGD')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Round     : {args.round}')
    print(f'    Beta/q    : {args.beta}\n')
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : 100%')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def get_beta(args, current_beta, current_std, client_accuracies, current_global_model_accuracy):
    if args.round == 0:
        solo_accs = [.97, .9375, .945, .92166, .9116]
    elif args.round == 1: 
        solo_accs = [.9683, .9408, .9233, .895, .92083]
    elif args.round == 2:
        solo_accs = [.9483, .945, .9308, .9091, .905]

    if args.round == 1:
        ordering = [1, 2, 3, 5, 4] # highest au to lowest
    else:
        ordering = [1, 2, 3, 4, 5] 

    clients = [1, 2, 3, 4, 5]

    client_acc_ = np.vstack((clients, client_accuracies))
    client_acc_ = client_acc_[:, client_acc_[1,:].argsort()]

    if args.fairness == 'rawls':
        # Want the absolute increase in accuracy for clients with high au to be more than the absolute increase in accuracy for clients with low au

        absolute_difference = [s - a for s, a in zip(solo_accs, client_accuracies)]

        if args.round != 1:
            highest_au_diff = absolute_difference[-1]
        else:
            highest_au_diff = absolute_difference[-2]
        
        lowest_au_diff = absolute_difference[0]


        if highest_au_diff >= 0 and lowest_au_diff >= 0:
            total_disparity = highest_au_diff - lowest_au_diff
        elif highest_au_diff >= 0 and lowest_au_diff <= 0:
            total_disparity = highest_au_diff + abs(lowest_au_diff)
        elif highest_au_diff <= 0 and lowest_au_diff >= 0:
            total_disparity = 0
        elif highest_au_diff <= 0 and lowest_au_diff <= 0:
            total_disparity =  abs(lowest_au_diff) - abs(highest_au_diff) 
        else:
            print('hit', highest_au_diff, lowest_au_diff)

        if total_disparity > 0:
                current_beta += total_disparity

    elif args.fairness == 'egal':
        std = np.std(client_accuracies)
        if std > 0:
            if std < current_std:
                current_beta += std
            else:
                current_beta = current_beta
        current_std = std
    
    elif args.fairness == 'desert':
        ordered_accuracies = [client_accuracies[o-1] for o in ordering]
        differences = [ordered_accuracies[i] - ordered_accuracies[i+1] for i in range(len(ordered_accuracies)-1)]

        # print(ordered_accuracies)
        # print(differences)
        # avg_difference = np.mean(differences)
        # if avg_difference < 0:
        #     current_beta =+ avg_difference
        for d in differences:
            if d < 0:
                current_beta += d
    print(current_beta)


        # mismatch = sum([1 for i, (a, b) in enumerate(zip(client_acc_[0], ordering)) if a != b]) / args.num_users
        
        # if mismatch != 0:
        #     current_beta += mismatch / 100


    # print()

    return current_beta, 0
        

        





