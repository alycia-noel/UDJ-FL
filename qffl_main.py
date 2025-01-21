import os
import copy
import time
import warnings
import random
warnings.filterwarnings("ignore")
import pickle
import numpy as np
from tqdm import tqdm

import torch
from copy import deepcopy
import torch.nn.functional as F
from torch import nn
from options import arg_parser
from torch.utils.data import DataLoader
from models import MLP, MLP_colored, CNNMnist
from utils import get_dataset, exp_details

class Client(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.trainloader, self.testloader = train_loader, test_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model):
        # set mode to train model
        model.train()

        # set optimizer for local updates - must be set each time bc we have diff parameters than we left off on
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        loss = 0
      
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels) 
                loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()

        return model.state_dict(), loss/len(self.trainloader)

def test_inference(model, testloaders): #, test_loaders):
    """
    Returns the test accuracy loss, and uncertainty values
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss().to(device)
 
    all_loss, all_accs, all_aus = [], [], []

    for t in testloaders:
        loss, total, correct = 0.0, 0.0, 0.0
        logits = []

        for _, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images)
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(F.softmax(outputs), 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for o in outputs:
                logits.append(o)

        logits = torch.stack(logits)
        all_accs.append(correct/total)
        all_loss.append(loss/len(t))
        all_aus.append(entropy(logits).item())

    return all_loss, all_accs, all_aus

def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return torch.mean(entropy)
    
def zero_model(model):
    zero = deepcopy(model)
    for i, key in enumerate(zero.keys()):
        zero[key] = torch.zeros_like(zero[key])
    return zero

def avg_models(models, weights=None):
    if weights == None:
        total = len(models)
        weights = [1.0/total] * total
    avg = zero_model(models[0])
    for index, model in enumerate(models):
        for i, key in enumerate(avg.keys()):
            model_param = model_to_params(model)[i]
            avg[key] += model_param * weights[index]
    return avg

def sum_models(models):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    weights = [1.0] * len(models)
    return avg_models(models, weights=weights)

def norm2_model(model):
    params = model_to_params(model)
    sum_ = 0.0
    for param in params:
        sum_ += torch.norm(param) ** 2
    return sum_

def model_to_params(model):
    return [model[key] for key in model.keys()]

def sub_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    subtract = deepcopy(model1)
    model_1_params = model_to_params(model1)
    model_2_params = model_to_params(model2)

    for i, key in enumerate(subtract.keys()):
        with torch.no_grad():
            subtract[key] = model_1_params[i] - model_2_params[i] 

    return subtract

def scale_model(model, scale):
    scaled = deepcopy(model)
    model_params = model_to_params(model)
    for i, key in enumerate(scaled.keys()):
        scaled[key] = scale * model_params[i]
    return scaled

def main(r):
    print('-----------------------------------------------------------------------')
    start_time = time.time()
    args = arg_parser()
    args.round = r
    exp_details(args)

    Lipschitz = 1 / args.lr
    q = args.beta
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'dirtymnist':
        with open(f'data_files/final/train_{args.round}.pkl', 'rb') as f:  
            train_loaders = pickle.load(f)

        with open(f'data_files/final/test_loaders_{args.round}.pkl', 'rb') as f:  
            test_loaders = pickle.load(f)

        with open(f'data_files/final/testloader_{args.round}.pkl', 'rb') as f:  
            testloader = pickle.load(f)
    elif args.dataset == 'mnist':
        with open(f'data_files/final/train_mnist_{args.round}.pkl', 'rb') as f:  
            train_loaders = pickle.load(f)

        with open(f'data_files/final/test_mnist_{args.round}.pkl', 'rb') as f:  
            test_loaders = pickle.load(f)

        with open(f'data_files/final/testloader_mnist_{args.round}.pkl', 'rb') as f:  
            testloader = pickle.load(f)
    elif args.dataset == 'curetsr':
        with open(f'data_files/final/train_curetsr_{args.round}.pkl', 'rb') as f:  
            train_loaders = pickle.load(f)

        with open(f'data_files/final/test_curetsr_{args.round}.pkl', 'rb') as f:  
            test_loaders = pickle.load(f)

        with open(f'data_files/final/testloader_curetsr_{args.round}.pkl', 'rb') as f:  
            testloader = pickle.load(f)
   
    amount_data = []

    for t in train_loaders:
        amount_data.append(len(t.dataset))

    print(amount_data)

    if args.model == 'cnn':
            global_model = CNNMnist(args=args)

    elif args.model == 'mlp':
            if args.dataset == 'curetsr':
                img_size = torch.Size([1, 28, 28])
                len_in = 1
                for x in img_size:
                    len_in *= x
                    global_model = MLP_colored(args, dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
            else:
                img_size = torch.Size([1, 28, 28])
                len_in = 1
                for x in img_size:
                    len_in *= x
                    global_model = MLP(args, dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    # copy weights
    global_weights = global_model.state_dict()
    global_loss, global_acc, global_au = [], [], []
    after_agg_loss, after_agg_acc, after_agg_aus = {i: [] for i in range(args.num_users)}, {i: [] for i in range(args.num_users)}, {i: [] for i in range(args.num_users)}
    round_weights = {i: [] for i in range(args.num_users)}

    for epoch in range(args.epochs):
        local_weights = []
        local_losses = []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        
        for idx in range(args.num_users):
            local_model = Client(args=args, train_loader=train_loaders[idx], test_loader=test_loaders[idx])
            w, local_loss = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(local_loss)

        delta_w = [scale_model(sub_models(global_weights, client_model), Lipschitz) for client_model in local_weights]  #\Delta w_k^t = L(w^t - \bar{w}_k^{t+1})
        Delta = [scale_model(delta_w[i], (local_losses[i] ** q)) for i in range(len(delta_w))]
        h = [q * (local_losses[i] ** (q - 1)) * norm2_model(delta_w[i]) + Lipschitz * (local_losses[i] ** q) for i in range(len(delta_w))]  # w_t - \eta w_t+1
      
        global_weights = sub_models(global_weights, scale_model(sum_models(Delta), 1.0 / sum(h)))
        global_model.load_state_dict(global_weights)

        test_loss_g, test_acc_g, au_g  = test_inference(global_model, [testloader])
        after_agg_test_loss, after_agg_test_acc, after_agg_au = test_inference(global_model, test_loaders)

        global_loss.append(round(test_loss_g[0],3))
        global_acc.append(round(test_acc_g[0],3))
        global_au.append(round(au_g[0], 3))
        

        for i in range(args.num_users):
            after_agg_acc[i].append(round(after_agg_test_acc[i],3))
            after_agg_loss[i].append(round(after_agg_test_loss[i],3))
            after_agg_aus[i].append(round(after_agg_au[i], 3))
            # round_weights[i].append(round(client_weights[i],3))

        # print(f'After Agg Loss: {[round(a, 3) for a in after_agg_test_loss]}')
        # print(f'After Agg Accuracy: {[round(a,3) for a in after_agg_test_acc]}')
        # print(f'After Agg AU: {[round(a,3) for a in after_agg_au]}')
        # print(f'Global Test Loss: {test_loss_g[0]:.4f}')
        # print(f'Global Test Accuracy: {100.*test_acc_g[0]:.3f}')
        # print(f'Global AU: {au_g[0]:.4f}')
        # print(f'Round weights: {client_weights}')

    final_client_test_accuracies = []
    final_global_model_accuracy = []
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print('--------------------------------------------------------------------------')
    for i in range(args.num_users):
        print(f'Client {i+1}')
        print(f'After Agg Loss: {after_agg_loss[i]}')
        print(f'After Agg Accuracy: {after_agg_acc[i]}')
        final_client_test_accuracies.append(after_agg_acc[i][-1])
        print(f'After Agg AU: {after_agg_aus[i]}')
        print(f'Weight: {round_weights[i]}')

    print(f'Global Test Loss: {global_loss}')
    print(f'Global Test Accuracy: {global_acc}')
    final_global_model_accuracy.append(global_acc[-1])
    print(f'Global AU: {global_au}')

    return final_client_test_accuracies, final_global_model_accuracy


if __name__ == '__main__':
    all_test_accs = [[] for _ in range(5)]
    final_global_accs = []

    for i in range(3):
        test_accs, global_acc = main(i)
        for j in range(5):
            all_test_accs[j].append(test_accs[j])
            final_global_accs.append(global_acc)

    avg_client_accs = [round(100*np.average(a),2) for a in all_test_accs]
    avg_client_full_test_accs = 100*round(np.average(final_global_accs),2)

    std_client_accs = [round(np.std(a),3) for a in all_test_accs]
    std_client_full_test_accs = round(np.std(final_global_accs),3)

    client_std = np.std([round(np.average(a),2) for a in all_test_accs])

    print('\n\n\n-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('Average/std over three rounds:')
    print('------------------------------')
    print('Client Test Acc:', avg_client_accs, std_client_accs)
    print('Global Acc:', avg_client_full_test_accs, std_client_full_test_accs)
    print('Client Acc STD:', round(client_std, 4))