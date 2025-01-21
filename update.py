import torch
import random
import math
import numpy as np
import torch.utils.data as data
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Client(object):
    def __init__(self, args, train_loader, test_loader, full_test):
        self.args = args
        self.trainloader, self.testloader, self.full_test = train_loader, test_loader, full_test
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def update_weights(self, model):
        # set mode to train model
        model.train()

        # set optimizer for local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            batch_loss = []

            batch_correct, batch_total = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                batch_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                batch_total += len(labels)

                batch_loss.append(loss.item())

        train_loss, train_acc, train_au, train_tu, train_eu = test_inference(model, [self.trainloader])
        test_loss, test_acc, test_au, test_tu, test_eu = test_inference(model, [self.testloader])
        ftest_loss, ftest_acc, _, _, _ = test_inference(model, [self.full_test])

        return model.state_dict(), train_loss[0], train_acc[0], test_acc[0], test_loss[0], train_tu[0], train_au[0], train_eu[0], test_tu[0], test_au[0], test_eu[0], ftest_acc[0], ftest_loss[0]
    
def comp_discr_entropy(probs: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """Compute Shannon entropy with base-two log."""
    probs_stabilized = probs + eps
    return -(probs * probs_stabilized.log2()).sum(-1)

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

    return all_loss, all_accs, all_aus, [0,0,0,0,0], [0,0,0,0,0]

def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    
    return torch.mean(entropy)

def entropy_prob(probs):
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy

def mutual_information_prob(probs):
    mean_output = torch.mean(probs, dim=0)
    predictive_entropy = entropy_prob(mean_output)

    # Computing expectation of entropies
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    exp_entropies = torch.mean(-torch.sum(plogp, dim=2), dim=0)

    # Computing mutual information
    mi = predictive_entropy - exp_entropies
    return predictive_entropy, exp_entropies, mi

def get_uncertainties(probs):
    outputs = torch.cat(probs, dim=0)
    tu, au, eu = mutual_information_prob(outputs)

    return tu, au, eu