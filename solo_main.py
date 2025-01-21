import torch
import numpy as np
import random
import copy
import pickle
import torch.utils.data as data
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from utils import get_dataset
from options import arg_parser
from update import test_inference, Client, get_uncertainties
from models import MLP, MLP_colored, CNNMnist

def main(r): 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    args = arg_parser()
    args.round = r
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

    # Build model
    if args.model == 'cnn':
        global_model = CNNMnist(args=args)

    elif args.model == 'mlp':
        if args.dataset == 'curetsr':
            img_size = torch.Size([1, 28, 28])
            len_in = 1
            for x in img_size:
                len_in *= x
                model = MLP_colored(args, dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
        else:
            img_size = torch.Size([1, 28, 28])
            len_in = 1
            for x in img_size:
                len_in *= x
                model = MLP(args, dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)


    final_accs = []
    final_full_accs = []
    final_aus = []

    print('-----------------------------------------------------------------------')
    for i in range(args.num_users):
        print('\n\n')
        # get train and test loaders
        current_client = Client(args=args, train_loader=train_loaders[i], test_loader=test_loaders[i], full_test=testloader)
        ctrainloader, ctestloader = current_client.trainloader, current_client.testloader
        
        global_model = copy.deepcopy(model)
        # set the model to train and send to device
        global_model.to(device)
        global_model.train()

        # Training
        # Set optimizer and criterion
        # sgd for dirtymnist and adam for curetsr
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)#, momentum=args.momentum)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        epoch_loss_train = []
        epoch_acc_train = []
        epoch_loss_test = []
        epoch_acc_test = []
        gepoch_loss_test = []
        gepoch_acc_test = []
        epoch_au_train, epoch_au_test = [], []
        epoch_tu_train, epoch_tu_test = [], []
        epoch_eu_train, epoch_eu_test = [], []

        for epoch in range(args.epochs):
            batch_loss = []
            batch_correct, batch_total = 0.0, 0.0
            
            for _, (images, labels) in enumerate(ctrainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                # print(loss)
                loss.backward()
                #for curetsr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
          

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                batch_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                batch_total += len(labels)

                batch_loss.append(loss.item())
           
            train_loss, train_acc, au, tu, eu= test_inference(global_model, [ctrainloader])
            # print("\nEpoch {}".format(epoch+1), "\nTrain Accuracy: {:.3f}%".format(train_acc[0]*100), "Train Loss: {:.6f}".format(train_loss[0]), "AU: {:.4f}".format(au[0]))
            epoch_loss_train.append(train_loss[0])
            epoch_acc_train.append(train_acc[0])
            epoch_au_train.append(au[0])
            epoch_tu_train.append(tu[0])
            epoch_eu_train.append(eu[0])

            # testing
            test_loss, test_acc, au, tu, eu= test_inference(global_model, [ctestloader])
            # print("Test Accuracy: {:.2f}%".format(100*test_acc[0]), "Test Loss: {:.6f}".format(test_loss[0]), "AU: {:.4f}".format(au[0]))
            epoch_loss_test.append(test_loss[0])
            epoch_acc_test.append(test_acc[0])
            epoch_au_test.append(au[0])
            epoch_eu_test.append(eu[0])
            epoch_tu_test.append(tu[0])

            # testing
            test_loss, test_acc, au, tu, eu = test_inference(global_model, [testloader])
            # print("Full Test Accuracy: {:.2f}%".format(100*test_acc[0]), "Full Test Loss: {:.6f}".format(test_loss[0]))
            gepoch_loss_test.append(test_loss[0])
            gepoch_acc_test.append(test_acc[0])

        print(f'Client {i+1}')
        print(f'Training Loss : {epoch_loss_train}')
        print(f'Train Accuracy: {epoch_acc_train}')
        print(f'Train AU: {epoch_au_train}')
        print(f'Test Loss : { epoch_loss_test}')
        print(f'Test Accuracy: {epoch_acc_test}')
        print(f'Test AU: {epoch_au_test}')
        print(f'Full Test Loss : {gepoch_loss_test}')
        print(f'Full Test Accuracy: {gepoch_acc_test}')

        final_accs.append(epoch_acc_test[-1])
        final_full_accs.append(gepoch_acc_test[-1])
        final_aus.append(epoch_au_train[-1])

    return  final_accs, final_full_accs, final_aus

if __name__ == '__main__':
    all_test_accs = [[] for _ in range(5)]
    all_full_test_accs = [[] for _ in range(5)]
    all_aus = [[] for _ in range(5)]

    for i in range(3):
        test_accs, full_test_accs, aus = main(i)
        for j in range(5):
            all_test_accs[j].append(test_accs[j])
            all_full_test_accs[j].append(full_test_accs[j])
            all_aus[j].append(aus[j])

    avg_client_accs = [round(100*np.average(a),2) for a in all_test_accs]
    avg_client_full_test_accs = [100*round(np.average(a),2) for a in all_full_test_accs]
    avg_client_aus = [round(np.average(a),4) for a in all_aus]

    std_client_accs = [round(np.std(a),3) for a in all_test_accs]
    std_client_full_test_accs = [round(np.std(a),3) for a in all_full_test_accs]
    std_client_aus = [round(np.std(a),4) for a in all_aus]

    client_std = np.std([round(np.average(a),4) for a in all_test_accs])

    print('\n\n\n-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('Average/std over three rounds:')
    print('------------------------------')
    print('Client Test Acc:', avg_client_accs, std_client_accs)
    print('Client Full Test Acc:', avg_client_full_test_accs, std_client_full_test_accs)
    print('Client AU:', avg_client_aus, std_client_aus)
    print('Client STD:', client_std)