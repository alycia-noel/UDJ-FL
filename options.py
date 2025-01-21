import argparse

def arg_parser():
    parser = argparse.ArgumentParser()

    # federated argumnets (Notation for the arguments followed from the paper)
    parser.add_argument('--round', type=int, default=0, help='Which round to run: 0, 1, 2')
    parser.add_argument('--which', type=str, default='normal', help='to run normal,even, clean, or dirty')
    parser.add_argument('--epochs', type=int, default=500, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: k")
    parser.add_argument('--frac', type=float, default=1, help="fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="the local batch size: B")
    parser.add_argument('--lr', type=float, default=.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--q', type=float, default=0, help='value of q [0, 1] for q-FFL')
    parser.add_argument('--which_agg', type=str, default='not_fedavg', help='which aggregation stratgey to use')
    parser.add_argument('--dropout', type=float, default=.5, help='what values of dropout')
    parser.add_argument('--fairness', type=str, default='rawls')
    parser.add_argument('--beta', type=float, default=0, help='beta value for our derivations')
    parser.add_argument('--r', type=float, default=1, help='r value for our derivations')
    parser.add_argument('--epsilon', type=float, default=0.2, help='for PropFair')
    parser.add_argument('--base', type=float, default=5.0, help='for propfair')
    parser.add_argument('--alpha', type=float, default=0.01, help='for TERM')
    parser.add_argument('--fedmgda_epsilon', type=float, default=0.05, help='for fedmgda+')
    parser.add_argument('--global_lr', type=float, default=1.0, help='for fedmgda+')
    parser.add_argument('--inv', type=str, default='not_inverse', help='inverse au weighing')
    parser.add_argument('--scaled', type=str, default='not_scaled', help='whether to scale au value')
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help="number or each type of kernel")
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help="comma-seperated kernel size to use for convolution")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default="batch_norm", help="batch_norm, layer_norm, or none")
    parser.add_argument('--num_filters', type=int, default=32, help="NUmber of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot")
    parser.add_argument('--max_pool', type=bool, default=True, help="Whether to user max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--iid', type=bool, default=False, help="Whether to use iid data or not")
    parser.add_argument('---unequal', type=bool, default=False, help="Whether to use unequal data splits for non-iid setting")
    parser.add_argument('--stopping_rounds', type=int, default=10, help="rounds of early stopping")
    parser.add_argument('--verbose', type=bool, default=True, help="verbose")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    
    args = parser.parse_args()

    return args