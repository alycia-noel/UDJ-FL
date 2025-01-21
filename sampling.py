import numpy as np
from torchvision import datasets, transforms


def mnist_iid(mnist, amnist, num_users):
    """
    Sample IID client data from MNIST dataset
    :param dataset: torch dataset object
    :param num_users: int, how many users in the federation
    :return: dict of image index
    """
    mnist_dict_users, amnist_dict_users = {}, {}
    num_mnist_items, num_amnist_items = int(len(mnist)/num_users), int(len(amnist)/num_users)
    all_mnist_idxs, all_amnist_idxs = [i for i in range(len(mnist))], [i for i in range(len(amnist))]

    for i in range(num_users):
        mnist_dict_users[i] = set(np.random.choice(all_mnist_idxs, num_mnist_items, replace=False))
        all_mnist_idxs = list(set(all_mnist_idxs) - mnist_dict_users[i])
        amnist_dict_users[i] = set(np.random.choice(all_amnist_idxs, num_amnist_items, replace=False))
        all_amnist_idxs = list(set(all_amnist_idxs) - amnist_dict_users[i])
    
    return mnist_dict_users, amnist_dict_users

def mnist_noniid(mnist, amnist, num_users):
    """
    Sample non-IID client data from MNIST dataset
    :param dataset: torch dataset object
    :param num_users: int, how many users in the federation
    :return: dict of image index
    """

    # 60,000 training images --> 200 images/shard x 300 shards
    mnist_num_shards, mnist_num_imgs = 200, 300
    amnist_num_shards, amnist_num_imgs = 400, 300

    mnist_idx_shard, amnist_idx_shard = [i for i in range(mnist_num_shards)], [i for i in range(amnist_num_shards)]
    mnist_dict_users, amnist_dict_users = {i:np.array([]) for i in range(num_users)}, {i:np.array([]) for i in range(num_users)}
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

    a = [19, 15, 10, 5, 1]
    b = [1, 5, 10, 15, 19]
    # divide and assign a,b shards/client for both mnist and amnist
    for i in range(num_users):
        mnist_rand_set = set(np.random.choice(mnist_idx_shard, a[i], replace=False))
        mnist_idx_shard = list(set(mnist_idx_shard) - mnist_rand_set)
        amnist_rand_set = set(np.random.choice(amnist_idx_shard, b[i], replace=False))
        amnist_idx_shard = list(set(amnist_idx_shard) - amnist_rand_set)
        for mrand in mnist_rand_set:
            mnist_dict_users[i] = np.concatenate((mnist_dict_users[i], mnist_idxs[mrand*mnist_num_imgs:(mrand+1)*mnist_num_imgs]), axis=0)
        for amrand in amnist_rand_set:
            amnist_dict_users[i] = np.concatenate((amnist_dict_users[i], amnist_idxs[amrand*amnist_num_imgs:(amrand+1)*amnist_num_imgs]), axis=0)
    
        print(len(mnist_dict_users[i])+ len(amnist_dict_users[i]))
    return mnist_dict_users, amnist_dict_users

def mnist_noniid_dirichlet(mnist, amnist, num_users):
    """
    Sample non-IID client data from MNIST dataset
    :param dataset: torch dataset object
    :param num_users: int, how many users in the federation
    :return: dict of image index
    """

    # 60,000 training images --> 200 images/shard x 300 shards

    mnist_dict_users, amnist_dict_users = {i:[] for i in range(num_users)}, {i:[] for i in range(num_users)}
    mnist_idxs = np.arange(200*300)
    amnist_idxs = np.arange(400*300)
    mnist_labels = mnist.targets.numpy()
    amnist_labels = amnist.labels.numpy()

    # sort labels
    mnist_idxs_labels = np.vstack((mnist_idxs, mnist_labels))
    mnist_idxs_labels = mnist_idxs_labels[:, mnist_idxs_labels[1,:].argsort()]
    mnist_label_dict = {}

    for j in range(10):
        initial = list(mnist_idxs_labels[1]).index(j)
        if j != 9:
            final = list(mnist_idxs_labels[1]).index(j+1)
        else:
            final = len(mnist_idxs_labels[1])
        mnist_label_dict[j] = list(mnist_idxs_labels[0][initial:final])

 
    amnist_idxs_labels = np.vstack((amnist_idxs, amnist_labels))
    amnist_idxs_labels = amnist_idxs_labels[:, amnist_idxs_labels[1,:].argsort()]
    amnist_label_dict = {}

    for j in range(10):
        initial = list(amnist_idxs_labels[1]).index(j)
        if j != 9:
            final = list(amnist_idxs_labels[1]).index(j+1)
        else:
            final = len(amnist_idxs_labels[1])
        amnist_label_dict[j] = list(amnist_idxs_labels[0][initial:final])


    a = [.95, .75, .5, .25, .05]
    b = [.05, .25, .5, .75, .95]
    probs = np.random.dirichlet([.25, .25, .25, .25, .25, .25, .25, .25, .25, .25], 5)

    # divide and assign a,b shards/client for both mnist and amnist
    for i in range(num_users):
        clean_num = [int((a[i]*p)*600) for p in probs[i]]
        dirty_num = [int((b[i]*p)*600) for p in probs[i]]

        mnist_rand_indicies = []
        amnist_rand_indicies = []

        for c in range(10):
            if clean_num[c] != 0:
                mnist_rand_set = set(np.random.choice(mnist_label_dict[c], clean_num[c], replace=False))
                mnist_label_dict[c] = list(set(mnist_label_dict[c]) - mnist_rand_set)
                mnist_rand_indicies.extend(list(mnist_rand_set))
            if dirty_num[c] != 0:
                amnist_rand_set = set(np.random.choice(amnist_label_dict[c], dirty_num[c], replace=False))
                amnist_label_dict[c] = list(set(amnist_label_dict[c]) - amnist_rand_set)
                amnist_rand_indicies.extend(list(amnist_rand_set))

        print(len(mnist_rand_indicies)+ len(amnist_rand_indicies))
        
        mnist_dict_users[i] = mnist_rand_indicies
        amnist_dict_users[i] = amnist_rand_indicies
    
    return mnist_dict_users, amnist_dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-IID client data from MNIST dataset s.t. clients have unequal amount of data
    :param dataset: torch dataset object
    :param num_users: int, how many users in the federation
    :return: dict of image index
    """

    # 60,000 training imgs --> 50 imgs/shard x 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shard assigned per client
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client s.t. the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1, size=num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        
        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards 
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else: 
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate((dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users