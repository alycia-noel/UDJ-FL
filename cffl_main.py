import os
import sys
import json
import copy
import time
import datetime
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
from itertools import product

import numpy as np
import torch
from torch import nn, optim
import pandas as pd
from collections import defaultdict
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import MLP_colored_for_CFFL, MLP_for_CFFL

def aggregate_gradient_updates(grad_updates, R, device=None, mode='sum', credits=None, shard_size=None):
    if grad_updates:
        len_first = len(grad_updates[0])
        assert all(len(i) == len_first for i in grad_updates), "Different shapes of parameters. Cannot aggregate."
    else:
        return
    
    grad_updates_ = [copy.deepcopy(grad_update) for i, grad_update in enumerate(grad_updates) if i in R]

    if device:
        for i, grad_update in enumerate(grad_updates_):
            grad_updates_[i] = [param.to(device) for param in grad_update]
    
    if credits is not None:
        credits = [credit for i, credit in enumerate(credits) if i in R]

    if shard_size is not None:
        shard_sizes = [shard_size for i, shard_size in enumerate(shard_sizes) if i in R]

    aggregated_gradient_updates = []
    if mode=='mean':
        # default mean is FL-avg: weighted avg according to nk/n
        if shard_sizes is None:
            shard_sizes = torch.ones(len(grad_updates))

        for i, (grad_update, shard_size) in enumerate(zip(grad_updates_, shard_sizes)):
            grad_updates_[i] = [(shard_size * update) for update in grad_update]
        for i in range(len(grad_updates_[0])):
            aggregated_gradient_updates.append(torch.stack([grad_update[i] for grad_update in grad_updates_]).mean(dim=0))

    elif mode =='sum':
        for i in range(len(grad_updates_[0])):
            aggregated_gradient_updates.append(torch.stack([grad_update[i] for grad_update in grad_updates_]).sum(dim=0))

    elif mode == 'credit-sum':
		# first changes the grad_updates altogether
        for i, (grad_update, credit) in enumerate(zip(grad_updates_, credits)):
            grad_updates_[i] = [(credit * update) for update in grad_update]

		# then compute the credit weight sum
        for i in range(len(grad_updates_[0])):
            aggregated_gradient_updates.append(torch.stack([grad_update[i] for grad_update in grad_updates_]).sum(dim=0))

    return aggregated_gradient_updates

def average_models(models, device=None):
    final_model = copy.deepcopy(models[0])
    if device:
        models = [model.to(device) for model in models]
        final_model = final_model.to(device)

    averaged_parameters = aggregate_gradient_updates([list(model.parameters()) for model in models], mode='mean')

    for param, avg_param in zip(final_model.parameters(), averaged_parameters):
        param.data = avg_param.data
    return final_model

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update:
        return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    
    return model 

def compute_grad_update(old_model, new_model, device=None):
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        return True
    
def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
    assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight

def evaluate(model, eval_loader, device, loss_fn=None, verbose=True):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            batch_data, batch_target = batch[0].to(device), batch[1].to(device)
            outputs = model(batch_data)
            if loss_fn:
                loss = loss_fn(outputs, batch_target)
            else:
                loss = None
            correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
            total += len(batch_target)

        accuracy = correct.float() / total 

    if verbose:
        print("Loss: {:.6f}. Accuracy {:.4f}.".format(loss, accuracy))
    
    return loss, accuracy

class Participant():
    def __init__(self, train_loader, model, optimizer, scheduler, model_pretrain, optimizer_pretrain, pretraining_lr, scheduler_pretrain, standalone_model, standalone_optimizer, standalone_scheduler, dssgd_model, dssgd_optimizer, dssgd_scheduler, fedavg_model, fedavg_optimizer, fedavg_scheduler, loss_fn, theta=0.1, grad_clip=0.01, epoch_sample_size=-1, device='cuda', id=None, is_free_rider=False):
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_pretrain = model_pretrain
        self.optimizer_pretrain = optimizer_pretrain
        self.pretraining_lr = pretraining_lr
        self.scheduler_pretrain = scheduler_pretrain
        self.standalone_model = standalone_model
        self.standalone_optimizer = standalone_optimizer
        self.standalone_scheduler = standalone_scheduler
        self.dssgd_model = dssgd_model
        self.dssgd_optimizer = dssgd_optimizer
        self.dssgd_scheduler = dssgd_scheduler
        self.fedavg_model = fedavg_model
        self.fedavg_optimizer = fedavg_optimizer
        self.fedavg_scheduler = fedavg_scheduler
        self.loss_fn = loss_fn
        self.theta = theta
        self.grad_clip = grad_clip
        self.device = device
        self.id = id
        self.epoch_sample_size = epoch_sample_size
        self.param_count = sum([p.numel() for p in self.model.parameters()])
        self.is_free_rider = is_free_rider

    def train(self, epochs, is_pretrain=False, save_gpu=False):
        if self.is_free_rider:
            for model in [self.model, self.model_pretrain, self.dssgd_model, self.standalone_model]:
                model = model.to(self.device)

                for param in model.parameters():
                    param.data += (torch.rand(param.data.shape) * 2 - 1).to(self.device)
            return
        
        self.model_pretrain.train()
        self.model_pretrain = self.model_pretrain.to(self.device)

        self.model.train()
        self.model = self.model.to(self.device)

        self.standalone_model.train()
        self.standalone_model = self.standalone_model.to(self.device)

        self.dssgd_model.train()
        self.dssgd_model = self.dssgd_model.to(self.device)

        self.fedavg_model.train()
        self.fedavg_model = self.fedavg_model.to(self.device)

        for epoch in range(int(epochs)):
          
            iter = 0
            for i, batch in enumerate(self.train_loader):
                batch_data, batch_target = batch[0].to(self.device), batch[1].to(self.device)

                # introduce separate (and slower) pretraining
                if is_pretrain:
                    prev_lrs = []
                    if self.pretraining_lr is not None:
                        for g in self.optimizer_pretrain.param_groups:
                            prev_lrs.append(g['lr'])
                            g['lr'] = self.pretraining_lr
                    
                    self.optimizer_pretrain.zero_grad()
                    self.loss_fn(self.model_pretrain(batch_data), batch_target).backward()
                    self.optimizer_pretrain.step()

                    # change the lr back so it does not affect FL training
                    if self.pretraining_lr is not None:
                        for g, lr in zip(self.optimizer_pretrain.param_groups, prev_lrs):
                            g['lr'] = lr
                    continue 

                iter += len(batch_data)
                for optimizer, model in zip([self.optimizer_pretrain, self.optimizer, self.standalone_optimizer, self.dssgd_optimizer, self.fedavg_optimizer], [self.model_pretrain, self.model, self.standalone_model, self.dssgd_model, self.fedavg_model]):
                    optimizer.zero_grad()
                    self.loss_fn(model(batch_data), batch_target).backward()
                    optimizer.step()

                if iter >= self.epoch_sample_size:
                    # specifically for NLP task to terminate for training efficiency
                    break

        if not is_pretrain:
            # No lr decay during pretraining
            self.standalone_scheduler.step()
            self.scheduler_pretrain.step()
            self.scheduler.step()
            self.dssgd_scheduler.step()
            self.fedavg_scheduler.step()

        if 'cuda' in str(self.device) and save_gpu:
            cpu = torch.device('cpu')
            self.model_pretrain = self.model_pretrain.to(cpu)
            self.model = self.model.to(cpu)
            self.standalone_model = self.standalone_model.to(cpu)
            self.dssgd_model = self.dssgd_model.to(cpu)
            self.fedavg_model = self.fedavg_model.to(cpu)

class Federated_Learner():
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.device_ids = args['device_ids']
        self.save_gpu = args['save_gpu'] if 'save_gpu' in args else False
        self.n_participants = args['n_participants']
        self.n_freeriders = args['n_freeriders']
        self.test_loader = args['test_loader']
        self.test_loaders = args['test_loaders']
        self.participant_train_loaders = args['train_loaders']
        self.shard_sizes = args['shard_size']
        self.init_participants()
        self.performance_dict = defaultdict(list)
        self.performance_dict_pretrain = defaultdict(list)
        self.time_dict = defaultdict(float)

    def init_participants(self):
        assert self.n_participants == len(self.participant_train_loaders)

        model_fn = self.args['model_fn']
        optimizer_fn = self.args['optimizer_fn']
        lr = self.args['lr']
        fed_lr = self.args['fed_lr'] if 'fed_lr' in self.args else lr
        dssgd_lr = self.args['dssgd_lr']
        std_lr = self.args['std_lr'] if 'std_lr' in self.args else dssgd_lr

        device = self.args['device']
        loss_fn = self.args['loss_fn']
        theta = self.args['theta']
        epoch_sample_size = self.args['epoch_sample_size']
        grad_clip = self.args['grad_clip']
        gamma = self.args['gamma']
        dim_in = self.args['dim_in']
        dim_hidden = self.args['dim_hidden']
        dim_out = self.args['dim_out']

        self.federated_model = model_fn(args, dim_in, dim_hidden, dim_out)

        self.load_locked_model_initializations()

        self.param_count = sum([p.numel() for p in self.federated_model.parameters()])
        print("From Federated Learner - Parameters count is {}".format(self.param_count))

        self.federated_model_pretrain = copy.deepcopy(self.federated_model)

        self.participants = []

        for i, participant_train_loader in enumerate(self.participant_train_loaders):
            model = copy.deepcopy(self.federated_model)
            optimizer = optimizer_fn(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

            model_pretrain = copy.deepcopy(self.federated_model)
            optimizer_pretrain = optimizer_fn(model_pretrain.parameters(), lr=lr)
            scheduler_pretrain = torch.optim.lr_scheduler.ExponentialLR(optimizer_pretrain, gamma=gamma)

            standalone_model = copy.deepcopy(self.federated_model)
            standalone_optimizer = optimizer_fn(standalone_model.parameters(), lr=std_lr)
            standalone_scheduler = torch.optim.lr_scheduler.ExponentialLR(standalone_optimizer, gamma = gamma)

            dssgd_model = copy.deepcopy(self.federated_model)
            dssgd_optimizer = optimizer_fn(dssgd_model.parameters(), lr=dssgd_lr)
            dssgd_scheduler = torch.optim.lr_scheduler.ExponentialLR(dssgd_optimizer, gamma = gamma)

            fedavg_model = copy.deepcopy(self.federated_model)
            fedavg_optimizer = optimizer_fn(fedavg_model.parameters(), lr=fed_lr)
            fedavg_scheduler = torch.optim.lr_scheduler.ExponentialLR(fedavg_optimizer, gamma = gamma)
            
            participant = Participant(train_loader=participant_train_loader,
							model=model, optimizer=optimizer, scheduler=scheduler,
							model_pretrain=model_pretrain, optimizer_pretrain=optimizer_pretrain,scheduler_pretrain=scheduler_pretrain,
							pretraining_lr=self.args['pretraining_lr'],
							standalone_model=standalone_model, standalone_optimizer=standalone_optimizer, standalone_scheduler=standalone_scheduler,
							dssgd_model=dssgd_model, dssgd_optimizer=dssgd_optimizer,dssgd_scheduler=dssgd_scheduler,
							fedavg_model=fedavg_model, fedavg_optimizer=fedavg_optimizer, fedavg_scheduler=fedavg_scheduler,
							loss_fn=loss_fn, theta=theta,
							grad_clip=grad_clip, epoch_sample_size=epoch_sample_size,
							device=device,
							id=i,
							)
            self.participants.append(participant)
        return

    def train_locally(self, epochs, is_pretrain=False, save_gpu=False):
        if is_pretrain:
            for i, participant in enumerate(self.participants):
                participant.train(epochs, is_pretrain=is_pretrain)
            return
        
        self.filtered_updates = []
        self.filtered_updates_pretrain = []

        self.aggregated_gradient_updates = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]
        self.aggregated_gradient_updates_pretrain = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]

        participant_val_accs = []
        participant_val_accs_pretrain = []
        dssgd_val_accs = []
        fedavg_val_accs = []

        for i, participant in enumerate(self.participants):
            self.timestamp = time.time()

            model_before = copy.deepcopy(participant.model)
            dssgd_model_before = copy.deepcopy(participant.dssgd_model)
            model_pretrain_before = copy.deepcopy(participant.model_pretrain)
            fedavg_model_before = copy.deepcopy(participant.fedavg_model)

            participant.train(epochs, is_pretrain=is_pretrain, save_gpu=save_gpu)
            model_after = copy.deepcopy(participant.model)
            dssgd_model_after = copy.deepcopy(participant.dssgd_model)
            model_pretrain_after = copy.deepcopy(participant.model_pretrain)
            fedavg_model_after = copy.deepcopy(participant.fedavg_model)

            self.clock('participants local training')

            # recover the model before training for clipped grad update later
            participant.model.load_state_dict(model_before.state_dict())

            raw_grad_update = compute_grad_update(model_before, model_after, device=self.device)
            del model_before, model_after  # to free up memory immediately

            '''
            # clipped stats
            columns = ['','grad_mean','clipped_count', 'clipped_ratio']
            data_rows = []

            all_update_mod = torch.cat([update.data.view(-1).abs()for update in raw_grad_update])
            n_clipped = (all_update_mod > self.args['grad_clip']).sum().item()
            data_rows.append([ 'w/o pretrain: ', all_update_mod.mean().item(), n_clipped, torch.true_divide(n_clipped, len(all_update_mod)).item() ])
            '''

            clipped_grad_update = clip_gradient_update(raw_grad_update, self.args['grad_clip'])
            # add the clipped grad to local model
            add_update_to_model(participant.model, clipped_grad_update, device=self.device)
            filtered_grad_update = mask_grad_update_by_order(clipped_grad_update, mask_order=None, mask_percentile=participant.theta, mode=self.args['largest_criterion']) 

            fed_val_acc = self.one_on_one_evaluate(self.federated_model, participant.model, participant.id, filtered_grad_update, participant.theta)
            participant_val_accs.append(fed_val_acc)

            # minus the uploaded grad updates
            # add_update_to_model(participant.model, filtered_grad_update, weight= -1.0)

            # register this filtered_updates for later to removed
            # NOTE that we do not minus this update because this participant may not be reputable 
            # after evaluation, meaning it does not receive allocated_grad, so no need to minus its own
            self.filtered_updates.append(filtered_grad_update)

            self.clock('gradient clipping and filtering')

            # for with pretraining

            participant.model_pretrain.load_state_dict(model_pretrain_before.state_dict())
            raw_grad_update = compute_grad_update(model_pretrain_before, model_pretrain_after, device=self.device)				
            del model_pretrain_before, model_pretrain_after

            '''
            # clipped stats
            all_update_mod = torch.cat([update.data.view(-1).abs()for update in raw_grad_update])
            n_clipped = (all_update_mod > self.args['grad_clip']).sum().item()
            data_rows.append([ 'w pretrain: ', all_update_mod.mean().item(), n_clipped, torch.true_divide(n_clipped, len(all_update_mod)).item() ])
            '''

            clipped_grad_update = clip_gradient_update(raw_grad_update, self.args['grad_clip'])
            add_update_to_model(participant.model_pretrain, clipped_grad_update, device=self.device)
            filtered_grad_update = mask_grad_update_by_order(clipped_grad_update, mask_order=None, mask_percentile=participant.theta, mode=self.args['largest_criterion']) 			

            fed_val_acc = self.one_on_one_evaluate(self.federated_model_pretrain, participant.model_pretrain, participant.id, filtered_grad_update, participant.theta, is_pretrain=True)
            participant_val_accs_pretrain.append(fed_val_acc)

            # minus the uploaded grad updates
            # add_update_to_model(participant.model_pretrain, filtered_grad_update, weight= -1.0)
            self.filtered_updates_pretrain.append(filtered_grad_update)

            self.clock('gradient clipping and filtering for pretrain')


            # for DSSGD model

            dssgd_grad_update = compute_grad_update(dssgd_model_before, dssgd_model_after, device=self.device)
            del dssgd_model_before, dssgd_model_after  # to free up memory immediately

            '''
            # clipped stats
            all_update_mod = torch.cat([update.data.view(-1).abs()for update in dssgd_grad_update])
            n_clipped = (all_update_mod > self.args['grad_clip']).sum().item()
            data_rows.append([ 'dssgd:', all_update_mod.mean().item(), n_clipped, torch.true_divide(n_clipped, len(all_update_mod)).item() ])
            '''

            filtered_grad_update = mask_grad_update_by_order(clip_gradient_update(dssgd_grad_update, 0.001), mask_order=None, mask_percentile=participant.theta, mode=self.args['largest_criterion'])

            # this is executed in a fixed sequence, so the self.dssgd_model gets gradually updated and 'downloaded' by each participant
            participant.dssgd_model.load_state_dict(add_update_to_model(self.dssgd_model, filtered_grad_update).state_dict(), strict=False)

            dssgd_val_acc = evaluate(participant.dssgd_model, self.test_loaders[participant.id], self.device, verbose=False)[1]
            dssgd_val_accs.append(dssgd_val_acc)

            self.clock('server aggregation dssgd')


            # for fedavg model

            fedavg_grad_update = compute_grad_update(fedavg_model_before, fedavg_model_after, device=self.device)
            del fedavg_model_before, fedavg_model_after  # to free up memory immediately

            # this is executed in a fixed sequence, so the self.dssgd_model gets gradually updated and 'downloaded' by each participant
            # to follow fedavg method, incorporate the weighting via the shardsize

            weight = torch.div(self.shard_sizes[i], np.sum(self.shard_sizes))
            participant.fedavg_model.load_state_dict(add_update_to_model(self.fedavg_model, fedavg_grad_update, weight = weight).state_dict(), strict=False)

            fedavg_val_acc = evaluate(participant.fedavg_model, self.test_loaders[participant.id], self.device, verbose=False)[1]
            fedavg_val_accs.append(fedavg_val_acc)

            self.clock('server aggregation fedavg')

            '''
            # clipped stats
            print(pd.DataFrame(data=data_rows, columns=columns))
            '''
        return participant_val_accs, participant_val_accs_pretrain, dssgd_val_accs, fedavg_val_accs
    
    def train(self):
        self.reputations = torch.zeros((self.n_participants))
        self.reputations_pretrain = torch.zeros((self.n_participants))

        self.reputation_threshold_coef = self.args['reputation_threshold_coef'] if 'reputation_threshold_coef' in self.args else 1.0/3.0
        # init the reputation_th to be a 2/3 * 1/(len(R)) instead of 0
        self.reputation_threshold = compute_reputation_threshold(self.n_participants,self.args['split'], self.reputation_threshold_coef)
        self.reputation_threshold_pretrain = compute_reputation_threshold(self.n_participants,self.args['split'], self.reputation_threshold_coef)

        self.R = list(range(self.n_participants))
        self.R_pretrain = list(range(self.n_participants))

        fl_epochs = self.args['fl_epochs']
        device = self.args['device']
        fl_individual_epochs = self.args['fl_individual_epochs']

        self.alpha = self.args['alpha'] if 'alpha' in self.args else 5
        self.reputation_fade = self.args['reputation_fade'] if 'reputation_fade' in self.args else 1

        self.performance_dict['shard_sizes'] = self.shard_sizes
        self.performance_dict_pretrain['shard_sizes'] = self.shard_sizes

        # print("Start local pretraining ")
        self.timestamp = time.time()

        self.train_locally(self.args['pretrain_epochs'], is_pretrain=True, save_gpu=self.save_gpu)

        self.clock('pretraining')

        self.participant_model_test_accs_before = self.evaluate_participants_performance(self.test_loaders)
        self.performance_dict['participant_model_test_accs_before'] = self.participant_model_test_accs_before

        self.participant_model_test_accs_before_w_pretrain = self.evaluate_participants_performance(self.test_loaders, mode='pretrain')
        self.performance_dict_pretrain['participant_model_test_accs_before'] = self.participant_model_test_accs_before_w_pretrain

        self.dssgd_model = copy.deepcopy(self.federated_model).to(device)
        # each participant needs a dssgd model to compute final fairness

        self.fedavg_model = copy.deepcopy(self.federated_model).to(device)

        # _, federated_val_acc = evaluate(self.federated_model, self.valid_loader, device, verbose=False)
        # print("CFFL server model validation accuracy : {:.4%}".format(federated_val_acc))

        _, federated_test_acc = evaluate(self.federated_model, self.test_loader, device, verbose=False)
        print("CFFL server model test accuracy : {:.4%}".format(federated_test_acc))


        self.clock('evaluation after pretraining')

        # print("\nStart federated learning \n")
        for epoch in range(fl_epochs):
            # 1. training locally
            participant_val_accs, participant_val_accs_pretrain, dssgd_val_accs, fedavg_val_accs = self.train_locally(fl_individual_epochs,save_gpu=self.save_gpu)

            if 'alpha_decay' in self.args and self.args['alpha_decay']:
                alpha = self.alpha * (1 + epoch/fl_epochs)
            else:
                alpha = self.alpha

			# 2. update the reputations and reputation_threshold
			# and update the reputable participants set
            self.reputations, self.reputation_threshold, self.R  = compute_reputations_sinh(self.reputations, self.reputation_threshold, self.R, participant_val_accs, alpha=alpha, reputation_fade=self.reputation_fade, split=self.args['split'], reputation_threshold_coef=self.reputation_threshold_coef)
            self.reputations_pretrain, self.reputation_threshold_pretrain, self.R_pretrain = compute_reputations_sinh(self.reputations_pretrain, self.reputation_threshold_pretrain, self.R_pretrain, participant_val_accs_pretrain, alpha=alpha, reputation_fade=self.reputation_fade, split=self.args['split'], reputation_threshold_coef=self.reputation_threshold_coef)

            self.clock('reputation updates')


			# 3. aggregate the gradients and update the federated model
            self.aggregate_gradients_and_update_federated_model()
            self.clock('aggregate gradients and update FL model')


            # 4. gradient downloads and uploads according to reputations and thetas
            self.assign_updates_with_filter()
            self.clock('assign updates')

            # update the performance dict as log
            if (epoch+1) % 20 == 0:
                print()
                print('Epoch {}:'.format(epoch + 1))
                print("Without pretraining:")
                print("Reputations: {}, Reputation threshold: {}.".format(np.around(self.reputations.tolist(), 3),
                    np.around(self.reputation_threshold.item(), 3)))
                print("Reputable participants: ", self.R)
                print()
                print("With pretraining:")
                print("Reputations: {}, Reputation threshold: {}.".format(np.around(self.reputations_pretrain.tolist(), 3),
                    np.around(self.reputation_threshold_pretrain.item(), 3))) 
                print("Reputable participants: ", self.R_pretrain)
                print()


            self.performance_summary(to_print=((epoch+1)%20==0))


            self.performance_dict['dssgd_val_accs'].append(dssgd_val_accs)
            self.performance_dict_pretrain['dssgd_val_accs'].append(dssgd_val_accs)
            self.performance_dict['fedavg_val_accs'].append(fedavg_val_accs)
            self.performance_dict_pretrain['fedavg_val_accs'].append(fedavg_val_accs)

            self.performance_dict['reputations'].append(self.reputations)
            self.performance_dict['reputation_threshold'].append(self.reputation_threshold)

            self.performance_dict_pretrain['reputations'].append(self.reputations_pretrain)
            self.performance_dict_pretrain['reputation_threshold'].append(self.reputation_threshold_pretrain)
            # print()
            self.clock('performance update')

        total_seconds = 0
        for key, value in self.time_dict.items():
            # print(key, value)
            total_seconds += value
            self.time_dict[key] = round(value, 3)
        print("Total execution time (seconds)", total_seconds)
        print('Runtime for individual components in seconds.')
        print('-----')
        print(json.dumps(self.time_dict))
        print('-----')

        self.convert_tensors_in_dicts()
        return
    
    def load_locked_model_initializations(self, dirname='initialized_models'):
        import os
        models_dir = os.path.join(dirname, self.args['dataset'])
        os.makedirs(models_dir, exist_ok=True)
        model_name =  self.federated_model.__class__.__name__
        model_path = os.path.join(models_dir, model_name)
        if os.path.isfile(model_path):
            try:
                self.federated_model.load_state_dict(torch.load(model_path))
                print("Successfully loaded the fixed initialization of {} for the {} dataset.".format(model_name, self.args['dataset']))
            except Exception as e:
                print(str(e))
                print("Due to the above error, saving {} to overwrite the existing file.".format(model_name))
                torch.save(self.federated_model.state_dict(), model_path)
        else:
            print("Saving a fresh {} model for {}.".format(model_name, self.args['dataset']))
            torch.save(self.federated_model.state_dict(), model_path)
    
    def one_on_one_evaluate(self, federated_model, participant_model, participant_id, filtered_grad_update, theta, is_pretrain=False):
        if theta == 1 and not is_pretrain:
            fed_val_acc = evaluate(participant_model, self.test_loaders[participant_id], self.device, verbose=False)[1]
        else:
            model_to_eval = copy.deepcopy(federated_model)
            add_update_to_model(model_to_eval, filtered_grad_update, device=self.device)
            fed_val_acc = evaluate(model_to_eval, self.test_loaders[participant_id], self.device, verbose=False)[1]
            del model_to_eval
        return fed_val_acc
    
    def aggregate_gradients_and_update_federated_model(self, eta=1):
        """
        collect all the gradients and aggregate them in the specified way
        sum: direct sum all the gradients from all participants
        reputation-sum: reputation-weighted sum over all gradients
        mean: fedavg
            NOTE: in the original fedavg, the weights are relative shard_size, i.e.  weight = (num_samples by participant-i) / (total num_samples of all participants).
            We follow this logic for 'random', 'equal', 'powerlaw' split of the datasets.

            We follow a modified logic for 'classimbalance' split for the datasets. This is because in 'classimbalance', we enforce that each participant has the same
            number of samples, but different classes of samples. As a result, the original fedavg logic is clearly not very robust.


        Arguments: 
        eta: is used as a way to manually introduce complex learning rate or lr scheduler.Default:1

        """
        self.aggregated_gradient_updates = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]
        self.aggregated_gradient_updates_pretrain = [torch.zeros(param.shape).to(self.device) for param in self.federated_model.parameters()]

        for i in self.R:
            filtered_grad_update = self.filtered_updates[i]
            if self.args['aggregate_mode'] == 'sum':
                weight = 1.0
            elif self.args['aggregate_mode'] == 'reputation-sum':
                weight = self.reputations[i]
            else: # default average
                if self.args['split'] != 'classimbalance':
                    weight = self.shard_sizes[i] * 1. / sum(self.shard_sizes)

                else:
                    assert self.args['dataset'] in ['mnist', 'cifar10'], "Fedavg and classimbalance Not supported for this dataset {}".format(self.args['dataset'])
                    # currently only for cifar10 and mnist, so a total of 10 classes
                    n_classes = 10
                    class_sizes = np.linspace(1, n_classes, self.n_participants, dtype='int')
                    weight = class_sizes[i] / n_classes

            add_gradient_updates(self.aggregated_gradient_updates, filtered_grad_update, weight)

        add_update_to_model(self.federated_model, self.aggregated_gradient_updates, weight=eta, device=self.device)
        # self.federated_val_acc = evaluate(self.federated_model, self.valid_loader, device=self.device, verbose=False)[1]

        for i in self.R_pretrain:
            filtered_grad_update = self.filtered_updates_pretrain[i]
            if self.args['aggregate_mode'] == 'sum':
                weight = 1.0
            elif self.args['aggregate_mode'] == 'reputation-sum':
                weight = self.reputations_pretrain[i]
            else: # default fedavg
                if self.args['split'] != 'classimbalance':
                    weight = self.shard_sizes[i] * 1. / sum(self.shard_sizes)

                else:
                    assert self.args['dataset'] in ['mnist', 'cifar10'], "Fedavg and classimbalance Not supported for this dataset {}".format(self.args['dataset'])
                    # currently only for cifar10 and mnist, so a total of 10 classes
                    n_classes = 10
                    class_sizes = np.linspace(1, n_classes, self.n_participants, dtype='int')
                    weight = class_sizes[i] / n_classes
            add_gradient_updates(self.aggregated_gradient_updates_pretrain, filtered_grad_update, weight)

        add_update_to_model(self.federated_model_pretrain, self.aggregated_gradient_updates_pretrain, weight=eta, device=self.device)
        # self.federated_val_acc_pretrain = evaluate(self.federated_model_pretrain, self.valid_loader, device=self.device, verbose=False)[1]

    def assign_updates_with_filter(self):
            """
            download the largest magnitude updates <reputations[i] * num_param> from the server
            and filter out its own updates in the local model
            and apply to its local model
            """
            if self.args['aggregate_mode'] == 'mean':
                if self.args['split']!='classimbalance':
                    weights = torch.div(self.shard_sizes , max(self.shard_sizes) )
                else:
                    n_classes=10
                    class_sizes = np.linspace(1, n_classes, self.n_participants, dtype='int')
                    weights = torch.div(torch.tensor(class_sizes).float(), max(class_sizes) )
            else:
                weights = torch.ones(self.n_participants)  

            # default download mode is 'topk'
            download = 'topk' if 'download' not in self.args else self.args['download']

            if self.args['largest_criterion'] == 'all':

                # preprocess to get the topk largest values for all (only need sort it once for the highest reputation)
                # no pretrain
                absolute_values = torch.cat([update.data.view(-1).abs() for update in self.aggregated_gradient_updates])

                if download == 'random':
                    random_permuted_indices = torch.randperm(len(absolute_values))
                else:
                    topk, _ = torch.topk(absolute_values, int(len(absolute_values))) 

                # pretrain
                absolute_values = torch.cat([update.data.view(-1).abs() for update in self.aggregated_gradient_updates_pretrain])
                if download == 'random':
                    random_permuted_pretrain_indices = torch.randperm(len(absolute_values)) if download == 'random' else None
                else:
                    topk_pretrain, _ = torch.topk(absolute_values, int(len(absolute_values))) 
                    del _

                del absolute_values

                
                for i, participant in enumerate(self.participants):

                    # no pretrain
                    if i in self.R:
                        agg_grad_update = copy.deepcopy(self.aggregated_gradient_updates)
                        if self.args['split']!='classimbalance':
                            num_downloads  = int(self.reputations[i]*1. / max(self.reputations) *self.shard_sizes[i] *1. / max(self.shard_sizes) * participant.param_count)
                        else:
                            n_classes = 10
                            class_sizes = np.linspace(1, n_classes, self.n_participants, dtype='int')
                            num_downloads  = int(self.reputations[i]*1. / max(self.reputations) *class_sizes[i] / n_classes * participant.param_count)
                        
                        if download == 'random':
                            assert random_permuted_indices is not None, "Uninitialized <random_permuted_indices>"
                            allocated_grad = mask_grad_update_by_indices(agg_grad_update, indices=random_permuted_indices[:num_downloads])
                        else:
                            allocated_grad = mask_grad_update_by_magnitude(agg_grad_update, topk[num_downloads-1])
                        
                        add_update_to_model(participant.model, allocated_grad)
                        add_update_to_model(participant.model, self.filtered_updates[i], weight=-weights[i])
                        
                    # with pretrain
                    if i in self.R_pretrain:
                        agg_grad_update = copy.deepcopy(self.aggregated_gradient_updates_pretrain)
                        if self.args['split']!='classimbalance':
                            num_downloads  = int(self.reputations_pretrain[i]*1. / max(self.reputations_pretrain) *self.shard_sizes[i] *1. / max(self.shard_sizes) * participant.param_count)
                        else:
                            n_classes = 10
                            class_sizes = np.linspace(1, n_classes, self.n_participants, dtype='int')
                            num_downloads  = int(self.reputations_pretrain[i]*1. / max(self.reputations_pretrain) *class_sizes[i] / n_classes * participant.param_count)
                        
                        if download == 'random':
                            assert random_permuted_pretrain_indices is not None, "Uninitialized <random_permuted_pretrain_indices>"
                            allocated_grad = mask_grad_update_by_indices(agg_grad_update, indices=random_permuted_pretrain_indices[:num_downloads])
                        else:				
                            allocated_grad = mask_grad_update_by_magnitude(agg_grad_update, topk_pretrain[num_downloads-1])

                        add_update_to_model(participant.model_pretrain, allocated_grad)
                        add_update_to_model(participant.model_pretrain, self.filtered_updates_pretrain[i], weight=-weights[i])

            elif self.args['largest_criterion'] == 'layer':
                
                for i, participant in enumerate(self.participants):
                    # no pretrain
                    if i in self.R:
                        allocated_grad = mask_grad_update_by_order(self.aggregated_gradient_updates, mask_order=None, mask_percentile=self.reputations[i], mode='layer')
                        add_update_to_model(participant.model, allocated_grad)
                        add_update_to_model(participant.model, self.filtered_updates[i], weight=-weights[i])

                    # with pretrain
                    if i in self.R_pretrain:
                        allocated_grad = mask_grad_update_by_order(self.aggregated_gradient_updates_pretrain, mask_order=None, mask_percentile=self.reputations_pretrain[i], mode='layer')
                        add_update_to_model(participant.model_pretrain, allocated_grad)
                        add_update_to_model(participant.model_pretrain, self.filtered_updates_pretrain[i], weight=-weights[i])
            return
    
    def performance_summary(self, to_print=False):
        self.dssgd_models_test_accs = self.evaluate_participants_performance(self.test_loader, mode='dssgd')
        self.fedavg_models_test_accs = self.evaluate_participants_performance(self.test_loader, mode='fedavg')
        self.participant_standalone_test_accs = self.evaluate_participants_performance(self.test_loader, mode='standalone')
        self.cffl_test_accs = self.evaluate_participants_performance(self.test_loaders)
        self.cffl_test_accs_w_pretrain = self.evaluate_participants_performance(self.test_loaders, mode='pretrain')

        # no pretrain
        self.performance_dict['DSSGD_model_test_accs'].append(self.dssgd_models_test_accs)
        self.performance_dict['fedavg_model_test_accs'].append(self.fedavg_models_test_accs)
        self.performance_dict['participant_standalone_test_accs'].append(self.participant_standalone_test_accs)
        self.performance_dict['cffl_test_accs'].append(self.cffl_test_accs)

        # with pretrain
        self.performance_dict_pretrain['DSSGD_model_test_accs'].append(self.dssgd_models_test_accs)
        self.performance_dict_pretrain['fedavg_model_test_accs'].append(self.fedavg_models_test_accs)
        self.performance_dict_pretrain['participant_standalone_test_accs'].append(self.participant_standalone_test_accs)
        self.performance_dict_pretrain['cffl_test_accs'].append(self.cffl_test_accs_w_pretrain)

        if to_print:
            print('Below are testset  accuracies: ---')
            print('Participants standalone accuracies: ', ["{:.3%}".format(acc_std) for acc_std in self.participant_standalone_test_accs])
            print('Participants DSSGD      accuracies: ', ["{:.3%}".format(dssgd_acc) for dssgd_acc in self.dssgd_models_test_accs])
            print('Participants Fedavg     accuracies: ', ["{:.3%}".format(fedavg_acc) for fedavg_acc in self.fedavg_models_test_accs])
            print()
            print('Participants before     accuracies: ', ["{:.3%}".format(acc_b4) for acc_b4 in self.participant_model_test_accs_before])
            print('Participants CFFL       accuracies: ', ["{:.3%}".format(acc_aft) for acc_aft in self.cffl_test_accs])
            print()
            print('Participants before pr  accuracies: ', ["{:.3%}".format(acc_b4) for acc_b4 in self.participant_model_test_accs_before_w_pretrain])
            print('Participants CFFL pret  accuracies: ', ["{:.3%}".format(acc_aft) for acc_aft in self.cffl_test_accs_w_pretrain])

        return

    def convert_tensors_in_dicts(self):

        for key, value in self.performance_dict.items():
            if isinstance(value[0], torch.Tensor):
                self.performance_dict[key] = torch.stack(value).tolist()
            elif isinstance(value[0], list):
                self.performance_dict[key] = [torch.stack(v).tolist() for v in value]


        for key, value in self.performance_dict_pretrain.items():
            if isinstance(value[0], torch.Tensor):
                self.performance_dict_pretrain[key] = torch.stack(value).tolist()
            elif isinstance(value[0], list):
                self.performance_dict_pretrain[key] = [torch.stack(v).tolist() for v in value]

    def get_fairness_analysis(self):
        print("Predictive Performance and Fairness results: ")
        participant_thetas = [participant.theta for participant in self.participants]

        print('Participant reputations :', self.reputations.tolist())
        print('Number of reputable participants: ', len(self.R))

        print('Participant reputations pretrain:', self.reputations_pretrain.tolist())
        print('Number of reputable participants with pretrain: ', len(self.R_pretrain))


        # no pretrain
        participant_standalone_test_accs = self.performance_dict['participant_standalone_test_accs'][-1]
        DSSGD_model_test_accs = self.performance_dict['DSSGD_model_test_accs'][-1]
        fedavg_model_test_accs = self.performance_dict['fedavg_model_test_accs'][-1]
        cffl_test_accs = self.performance_dict['cffl_test_accs'][-1]

        from scipy.stats import pearsonr
        corrs = pearsonr(participant_standalone_test_accs, DSSGD_model_test_accs)
        self.performance_dict['standalone_vs_rrdssgd'].append(corrs[0])

        corrs = pearsonr(participant_standalone_test_accs, cffl_test_accs)
        self.performance_dict['standalone_vs_final'].append(corrs[0])
        
        corrs = pearsonr(participant_standalone_test_accs, fedavg_model_test_accs)
        self.performance_dict['standalone_vs_fedavg'].append(corrs[0])

        self.performance_dict['CFFL_best_participant'] = max(cffl_test_accs)
        best_participant_id = np.argmax(cffl_test_accs)
        self.performance_dict['standalone_best_participant'] = participant_standalone_test_accs[best_participant_id]
        self.performance_dict['rr_dssgd_best'] = DSSGD_model_test_accs[best_participant_id]
        self.performance_dict['rr_fedavg_best'] = fedavg_model_test_accs[best_participant_id]

        # with pretrain
        participant_standalone_test_accs = self.performance_dict_pretrain['participant_standalone_test_accs'][-1]
        DSSGD_model_test_accs = self.performance_dict_pretrain['DSSGD_model_test_accs'][-1]
        fedavg_model_test_accs = self.performance_dict_pretrain['fedavg_model_test_accs'][-1]
        cffl_test_accs = self.performance_dict_pretrain['cffl_test_accs'][-1]

        corrs = pearsonr(participant_standalone_test_accs, DSSGD_model_test_accs)
        self.performance_dict_pretrain['standalone_vs_rrdssgd'].append(corrs[0])

        corrs = pearsonr(participant_standalone_test_accs, cffl_test_accs)
        self.performance_dict_pretrain['standalone_vs_final'].append(corrs[0])

        corrs = pearsonr(participant_standalone_test_accs, fedavg_model_test_accs)
        self.performance_dict_pretrain['standalone_vs_fedavg'].append(corrs[0])

        self.performance_dict_pretrain['CFFL_best_participant'] = max(cffl_test_accs)
        best_participant_id = np.argmax(cffl_test_accs)

        self.performance_dict_pretrain['standalone_best_participant'] = participant_standalone_test_accs[best_participant_id]
        self.performance_dict_pretrain['rr_dssgd_best'] = DSSGD_model_test_accs[best_participant_id]
        self.performance_dict_pretrain['rr_fedavg_best'] = fedavg_model_test_accs[best_participant_id]


        keys = ['standalone_best_participant', 'CFFL_best_participant', 'rr_dssgd_best', 'rr_fedavg_best',
            'standalone_vs_rrdssgd', 'standalone_vs_final', 'standalone_vs_fedavg']
        print("----Predictive performance results without pretrain")
        for key in keys:
            print(key, ' - ', np.around(self.performance_dict[key], 3))			
        print("----")

        print("----Predictive performance results with pretrain")
        for key in keys:
            print(key, ' - ', np.around(self.performance_dict_pretrain[key], 3))
        print("----")

        return

    def evaluate_participants_performance(self, eval_loader, mode=None):
        device = self.args['device']
        if mode == 'standalone':
            return [evaluate(participant.standalone_model, eval_loader, device, verbose=False)[1] for participant in self.participants]
        elif mode == 'dssgd':
            return [evaluate(participant.dssgd_model, eval_loader, device, verbose=False)[1] for participant in self.participants]
        elif mode == 'pretrain':
            return [evaluate(participant.model_pretrain, eval_loader[participant.id], device, verbose=False)[1] for participant in self.participants]
        elif mode == 'fedavg':
            return [evaluate(participant.fedavg_model, eval_loader, device, verbose=False)[1] for participant in self.participants]
        else:
            return [evaluate(participant.model, eval_loader[participant.id], device, verbose=False)[1] for participant in self.participants]
    def clock(self, key):
        self.timestamp_  = time.time()
        self.time_dict[key] += self.timestamp_ - self.timestamp
        self.timestamp = self.timestamp_


    def update_reputations(self, participant_val_accs, participant_val_accs_pretrain):
        self.reputations, self.reputation_threshold, self.R  = compute_reputations_sinh(self.reputations, self.reputation_threshold, self.R, participant_val_accs, alpha=self.args['alpha'], split=self.args['split'], reputation_threshold_coef=self.reputation_threshold_coef)
        self.reputations_pretrain, self.reputation_threshold_pretrain, self.R_pretrain = compute_reputations_sinh(self.reputations_pretrain, self.reputation_threshold_pretrain, self.R_pretrain, participant_val_accs_pretrain, alpha=self.args['alpha'],split=self.args['split'],reputation_threshold_coef=self.reputation_threshold_coef)

def compute_reputations_sinh(reputations, reputation_threshold, R, val_accs, alpha=5, reputation_fade=1, split='powerlaw', reputation_threshold_coef=1.0/3.0):
	# print('alpha used is :', alpha, ' current reputations are : ', reputations, ' current threshold: ', reputation_threshold)
	R_size = len(R)
	total_val_accs = sum([val_accs[i] for i in R])
	for i in R:
		reputation_epoch = val_accs[i] / total_val_accs

		if reputation_fade == 1:
			reputations[i] = reputations[i] * 0.8 + reputation_epoch * 0.2
		else:
			reputations[i] = (reputations[i] + reputation_epoch) * 0.5

	reputations = torch.sinh(alpha * reputations)

	# normalize among the reputable participants
	reputations /= reputations.sum().float()

	# update reputable participants
	R = [i for i in R if reputations[i] >= reputation_threshold]

	# isolate the non-reputable participants by setting their reputations to 0	
	for i in range(len(reputations)):
		if reputations[i] < reputation_threshold:
			reputations[i] = 0

	if R_size != len(R):
		# normalize among the reputable participants
		reputations /= reputations.sum().float()
		reputation_threshold = compute_reputation_threshold(len(R), split, reputation_threshold_coef)

		# print('------')
		# print("The reputable participant set size has changed from {} to {}.".format(R_size, len(R)))
		# print("The new reputation_threshold is {}, and the reputations are {}.".format(round(reputation_threshold.item(),3), np.around(reputations.tolist(), 3)))
		# print('------')

	return reputations, reputation_threshold, R

def clip_gradient_update(grad_update, grad_clip):
    """
    Return a copy of clipped grad update 

    """
    return [torch.clamp(param.data, min=-grad_clip, max=grad_clip) for param in grad_update]


def mask_grad_update_by_order(grad_update, mask_order, mask_percentile=None, mode='all'):

    if mode == 'all':
        # mask all but the largest <mask_order> updates (by magnitude) to zero
        all_update_mod = torch.cat([update.data.view(-1).abs()
                                    for update in grad_update])
        if not mask_order and mask_percentile:
            mask_order = int(len(all_update_mod) * mask_percentile)
        
        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float('inf'))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

    elif mode == 'layer': # layer wise largest-values criterion
        grad_update = copy.deepcopy(grad_update)

        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)

            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update

def mask_grad_update_by_indices(grad_update, indices=None):
    """
    Mask the grad.data to be 0, if the position is not in the list of indices
    If indicies is empty, mask nothing.
    
    Arguments: 
    grad_update: as in the shape of the model parameters. A list of tensors.
    indices: a tensor of integers, corresponding to the specific individual scalar values in the grad_update, 
    as if the entire grad_update is flattened.

    e.g. 
    grad_update = [[1, 2, 3], [3, 2, 1]]
    indices = [4, 5]
    returning masked grad_update = [[0, 0, 0], [0, 2, 1]]
    """

    grad_update = copy.deepcopy(grad_update)
    if indices is None or len(indices)==0: return grad_update

    #flatten and unflatten
    flattened = torch.cat([update.data.view(-1) for update in grad_update])	
    masked = torch.zeros_like(torch.arange(len(flattened)), device=flattened.device).float()
    masked.data[indices] = flattened.data[indices]

    pointer = 0
    for m, update in enumerate(grad_update):
        size_of_update = torch.prod(torch.tensor(update.shape)).long()
        grad_update[m].data = masked[pointer: pointer + size_of_update].reshape(update.shape)
        pointer += size_of_update
    return grad_update

def compute_reputation_threshold(R_size, split, coef=1.0/3.0):
    if split=='classimbalance':
        return torch.clamp(1./6 * torch.div(1., R_size), min=0, max=1).float()
    else:
        return torch.clamp(coef * torch.div(1., R_size), min=0, max=1).float()
    
use_cuda = True
cuda_available = torch.cuda.is_available()

def update_gpu(args):
	if 'cuda' in str(args['device']):
		args['device'] = torch.device('cuda:{}'.format(args['gpu']))
	if torch.cuda.device_count() > 0:
		args['device_ids'] = [device_id for device_id in range(torch.cuda.device_count())]
	else:
		args['device_ids'] = []

def init_deterministic():
	# call init_deterministic() in each run_experiments function call

	torch.manual_seed(1234)
	np.random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(1234)

def run_experiments(args, repeat=5, logs_dir='logs'):
    update_gpu(args)
    init_deterministic()
    model_name = str(args['model_fn']).split('.')[-1][:-2]
    subdir = "{}_p{}_e{}-{}-{}_b{}_size{}_lr{}_theta{}_{}runs_{}_a{}_fr{}_{}".format(args['dataset']+'@'+args['split'],args['n_participants'], 
                            args['pretrain_epochs'], args['fl_epochs'], args['fl_individual_epochs'],
                            args['batch_size'], args['sample_size_cap'], args['lr'], args['theta'],
                            str(repeat), args['aggregate_mode'], args['alpha'],args['n_freeriders'], model_name,
                            )
    logdir = os.path.join(logs_dir, subdir)
    os.makedirs(logdir, exist_ok=True)

    if 'complete.txt' in os.listdir(logdir):
        return

    with open(os.path.join(logdir,'settings_dict.txt'), 'w') as file:
        [file.write(key + ' : ' + str(value) + '\n') for key,value in args.items()]

    log = open(os.path.join(logdir, 'log'), "w")
    sys.stdout = log
    print("Experimental settings are: ", args, '\n')

    performance_dicts = []
    performance_dicts_pretrain = []

    for i in range(repeat):
        print()
        print("Experiment : No.{}/{}".format(str(i+1) ,str(repeat)))
        # data_prep = Data_Prepper(args['dataset'], train_batch_size=args['batch_size'], sample_size_cap=args['sample_size_cap'], train_val_split_ratio=args['train_val_split_ratio'])
        federated_learner = Federated_Learner(args)

        # train
        federated_learner.train()
        # analyze
        federated_learner.get_fairness_analysis()

        performance_dicts.append(federated_learner.performance_dict)
        
        with open(os.path.join(logdir, 'performance_dict.log'), 'a') as log:
            log.write(json.dumps(federated_learner.performance_dict))
            log.write('\n')

        performance_dicts_pretrain.append(federated_learner.performance_dict_pretrain)
        with open(os.path.join(logdir, 'performance_dict_pretrain.log'), 'a') as log:
            log.write(json.dumps(federated_learner.performance_dict_pretrain))
            log.write('\n')


    write_aggregate_dict(performance_dicts, os.path.join(logdir, 'aggregate_dict.txt'))
    write_aggregate_dict(performance_dicts_pretrain, os.path.join(logdir, 'aggregate_dict_pretrain.txt'))

    with open(os.path.join(logdir, 'complete.txt'), 'w') as file:
        file.write('complete')

    return

def write_aggregate_dict(performance_dicts, filename):

	keys = ['standalone_best_participant', 'CFFL_best_participant', 'rr_dssgd_best', 'rr_fedavg_best',
		'standalone_vs_rrdssgd', 'standalone_vs_final', 'standalone_vs_fedavg']

	aggregate_dict = {}
	for key in keys:
		list_of_performance = [performance_dict[key] for performance_dict in performance_dicts]
		aggregate_dict[key] = np.array(list_of_performance).tolist()
		aggregate_dict[key +'_mean'] = np.mean(aggregate_dict[key], axis=0).tolist()
		aggregate_dict[key +'_std'] = np.std(aggregate_dict[key], axis=0).tolist()

		'''
		print(key, aggregate_dict[key])
		print(key +'_mean', aggregate_dict[key +'_mean'])
			# result_list.append(res)
		print(key +'_std', aggregate_dict[key +'_std'])
		'''
	with open(filename, 'w') as file:
		file.write

def run_experiments_full(experiment_args, repeat=1):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')
    experiment_dir = 'Experiments_{}'.format(st)
    experiment_dir = os.path.join("{}".format(experiment_args[0]['dataset']), experiment_dir)

    os.makedirs(experiment_dir, exist_ok=True)
	

    for args in experiment_args:
        run_experiments(args, repeat, experiment_dir)

    try:
        examine(experiment_dir)
    except Exception as e:
        print(str(e))
    return

def examine(dirname):
	# experiment_results = plot_convergence(dirname)
	# collate_pngs(dirname)
	fair_df, perf_df = collect_and_compile_performance(dirname)

def parse(dirname, folder):
	setup = {}

	try:
		with open(os.path.join(dirname,folder, 'settings_dict.txt'), 'r') as settings:
			lines = settings.readlines()
		for line in lines:
			key, value = line.split(':', 1)
			setup[key.strip()] = value.strip()
	except Exception as e:
		print(str(e))
		print('Error reading from the settings_dict.txt, reading from the name of the folder')

	if setup:
		setup['model'] = setup['model_fn'].split('.')[-1][:-2]
		setup['P'] = int(setup['n_participants'])
		setup['E'] = int(setup['fl_individual_epochs'])
		setup['Communication Rounds'] = int(setup['fl_epochs'])
		setup['size'] = int(setup['sample_size_cap'])
		setup['B'] = int(setup['batch_size'])
		setup['lr'] = float(setup['lr'])
		setup['alpha'] = float(setup['alpha'])
		setup['theta'] = float(setup['theta'])
		setup['n_freeriders'] = int(setup['n_freeriders'])
		setup['pretrain_epochs'] = int(setup['pretrain_epochs'])
	else:
		param = folder.split('_')

		setup['dataset'] = 'adult'
		if 'MLP_Net' in folder or 'CNN_Net' in folder or 'classimbalance' in folder:
			setup['dataset'] = 'mnist'
		setup['model'] = param[-1]
		if setup['model'] == 'LogisticRegression':
			setup['model'] = 'LR'

		setup['split'] = param[0]
		setup['P'] = int(param[1][1:])
		setup['pretrain_epochs'] = int(param[2].split('-')[0][1:])
		setup['Communication Rounds'] = int(param[2].split('-')[1])
		setup['theta'] = float(param[6].replace('theta', ''))
		setup['E'] = int(param[2].split('-')[2])
		setup['B'] = int(param[3][1:])
		setup['size'] = int(param[4][4:])
		setup['lr'] = float(param[5][2:])
		setup['alpha'] = int(folder[folder.find('_a')+2])

	return setup

def collect_and_compile_performance(dirname):

	fairness_rows = []
	performance_rows = []
	for folder in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, folder)) or not 'complete.txt' in os.listdir(os.path.join(dirname, folder)):
			continue

		setup = parse(dirname, folder)
		n_participants = setup['P']
		fl_epochs = setup['Communication Rounds']
		theta = setup['theta']

		try:
			with open(os.path.join(dirname, folder, 'aggregate_dict.txt')) as dict_log:
				aggregate_dict = json.loads(dict_log.read())

			with open(os.path.join(dirname, folder, 'aggregate_dict_pretrain.txt')) as dict_log:
				aggregate_dict_pretrain = json.loads(dict_log.read())

			f_data_row = ['P' + str(n_participants) + '_' + str(theta)] + [aggregate_dict[f_key][0] for f_key in fairness_keys]
			f_data_row.append(aggregate_dict_pretrain['standalone_vs_final_mean'][0])

			p_data_row = ['P' + str(n_participants) + '_' + str(theta)] + [aggregate_dict['rr_fedavg_best'][0],
																	aggregate_dict['rr_dssgd_best'][0], 
																	aggregate_dict['standalone_best_participant'][0], 
																	aggregate_dict['CFFL_best_participant'][0], 
																	aggregate_dict_pretrain['CFFL_best_participant'][0]
																	]

			fairness_rows.append(f_data_row)
			performance_rows.append(p_data_row)
		except Exception as e:
			print("Compiling fairness and accuracy csvs")
			print(e)

	shorthand_f_keys = ['Fedavg', 'DSSGD', 'CFFL', 'CFFL pretrain']
	fair_df = pd.DataFrame(fairness_rows, columns=[' '] + shorthand_f_keys).set_index(' ')
	fair_df = fair_df.sort_values(' ')
	print(fair_df.to_markdown())
	
	print(os.path.join(dirname, 'fairness.csv'))
	fair_df.to_csv( os.path.join(dirname, 'fairness.csv'))

	shorthand_p_keys = ['Fedavg', 'DSSGD', 'Standalone', 'CFFL', 'CFFL pretrain']
	pd.options.display.float_format = '{:,.2f}'.format
	perf_df = pd.DataFrame(performance_rows, columns=[' '] + shorthand_p_keys).set_index(' ').T
	perf_df = perf_df[sorted(perf_df.columns)]
	print(perf_df.to_markdown())
	perf_df.to_csv( os.path.join(dirname, 'performance.csv'))

	return fair_df, perf_df

fairness_keys = [
		'standalone_vs_fedavg_mean',
		'standalone_vs_rrdssgd_mean',
		'standalone_vs_final_mean',
		]

mnist_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'curetsr',
	'sample_size_cap': 3000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'
	'theta': 1,  # privacy level -> at most (theta * num_of_parameters) updates
	'batch_size' : 16, 
	'train_val_split_ratio': 0.9,
	'alpha': 2, #standard is 5
	'epoch_sample_size':float("Inf"),
	'n_freeriders': 0,


	# model parameters
	'model_fn': MLP_for_CFFL,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'pretraining_lr' : .01, # only used during pretraining for CFFL models, no decay
	'dssgd_lr': 1e-2, # used for dssgd model, no decay
	'std_lr': 1e-2,
	'fed_lr': 2e-2,
	'lr': 0.1,
	'grad_clip': 1e-2,
	'gamma': 0.977,   #0.955**100 ~= 0.01
	'reputation_threshold_coef': 1.0/3.0,


	# training parameters
	'pretrain_epochs': 5,
	'fl_epochs': 75,
	'fl_individual_epochs': 1,
	'aggregate_mode':'sum',  # 'mean', 'sum'
	'largest_criterion': 'all', #'layer'
	'download':'random',
	'reputation_fade':1,
	'alpha_decay':True,
    'round': 0
}


if __name__ == '__main__':
	# init steps	
    r = 2
    experiment_args = []	
    args = copy.deepcopy(mnist_args)

    for n_participants, sample_size_cap in [[5, 3000]]:
        args['n_participants'] = n_participants
        args['sample_size_cap'] = sample_size_cap

        with open(f'data_files/final/train_curetsr_{r}.pkl', 'rb') as f:  
            args['train_loaders'] = pickle.load(f)
    
        with open(f'data_files/final/test_curetsr_{r}.pkl', 'rb') as f:  
            args['test_loaders'] = pickle.load(f)

        with open(f'data_files/final/testloader_curetsr_{r}.pkl', 'rb') as f:  
            args['test_loader'] = pickle.load(f)

        args['shard_size'] = [len(t.dataset) for t in args['train_loaders']]

        if args['dataset'] == 'curetsr':
                img_size = torch.Size([1, 28, 28])
                len_in = 1
                for x in img_size:
                    len_in *= x
                    args['dim_in'] = len_in
                    args['dim_hidden'] = 64
                    args['dim_out'] = 14
                    args['model_fn'] = MLP_colored_for_CFFL
        else:
            img_size = torch.Size([1, 28, 28])
            len_in = 1
            for x in img_size:
                len_in *= x
                args['dim_in'] = len_in
                args['dim_hidden'] = 64
                args['dim_out'] = 10

           

        for theta in [1]:
            args['theta'] = theta

            # if n_participants == 5:
            #     args['lr'] = 0.15
            # else:
            #     args['lr'] = 0.25

            experiment_args.append(copy.deepcopy(args))

    run_experiments_full(experiment_args, repeat=1)

