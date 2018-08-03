import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from modules.optimizer import Optimizer
from modules.loss import LossFunction
from modules.layer import GRU
import modules.evaluate as E
from modules.data import SessionDataset, SessionDataLoader


class GRU4REC:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 optimizer_type='Adagrad', lr=.01, weight_decay=0,
                 momentum=0, eps=1e-6, loss_type='TOP1',
                 clip_grad=-1, dropout_input=.0, dropout_hidden=.5,
                 batch_size=50, use_cuda=True, time_sort=False, pretrained=None,
                 n_sample=2048, sample_alpha=0.75, sample_store=10000000, bpreg=1.0):
        """ The GRU4REC model

        Args:
            input_size (int): dimension of the gru input variables
            hidden_size (int): dimension of the gru hidden units
            output_size (int): dimension of the gru output variables
            num_layers (int): the number of layers in the GRU
            optimizer_type (str): optimizer type for GRU weights
            lr (float): learning rate for the optimizer
            weight_decay (float): weight decay for the optimizer
            momentum (float): momentum for the optimizer
            eps (float): eps for the optimizer
            loss_type (str): type of the loss function to use
            clip_grad (float): clip the gradient norm at clip_grad. No clipping if clip_grad = -1
            dropout_input (float): dropout probability for the input layer
            dropout_hidden (float): dropout probability for the hidden layer
            batch_size (int): mini-batch size
            use_cuda (bool): whether you want to use cuda or not
            time_sort (bool): whether to ensure the the order of sessions is chronological (default: False)
            pretrained (modules.layer.GRU): pretrained GRU layer, if it exists (default: None)
        """
        
        # Initialize the GRU Layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        ###修改###
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.sample_store = sample_store
        self.bpreg = bpreg
        ###修改###
        if pretrained is None:
            self.gru = GRU(input_size, hidden_size, output_size, num_layers,
                           dropout_input=dropout_input,
                           dropout_hidden=dropout_hidden,
                           batch_size=batch_size,
                           use_cuda=use_cuda)
        else:
            self.gru = pretrained

        # Initialize the optimizer
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.eps = eps
        self.optimizer = Optimizer(self.gru.parameters(),
                                   optimizer_type=optimizer_type,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   momentum=momentum,
                                   eps=eps)

        # Initialize the loss function
        self.loss_type = loss_type
        self.loss_fn = LossFunction(loss_type, use_cuda)

        # gradient clipping(optional)
        self.clip_grad = clip_grad 

        # etc
        self.time_sort = time_sort

    def generate_neg_samples(self, n_items, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_sample * length))
        else:
            # 从 np.arange(n_items) 中产生一个size为n_sample * length的随机采样
            sample = np.random.choice(n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample

    def run_epoch(self, dataset, k=20, training=True):
        """ Run a single training epoch """
        start_time = time.time()
        
        # initialize
        losses = []
        recalls = []
        mrrs = []
        ##增加###
        zippers = []
        ##增加###
        optimizer = self.optimizer
        hidden = self.gru.init_hidden()
        if not training:
            self.gru.eval()
        device = self.device
        
        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            
            return hidden

        # Start the training loop
        loader = SessionDataLoader(dataset, batch_size=self.batch_size)
        # 一个bach一个bach的迭代,每次迭代是一个input:tensor([ 31,  26,  27,  29,  24]);一个output:tensor([ 31,  26,  28,  17,  24])
        #
        if training==True:
            n_items = len(dataset.items)
            # sampling 增加额外负样本采样
            if self.n_sample > 0:
                pop = dataset.df.groupby('ItemId').size()  # item的流行度supp,数据如下格式
                # ItemId
                # 214507331  1
                # 214507365  1
                # 将sample_alpha设置为1会导致基于流行度的采样，将其设置为0会导致均匀采样
                pop = pop[dataset.itemmap[
                    dataset.item_key].values].values ** self.sample_alpha  # item选择作为样本的概率为supp ^ sample_alpha
                pop = pop.cumsum() / pop.sum()
                pop[-1] = 1
                if self.sample_store:
                    generate_length = self.sample_store // self.n_sample
                    if generate_length <= 1:
                        sample_store = 0
                        print('No example store was used')
                    else:
                        neg_samples = self.generate_neg_samples(n_items, pop, generate_length)
                        sample_pointer = 0
                else:
                    print('No example store was used')

        for input, target, mask in loader:
            input = input.to(device)
            target = target.to(device)
            # print(input)
            # print(target)
            #额外的 SAMPLING THE OUTPUT
            if self.n_sample>0 and training:
                if self.sample_store:
                    if sample_pointer == generate_length:
                        neg_samples = self.generate_neg_samples(n_items, pop, generate_length)
                        sample_pointer = 0
                    sample = neg_samples[sample_pointer]
                    sample_pointer += 1
                else:
                    sample = self.generate_neg_samples(pop, 1)
                y = torch.LongTensor(np.hstack([target, sample]))
            else:
                y = target   #不增加额外采样
            # reset the hidden states if some sessions have just terminated
            hidden = reset_hidden(hidden, mask).detach()
            # Go through the GRU layer
            logit, hidden = self.gru(input, target, hidden)
            # Output sampling   #理解,很重要！！！！！！！
            y = y.to(device)
            logit_sampled = logit[:, y]
            # Calculate the mini-batch loss
            loss = self.loss_fn(logit_sampled)
            with torch.no_grad():
                recall, mrr = E.evaluate(logit, target, k)
            losses.append(loss.item())         
            recalls.append(recall)
            mrrs.append(mrr)
            # Gradient Clipping(Optional)
            if self.clip_grad != -1:
                for p in self.gru.parameters():
                    p.grad.data.clamp_(max=self.clip_grad)
            # Mini-batch GD
            if training:
                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() # flush the gradient after the optimization

        results = dict()
        results['loss'] = np.mean(losses)
        results['recall'] = np.mean(recalls)
        results['mrr'] = np.mean(mrrs)
        
        end_time = time.time()
        results['time'] = (end_time - start_time) / 60
        
        if not training:
            self.gru.train()

        return results
    
    def train(self, dataset, k=20, n_epochs=10, save_dir='./models', save=True, model_name='GRU4REC'):
        """
        Train the GRU4REC model on a pandas dataframe for several training epochs,
        and store the intermediate models to the user-specified directory.

        Args:
            n_epochs (int): the number of training epochs to run
            save_dir (str): the path to save the intermediate trained models
            model_name (str): name of the model
        """
        print(f'Training {model_name}...')

        for epoch in range(n_epochs):
            results = self.run_epoch(dataset, k=k, training=True)
            results = [f'{k}:{v:.3f}' for k, v in results.items()]
            print(f'epoch:{epoch+1:2d}/{"/".join(results)}')
            
            # Store the intermediate model
            if save:
                save_dir = Path(save_dir)
                if not save_dir.exists(): save_dir.mkdir()
                model_fname = f'{model_name}_{self.loss_type}_{self.optimizer_type}_{self.lr}_epoch{epoch+1:d}'
                torch.save(self.gru.state_dict(), save_dir/model_fname)
    

    def test(self, dataset, k=20):
        """ Model evaluation

        Args:
            k (int): the length of the recommendation list

        Returns:
            avg_loss: mean of the losses over the session-parallel minibatches
            avg_recall: mean of the Recall@K over the session-parallel mini-batches
            avg_mrr: mean of the MRR@K over the session-parallel mini-batches
            wall_clock: time took for testing
        """
        results = self.run_epoch(dataset, k=k, training=False)
        results = [f'{k}:{v:.3f}' for k, v in results.items()]
        print(f'Test result: {"/".join(results)}')
    

