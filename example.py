
# coding: utf-8

# # Load the Data & Train on a small subset

# In[4]:

import pandas as pd
import numpy as np
from pathlib import Path
from modules.data import SessionDataset

PATH_HOME = Path.home()
PATH_PROJ = PATH_HOME/'pyGRU4REC' 
PATH_DATA = PATH_PROJ/'data'
PATH_MODEL = PATH_PROJ/'models'
train = 'train.tsv'
test = 'test.tsv'
PATH_TRAIN = PATH_DATA / train
PATH_TEST = PATH_DATA / test
n_samples = -1
n_samples = 100000
train_dataset = SessionDataset(PATH_TRAIN, n_samples=n_samples)
test_dataset = SessionDataset(PATH_TEST, n_samples=n_samples, itemmap=train_dataset.itemmap)


# Train on a small subset of data

# In[2]:

from modules.model import GRU4REC
import torch

input_size = len(train_dataset.items)
hidden_size = 100
num_layers = 1
output_size = input_size
batch_size = 50

optimizer_type = 'Adagrad'
lr = .01
weight_decay = 0
momentum = 0
eps = 1e-6

loss_type = 'TOP1'

n_epochs = 10
use_cuda = True

torch.manual_seed(7)

model = GRU4REC(input_size, hidden_size, output_size,
                num_layers=num_layers,
                batch_size=batch_size,
                optimizer_type=optimizer_type,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                eps=eps,
                loss_type=loss_type,
                use_cuda=use_cuda)

model_name = 'GRU4REC'
model.train(train_dataset, n_epochs=n_epochs, model_name=model_name, save=False)
model.test(test_dataset)


# # Full Training

# In[ ]:

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from modules.data import SessionDataset
from modules.model import GRU4REC


PATH_HOME = Path.home()
PATH_PROJ = PATH_HOME/'pyGRU4REC' 
PATH_DATA = PATH_PROJ/'data'
PATH_MODEL = PATH_PROJ/'models'
train = 'train.tsv'
test = 'test.tsv'
PATH_TRAIN = PATH_DATA / train
PATH_TEST = PATH_DATA / test

train_dataset = SessionDataset(PATH_TRAIN)
test_dataset = SessionDataset(PATH_TEST, itemmap=train_dataset.itemmap)

input_size = len(train_dataset.items)
hidden_size = 100
num_layers = 1
output_size = input_size
batch_size = 50

optimizer_type = 'Adagrad'
lr = .01
weight_decay = 0
momentum = 0
eps = 1e-6

loss_type = 'TOP1'

n_epochs = 10
use_cuda = True

torch.manual_seed(7)

model = GRU4REC(input_size, hidden_size, output_size,
                num_layers=num_layers,
                batch_size=batch_size,
                optimizer_type=optimizer_type,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                eps=eps,
                loss_type=loss_type,
                use_cuda=use_cuda)

model_name = 'GRU4REC'
model.train(train_dataset, n_epochs=n_epochs, model_name=model_name, save=False)
model.test(test_dataset)


# ## Evaluate using the trained models

# In[3]:

from pathlib import Path
from modules.data import SessionDataset

PATH_HOME = Path.home()
PATH_PROJ = PATH_HOME/'pyGRU4REC' 
PATH_DATA = PATH_PROJ/'data'
PATH_MODEL = PATH_PROJ/'models'
train = 'train.tsv'
test = 'test.tsv'
PATH_TRAIN = PATH_DATA / train
PATH_TEST = PATH_DATA / test
train_dataset = SessionDataset(PATH_TRAIN)
test_dataset = SessionDataset(PATH_TEST, itemmap=train_dataset.itemmap)


# ## 1.Load the Common Parameters

# In[4]:

import torch
from modules.layer import GRU
from modules.model import GRU4REC

input_size = len(train_dataset.items)
output_size = input_size
hidden_size = 100
num_layers = 1

use_cuda = True
time_sort = False

optimizer_type = 'Adagrad'


# ## 2. Evaluation(TOP1 Loss)

# In[5]:

model_name = 'GRU4REC_TOP1_Adagrad_0.01_epoch5'
model_file = PATH_MODEL/model_name

loss_type = 'TOP1'
lr = 0.01

dropout_hidden = .5
dropout_input = 0
batch_size = 50
momentum = 0

gru = GRU(input_size, hidden_size, output_size,
          num_layers = num_layers,
          dropout_input = dropout_input,
          dropout_hidden = dropout_hidden,
          batch_size = batch_size,
          use_cuda = use_cuda)

gru.load_state_dict(torch.load(model_file))

model = GRU4REC(input_size, hidden_size, output_size,
                num_layers = num_layers,
                dropout_input = dropout_input,
                dropout_hidden = dropout_hidden,
                batch_size = batch_size,
                use_cuda = use_cuda,
                loss_type = loss_type,
                optimizer_type = optimizer_type,
                lr=lr,
                momentum=momentum,
                time_sort=time_sort,
                pretrained=gru)

k = 20
model.test(test_dataset, k=k)


# ## 3. Evaluation(BPR Loss)

# In[6]:

model_name = 'GRU4REC_BPR_Adagrad_0.05_epoch5'
model_file = PATH_MODEL/model_name

loss_type = 'BPR'
lr = 0.05

dropout_hidden = .2
dropout_input = 0
batch_size = 50
momentum = 0.2

gru = GRU(input_size, hidden_size, output_size,
          num_layers = num_layers,
          dropout_input = dropout_input,
          dropout_hidden = dropout_hidden,
          batch_size = batch_size,
          use_cuda = use_cuda)

gru.load_state_dict(torch.load(model_file))

model = GRU4REC(input_size, hidden_size, output_size,
                num_layers = num_layers,
                dropout_input = dropout_input,
                dropout_hidden = dropout_hidden,
                batch_size = batch_size,
                use_cuda = use_cuda,
                loss_type = loss_type,
                optimizer_type = optimizer_type,
                lr=lr,
                momentum=momentum,
                time_sort=time_sort,
                pretrained=gru)

k = 20
model.test(test_dataset, k=k)


# ## 4. Evaluation(CrossEntropyLoss)

# In[7]:

model_name = 'GRU4REC_CrossEntropy_Adagrad_0.01_epoch5'
model_file = PATH_MODEL/model_name

loss_type = 'CrossEntropy'
lr = 0.01

dropout_hidden = 0
dropout_input = 0
batch_size = 500
momentum = 0

gru = GRU(input_size, hidden_size, output_size,
          num_layers = num_layers,
          dropout_input = dropout_input,
          dropout_hidden = dropout_hidden,
          batch_size = batch_size,
          use_cuda = use_cuda)

gru.load_state_dict(torch.load(model_file))

model = GRU4REC(input_size, hidden_size, output_size,
                num_layers = num_layers,
                dropout_input = dropout_input,
                dropout_hidden = dropout_hidden,
                batch_size = batch_size,
                use_cuda = use_cuda,
                loss_type = loss_type,
                optimizer_type = optimizer_type,
                lr=lr,
                momentum=momentum,
                time_sort=time_sort,
                pretrained=gru)

k = 20
model.test(test_dataset, k=k)

