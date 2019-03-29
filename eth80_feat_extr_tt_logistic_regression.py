
# coding: utf-8

# In[1]:


import random
import functools
import itertools
import operator
import gc

with __import__('importnb').Notebook():
    # github.com/deathbeds/importnb
    import eth80
    from eth80 import Eth80Dataset, extract_X_y_train_test
    from feature_extraction import TuckerFeatureExtractor
    from ml_utils import (
        do_every_num_epochs, memreport, train_and_evaluate
    )
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorly as tl

import torch
import torch.nn as nn
import torch.nn.functional as tnnf

from t3nsor import TTLinear # install https://github.com/KhrulkovV/tt-pytorch


# In[2]:


tl.set_backend("pytorch")
device = torch.device("cuda:0")
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# In[3]:


eth80_dataset = Eth80Dataset(
    "/mnt/hdd_1tb/smiles_backup/Documents/datasets/eth80/eth80-cropped-close128/"
)


# In[4]:


def make_logistic_regression(
    num_extracted_features, learning_rate, regularization_coefficient
):
    model = TTLinear(
        in_features=num_extracted_features,
        out_features=eth80.NUM_CLASSES,
        d=4,
        tt_rank=8
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=regularization_coefficient
    )
    return model, optimizer


# In[5]:


def evaluate_logistic_regression():
    extracted_features_shape = (14, 3, 23, 2)
    X_train, y_train, X_test, y_test = extract_X_y_train_test(
        eth80_dataset, num_test_objects_per_class=2,
        extracted_features_shape=extracted_features_shape
    )
    model, optimizer = make_logistic_regression(
        functools.reduce(operator.mul, extracted_features_shape),
        learning_rate=1e-3,
        regularization_coefficient=0.01933
    )
    train_and_evaluate(
        X_train, y_train, X_test, y_test,
        model, optimizer,
        eval_every_num_epochs=3, plot_every_num_epochs=60,
        num_epochs=1500
    )


# In[6]:


evaluate_logistic_regression()

