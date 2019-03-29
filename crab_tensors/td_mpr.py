
# coding: utf-8

# In[1]:


import functools
import operator

import torch
import torch.nn as nn


# In[3]:


class MPRLowRankCP(nn.Module):
    def __init__(self, num_inputs, num_outputs, polynom_order, rank, bias=True):
        assert polynom_order >= 1
        assert rank >= 1
        super().__init__()
        self.polynom_order = polynom_order
        self.rank = rank
        self.bias = bias
        self.num_outputs = num_outputs
        self.actual_num_inputs = num_inputs + bias
        self.factors = nn.ParameterList(
            nn.Parameter(
                torch.ones(self.num_outputs, self.actual_num_inputs, self.rank),
                requires_grad=True
            )
            for i in range(self.polynom_order)
        )
        self.reset_parameters()
    
    def _actual_num_features(self):
        return self.num_features + bias

    def reset_parameters(self):
        for i in range(len(self.factors)):
            torch.randn(
                size=self.factors[i].data.shape,
                out=self.factors[i].data
            )
    
    def forward(self, X):
        assert len(X.shape) == 2
        num_samples = X.shape[0]
        if self.bias:
            X = torch.cat(
                [X, torch.ones(num_samples, 1, device=X.device)],
                dim=1
            )
        thingies = [
            # n - num of sample, f - num o feature
            # r - num of rank one component, o - num of output
            torch.einsum("nf,ofr->nor", X, factor)
            for factor in self.factors
        ]
        return functools.reduce(operator.mul, thingies).sum(dim=2)

