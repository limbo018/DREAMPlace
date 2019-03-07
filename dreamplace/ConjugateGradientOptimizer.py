##
# @file   ConjugateGradientOptimizer.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Conjugate gradient descent optimizer. 
#

import os 
import sys
import time 
import pickle 
import numpy as np 
import torch 
from torch.optim.optimizer import Optimizer, required
import pdb 

class ConjugateGradientOptimizer(Optimizer):
    """
    @brief Implement conjugate gradient descent optimization algorithm. 
    @param params variables 
    @param lr learning rate  
    @param line_search_fn a callable line search function 
    """
    def __init__(self, params, lr=required, line_search_fn=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # g_k_1 is previous gradient, initial is zero  
        # d_k_1 is previous search direction, initial zero  
        defaults = dict(lr=lr, g_k_1=[], d_k_1=[], line_search_fn=line_search_fn, obj_eval_count=0, obj_at_alpha_k=[None])
        super(ConjugateGradientOptimizer, self).__init__(params, defaults)

        # I do not know how to get generator's length 
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super(ConjugateGradientOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            line_search_fn = group['line_search_fn']
            obj_at_alpha_k = group['obj_at_alpha_k']
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                g_k = p.grad.data
                if not group['g_k_1']: 
                    group['g_k_1'].append(torch.zeros_like(g_k))
                    group['d_k_1'].append(torch.zeros_like(g_k))
                    g_k_1 = group['g_k_1'][i]
                    d_k_1 = group['d_k_1'][i]
                    beta_k = torch.zeros(1, dtype=g_k.dtype, device=g_k.device)
                else:
                    g_k_1 = group['g_k_1'][i]
                    d_k_1 = group['d_k_1'][i]
                    # compute beta_k 
                    # g_k^T(g_k-g_k_1) / |g_k_1|_2^2
                    beta_k = g_k.dot(g_k.sub(g_k_1)).div(g_k_1.pow(2).sum())
                # compute conjugate directions 
                # d_k = -g_k + beta_k*d_k_1
                d_k = d_k_1.mul(beta_k).sub(g_k)
                if line_search_fn is None: # heuristic from NTUPlace3 for CG without line search 
                    alpha_k = torch.tensor(group['lr']/torch.norm(d_k), dtype=d_k.dtype, device=d_k.device)
                else:
                    alpha_k = torch.tensor(group['lr'], dtype=d_k.dtype, device=d_k.device)
                    alpha_k, line_search_count, obj_at_alpha_k[0] = line_search_fn(xk=p.data, pk=d_k, gfk=g_k, fk=None, alpha0=alpha_k)
                    group['obj_eval_count'] += line_search_count
                    print("alpha_k = %g, line_search_count = %d, obj_at_alpha_k = %g, obj_eval_count = %d" % (alpha_k, line_search_count, obj_at_alpha_k[0], group['obj_eval_count']))
                p.data.add_(alpha_k.mul(d_k))

                g_k_1.data.copy_(g_k)
                d_k_1.data.copy_(d_k)

        return loss

