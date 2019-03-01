##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
#

import os 
import sys
import time 
import pickle 
import numpy as np 
import torch 
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import pdb 

class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    Follow the implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    """
    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution 
        # v_k is reference solution 
        # a_k is optimization parameter 
        # alpha_k is the step size 
        # v_k_1 is previous reference solution 
        # g_k_1 is gradient to v_k_1 
        defaults = dict(lr=lr, 
                u_k=[], v_k=[], g_k=[], a_k=[], alpha_k=[], 
                v_k_1=[], g_k_1=[], 
                v_kp1 = [None], 
                obj_and_grad_fn=obj_and_grad_fn, 
                constraint_fn=constraint_fn,
                obj_eval_count=0)
        super(NesterovAcceleratedGradientOptimizer, self).__init__(params, defaults)

        # I do not know how to get generator's length 
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super(NesterovAcceleratedGradientOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = group['obj_and_grad_fn']
            constraint_fn = group['constraint_fn']
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']: 
                    group['u_k'].append(p.data.clone())
                    # directly use p as v_k to save memory 
                    #group['v_k'].append(torch.autograd.Variable(p.data, requires_grad=True))
                    group['v_k'].append(p)
                    obj, grad = obj_and_grad_fn(group['v_k'][i])
                    group['g_k'].append(grad.data.clone()) # must clone 
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                g_k = group['g_k'][i]
                if not group['a_k']: 
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)
                    obj, grad = obj_and_grad_fn(group['v_k_1'][i])
                    group['g_k_1'].append(grad.data)
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                g_k_1 = group['g_k_1'][i]
                if not group['alpha_k']: 
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]

                if group['v_kp1'][i] is None: 
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]

                # line search with alpha_k as hint 
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10

                ttt = time.time()
                while True: 
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof: 
                    u_kp1 = v_k - alpha_k*g_k
                    #constraint_fn(u_kp1)
                    v_kp1.data.copy_(u_kp1 + coef*(u_kp1-u_k))
                    # make sure v_kp1 subjects to constraints 
                    # g_kp1 must correspond to v_kp1 
                    constraint_fn(v_kp1)

                    f_kp1, g_kp1 = obj_and_grad_fn(v_kp1)

                    #tt = time.time()
                    alpha_kp1 = torch.dist(v_kp1.data, v_k.data, p=2) / torch.dist(g_kp1.data, g_k.data, p=2) 
                    backtrack_cnt += 1
                    group['obj_eval_count'] += 1
                    #print("\t\talpha_kp1 %.3f ms" % ((time.time()-tt)*1000))
                    #torch.cuda.synchronize()
                    #print(prof)

                    #print("alpha_kp1 = %g, line_search_count = %d, obj_eval_count = %d" % (alpha_kp1, backtrack_cnt, group['obj_eval_count']))
                    #print("|g_k| = %.6E, |g_kp1| = %.6E" % (g_k.norm(p=2), g_kp1.norm(p=2)))
                    if alpha_kp1 > 0.95*alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        alpha_k.data.copy_(alpha_kp1.data)
                        break 
                    else:
                        alpha_k.data.copy_(alpha_kp1.data)
                torch.cuda.synchronize()
                #print("\tline search %.3f ms" % ((time.time()-ttt)*1000))

                #print("v_k")
                #print(v_k.view([2, -1]).t())
                #print("v_k_1")
                #print(v_k_1.view([2, -1]).t())
                #print("v_kp1")
                #print(v_kp1.view([2, -1]).t())
                #pdb.set_trace()
                v_k_1.data.copy_(v_k.data)
                g_k_1.data.copy_(g_k.data)

                u_k.data.copy_(u_kp1.data)
                #print("|displace| = %g" % (torch.dist(v_k.data, v_kp1.data, p=1)))
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                a_k.data.copy_(a_kp1.data)

                # although the solution should be u_k 
                # we need the gradient of v_k 
                # the update of density weight also requires v_k 
                # I do not know how to copy u_k back to p when exit yet 
                #p.data.copy_(v_k.data)

        return loss

