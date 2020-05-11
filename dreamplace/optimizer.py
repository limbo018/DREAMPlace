#########################
#       Optimizer       #
#########################
import torch
from torch.optim.optimizer import Optimizer, required
import math
import numpy as np
from LineSearch import build_line_search_fn_armijo
__all__ = ["Adam_GC", "SGD_GC", "RAdam", "Nesterov_Armijo", "ZerothOrderSearch"]

class Adam_GC(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_GC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_GC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                #GC operation for Conv layers and FC layers
                if grad.dim()>1:
                   grad.add_(-grad.mean(dim = tuple(range(1,grad.dim())), keepdim = True))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class SGD_GC(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_GC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_GC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                #GC operation for Conv layers and FC layers
                if d_p.dim()>1:
                   d_p.add_(-d_p.mean(dim = tuple(range(1,d_p.dim())), keepdim = True))

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


class RAdam(Optimizer):
    r"""
    https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), obj_and_grad_fn=required, obj_fn=None, eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        self.obj_and_grad_fn = obj_and_grad_fn
        self.line_search_fn = build_line_search_fn_armijo(obj_fn) if(obj_fn) else None
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            for p in group['params']:
                if p.grad is None:
                    continue
                obj, grad = obj_and_grad_fn(p)
                # grad = p.grad.data.float()
                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if(self.line_search_fn is None):
                        if N_sma >= 5:
                            step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                    else:
                        ### armijo line search
                        step_size = self.line_search_fn(xk=p, pk=-grad, gfk=grad, fk=obj, alpha0=torch.tensor([10],device=p.device), c1=1e-4)[0]
                        print(f"done search, stepsize={step_size}")
                    buffered[2] = step_size

                # p_data_fp32.add_(-step_size * group['lr'] * exp_avg)
                p_data_fp32.add_(-step_size * grad)
                p.data.copy_(p_data_fp32)

                # more conservative since it's an approximated value
                # if N_sma >= 5:
                #     if group['weight_decay'] != 0:
                #         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                #     denom = exp_avg_sq.sqrt().add_(group['eps'])
                #     p_data_fp32 += -step_size * group['lr'] * exp_avg / denom
                #     p.data.copy_(p_data_fp32)
                # elif step_size > 0:
                #     if group['weight_decay'] != 0:
                #         p_data_fp32.add_(-group['weight_decay'] * group['lr'] * p_data_fp32)
                #     p_data_fp32.add_(-step_size * group['lr'] * exp_avg)
                #     p.data.copy_(p_data_fp32)

        return loss


class Nesterov_Armijo(Optimizer):
    def __init__(self, params, lr=required, momentum=0, obj_and_grad_fn=required, obj_fn=None, dampening=0,
                weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, eta=lr, eta_max=lr, gamma=0.5, step=1, tau=1, lamb=1, lamb_prev=0, max_backtrack_count=20, beta=0.8)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.obj_and_grad_fn = obj_and_grad_fn
        self.obj_fn = obj_fn

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def reset(self, eta, eta_max, gamma, step, opt):
        if(step == 1):
            return eta_max
        elif(opt == 0):
            return eta
        elif(opt == 1):
            return eta_max
        elif(opt == 2):
            return eta_max * gamma ** (step/80)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                obj, grad = self.obj_and_grad_fn(p)
                # if p.grad is None:
                #     continue
                # d_p = p.grad.data
                d_p = grad.data
                grad_norm = d_p.data.norm(p=2)
                print("grad avg:", d_p.data.mean(), "grad norm:", grad_norm)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # armijo line search
                # reset
                group["eta"] = self.reset(group["eta"], group["eta_max"], group["gamma"], group["step"], opt=2)
                group["eta"] = group["eta_max"] / grad_norm
                # if(group["step"] % 20 == 0):
                #     group["eta_max"] *= group["gamma"]
                grad_norm = grad.dot(grad)
                c = 0.5
                backtrack_count = 0
                obj_start = self.obj_fn(p)
                while(self.obj_fn(p - group["eta"]*grad) > obj_start - c*group["eta"]*grad_norm and backtrack_count < group["max_backtrack_count"]):
                    group["eta"] = group["beta"]*group["eta"]
                    backtrack_count += 1
                # if(backtrack_count == group["max_backtrack_count"]):
                #     print("fail to find optimal step size")
                #     group["eta"] = 0
                #     group["eta_max"] /= group["gamma"]

                print("find stepsize:", group["eta"])

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening * d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['eta'] * d_p)
                group["step"] += 1



class ZerothOrderSearch(Optimizer):
    def __init__(self, params, obj_fn=required, placedb=None, n_step=1, n_sample=1, r_max=8, r_min=1):
        defaults = dict(n_step=n_step, n_sample=n_sample, r_max=r_max, r_min=r_min)

        self.obj_fn = obj_fn
        self.placedb = placedb

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

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

            for p in group['params']:
                R, r = group["r_max"], group["r_min"]
                K = int(np.log2(R/r)) + 1
                T = group["n_step"]
                num_movable_nodes = self.placedb.num_movable_nodes
                num_nodes = self.placedb.num_nodes
                num_filler_nodes = self.placedb.num_filler_nodes
                num_fixed_nodes = num_nodes - num_movable_nodes - num_filler_nodes
                obj_min = self.obj_fn(p.data).data.item()
                pos_min = p.data
                v_min = 0
                # print(obj_min)
                for t in range(T):
                    obj_min = self.obj_fn(p.data).data.item()
                    obj_start = obj_min
                    # print(f"start obj: {obj_start}")
                    for k in range(K):
                        r_k = 2**(-k) * R
                        for i in range(group["n_sample"]):
                            v_k = torch.randn_like(p.data)
                            v_k[num_movable_nodes:num_nodes-num_filler_nodes] = 0
                            v_k[num_nodes+num_movable_nodes:-num_filler_nodes] = 0
                            # v_k = v_k / v_k.norm(p=2) * r_k
                            v_k = v_k / v_k.norm(p=2) * r_k
                            p1 = p.data + v_k
                            obj_k = self.obj_fn(p1).data.item()
                            if(obj_k < obj_min):
                                obj_min = obj_k
                                v_min = v_k.clone()
                                r_min = r_k
                                pos_min = p1.clone()
                                # print(v_min.sum(), pos_min.mean())
                    # zeroth-order optimization with decaying step size
                    diff = obj_start - obj_min
                    if(diff > 0.001):
                        step_size = max(0.8, min(1.2, diff / r_min))
                        # print(f"Search step: {t} stepsize: {step_size:5.2f} r_min: {r_min} obj reduce from {obj_start} to {obj_min}")
                        p.data.copy_(p.data + v_min * step_size)


