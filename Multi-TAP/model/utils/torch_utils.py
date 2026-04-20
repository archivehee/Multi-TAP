import torch
from torch.optim.optimizer import Optimizer

class MyAdagrad(Optimizer):
    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) * init_accu_value

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: 
                    continue
                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1
                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay not compatible with sparse gradients")
                    grad = grad.add(group['weight_decay'], p.data)
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                if p.grad.data.is_sparse:
                    grad = grad.coalesce()
                    gi = grad._indices(); gv = grad._values()
                    size = torch.Size([x for x in grad.size()])
                    def make_sparse(values):
                        ctor = type(p.grad.data)
                        if gi.dim() == 0 or values.dim() == 0:
                            return ctor()
                        return ctor(gi, values, size)
                    state['sum'].add_(make_sparse(gv.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(gv / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)
        return loss

def get_optimizer(name, parameters, lr, l2=0):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    if name in ['adagrad', 'myadagrad']:
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=l2)
    if name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2)
    if name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    raise Exception(f"Unsupported optimizer: {name}")

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
