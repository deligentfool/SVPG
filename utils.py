import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Parameters on different types!')
    return old_param_device


def parameters_to_vector(parameters, grad=False, both=False):
    param_device = None
    if not both:
        vec = []
        if not grad:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.view(-1))

        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.detach().view(-1))
        return torch.cat(vec)

    else:
        param_vec = []
        grad_vec = []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            param_vec.append(param.view(-1))
            grad_vec.append(param.grad.detach().view(-1))
        return torch.cat(param_vec), torch.cat(grad_vec)


def vector_to_parameters(vector, parameters, grad=True):
    param_device = None
    pointer = 0

    if grad:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vector[pointer: pointer + num_param].view(param.size())
            pointer += num_param
    else:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vector[pointer: pointer + num_param].view(param.size())
            pointer += num_param