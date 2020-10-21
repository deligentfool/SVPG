import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import parameters_to_vector, vector_to_parameters


def _square_dist(x):
    # * x: [num_sample, feature_dim]
    xxT = torch.mm(x, x.t())
    # * xxT: [num_sample, num_sample]
    xTx = xxT.diag()
    # * xTx: [num_sample]
    return xTx + xTx.unsqueeze(1) - 2. * xxT


def _Kxx_dxKxx(x, num_agent):
    square_dist = _square_dist(x)
    # * bandwidth = 2 * (med ^ 2)
    bandwidth = 2 * square_dist.median() / math.log(num_agent)
    Kxx = torch.exp(-1. / bandwidth * square_dist)

    dxKxx = 2 * (Kxx.sum(1).diag() - Kxx).matmul(x) / bandwidth

    return Kxx, dxKxx


def calc_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R += gamma * r
        returns.insert(0, R)
    return returns

