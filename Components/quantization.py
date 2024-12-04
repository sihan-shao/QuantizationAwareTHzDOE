import math
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.units import *
from utils.Helper_Functions import lut_mid, nearest_neighbor_search,nearest_idx
from Components.discrete_doe import DiscreteDOE

def tau_iter(quan_fn, iter_frac, tau_min, tau_max, r=None):
    if 'softmax' in quan_fn:
        if r is None:
            r = math.log(tau_max / tau_min)
        tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
    elif 'sigmoid' in quan_fn or 'poly' in quan_fn:
        tau = 1 + 10 * iter_frac
    else:
        tau = None
    return tau


def quantization(opt, lut):
    if opt.quan_method == 'None':
        qtz = None
    else:
        qtz = Quantization(opt.quan_method, lut=lut, c=opt.c_s, num_bits=opt.uniform_nbits if lut is None else 4,
                                  tau_max=opt.tau_max, tau_min=opt.tau_min, r=opt.r, offset=opt.phase_offset)

    return qtz




def score_thickness(thickness, lut, s=5., func='sigmoid'):
    # Here s is kinda representing the steepness

    # Do we need to constrain the thickness of the range?
    diff = thickness - lut
    diff /= torch.max(torch.abs(diff))  # normalize

    if func == 'sigmoid':
        z = s * diff
        scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
    elif func == 'log':
        scores = -torch.log(diff.abs() + 1e-20) * s
    elif func == 'poly':
        scores = (1-torch.abs(diff)**s)
    elif func == 'sine':
        scores = torch.cos(math.pi * (s * diff).clamp(-1., 1.))
    elif func == 'chirp':
        scores = 1 - torch.cos(math.pi * (1-diff.abs())**s)

    return scores


# Basic function for NN-based quantization, customize it with various surrogate gradients!
class NearestNeighborSearch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, thickness, s=torch.tensor(1.0)):
        thickness_raw = thickness.detach()
        idx = nearest_idx(thickness_raw, DiscreteDOE.lut_midvals)
        thickness_q = DiscreteDOE.lut[idx]
        ctx.mark_non_differentiable(idx)
        ctx.save_for_backward(thickness_raw, s, thickness_q, idx)
        return thickness_q

    def backward(ctx, grad_output):
        return grad_output, None
    
class NearestNeighborPolyGrad(NearestNeighborSearch):

    @staticmethod
    def forward(ctx, thickness, s=torch.tensor(1.0)):
        return NearestNeighborSearch.forward(ctx, thickness, s)

    def backward(ctx, grad_output):
        input, s, output, idx = ctx.saved_tensors
        grad_input = grad_output.clone()

        dx = input - output
        d_idx = (dx / torch.abs(dx)).int().nan_to_num()
        other_end = DiscreteDOE.lut[(idx + d_idx)].to(input.device)  # far end not selected for quantization

        # normalization
        mid_point = (other_end + output) / 2
        gap = torch.abs(other_end - output) + 1e-20
        z = (input - mid_point) / gap * 2  # normalize to [-1. 1]

        dout_din = (0.5 * s * (1 - abs(z)) ** (s - 1)).nan_to_num()
        scale = 2. #* dout_din.mean() / ((dout_din**2).mean() + 1e-20)
        grad_input *= (dout_din * scale) # scale according to distance

        return grad_input, None
    
class NearestNeighborSigmoidGrad(NearestNeighborSearch):

    @staticmethod
    def forward(ctx, thickness, s=torch.tensor(1.0)):
        return NearestNeighborSearch.forward(ctx, thickness, s)

    def backward(ctx, grad_output):
        x, s, output, idx = ctx.saved_tensors
        grad_input = grad_output.clone()

        dx = x - output
        d_idx = (dx / torch.abs(dx)).int().nan_to_num()
        other_end = DiscreteDOE.lut[(idx + d_idx)].to(x.device)  # far end not selected for quantization

        # normalization
        mid_point = (other_end + output) / 2
        gap = torch.abs(other_end - output) + 1e-20
        z = (x - mid_point) / gap * 2  # normalize to [-1, 1]
        z *= s

        dout_din = (torch.sigmoid(z) * (1 - torch.sigmoid(z)))
        scale = 4. * s#1 / 0.462 * gap * s#dout_din.mean() / ((dout_din**2).mean() + 1e-20)  # =100
        grad_input *= (dout_din * scale)

        return grad_input, None
    
nns = NearestNeighborSearch.apply
nns_poly = NearestNeighborPolyGrad.apply
nns_sigmoid = NearestNeighborSigmoidGrad.apply

class SoftmaxBasedQuantization(nn.Module):
    def __init__(self, lut, gumbel=True, tau_max=3.0, c=300.):
        super(SoftmaxBasedQuantization, self).__init__()

        if not torch.is_tensor(lut):
            self.lut = torch.tensor(lut, dtype=torch.float32)
        else:
            self.lut = lut
        self.lut = self.lut.reshape(1, len(lut), 1, 1)
        self.c = c  # boost the score
        self.gumbel = gumbel
        self.tau_max = tau_max

    def forward(self, thickness, tau=1.0, hard=False):
        # phase to score
        scores = score_thickness(thickness, self.lut.to(thickness.device), (self.tau_max / tau)**1) * self.c * (self.tau_max / tau)**1.0

        # score to one-hot encoding
        if self.gumbel:  # (N, 1, H, W) -> (N, C, H, W)
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=hard, dim=1)
        else:
            y_soft = F.softmax(scores/tau, dim=1)
            index = y_soft.max(1, keepdim=True)[1]
            one_hot_hard = torch.zeros_like(scores,
                                            memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            if hard:
                one_hot = one_hot_hard + y_soft - y_soft.detach()
            else:
                one_hot = y_soft

        # one-hot encoding to phase value
        q_height_map = (one_hot * self.lut.to(one_hot.device))
        q_height_map = q_height_map.sum(1, keepdims=True)
        return q_height_map


class Quantization(nn.Module):
    def __init__(self, method=None, max_thickness=None, num_bits=4, lut=None, dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 tau_min=0.5, tau_max=3.0, r=None, c=300.):
        super(Quantization, self).__init__()
        
        # build the look-up table
        if lut is None:
            assert max_thickness != None
            # linear look-up table
            lut = torch.linspace(0, max_thickness, 2**num_bits + 1).to(dev)
        else:
            # non-linear look-up table
            assert len(lut) == (2**num_bits) + 1
            lut = torch.tensor(lut, dtype=torch.float32).to(dev)

        self.quan_fn = None
        self.gumbel = 'gumbel' in method.lower()
        if method.lower() == 'nn':
            self.quan_fn = nns
        elif method.lower() == 'nn_sigmoid':
            self.quan_fn = nns_sigmoid
        elif method.lower() == 'nn_poly':
            self.quan_fn = nns_poly
        elif 'softmax' in method.lower():
            self.quan_fn = SoftmaxBasedQuantization(lut[:-1], self.gumbel, tau_max=tau_max, c=c)

        self.method = method
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.r = r

    def forward(self, input_thickness, iter_frac=None, hard=True):
        if iter_frac is not None:
            tau = tau_iter(self.method, iter_frac, self.tau_min, self.tau_max, self.r)
        if self.quan_fn is None:
            return input_thickness
        else:
            if isinstance(tau, float):
                tau = torch.tensor(tau, dtype=torch.float32).to(input_thickness.device)
            if 'nn' in self.method.lower():
                s = tau
                return self.quan_fn(input_thickness, s).squeeze(0,1)
            else:
                return self.quan_fn(input_thickness, tau, hard).squeeze(0,1)