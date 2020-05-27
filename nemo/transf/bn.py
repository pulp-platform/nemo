#
# bn.py
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2018-2020 ETH Zurich
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from nemo.precision import Precision
from nemo.quant.pact import *
from nemo.graph import DeployGraph
from torch.nn.modules.utils import _single,_pair
from collections import OrderedDict
import types
import logging
import numpy as np
import copy
import math
import torchvision.models
import re
from nemo.transf.common import *

BN_PRECISION_MAX = 20

def _absorb_affine_bn(self):
    for n,m in self.named_modules():
        if m.__class__.__name__ == "BatchNorm2d":
            m.weight.data[:] = m.weight.data[:] / torch.sqrt(m.running_var.data[:] + m.eps)
            m.bias.data[:] = m.bias.data[:] - m.weight.data[:] * m.running_mean.data[:]
            m.running_var.data[:] = (1. - m.eps)**2
            m.running_mean.data[:] = 0.

def _prime_affine_bn(self):
    for n,m in self.named_modules():
        if m.__class__.__name__ == "BatchNorm2d":
            m.weight.data[:] = m.weight.data[:] * torch.sqrt(m.running_var.data[:] + m.eps)
            m.bias.data[:] = m.bias.data[:] + m.weight.data[:] * m.running_mean.data[:] / torch.sqrt(m.running_var.data[:] + m.eps)

def _freeze_bn(self, reset_stats=False, disable_grad=False):
    r"""Sets :py:class:torch.nn.BatchNorm2d` layers to not collect statistics and keep the current `running_var` and `running_mean`.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "BatchNorm2d":
            if reset_stats:
                try:
                    eps = m.eps
                except AttributeError:
                    eps = 0.
                gamma = m.weight.data[:].clone().detach().cpu()
                beta  = m.bias.data[:].clone().detach().cpu()
                sigma = torch.sqrt(m.running_var.data[:] + eps).clone().detach().cpu()
                mu    = m.running_mean.data[:].clone().detach().cpu()
                kappa = gamma/sigma
                lamda = beta-gamma/sigma * mu
                m.weight.data[:]  = kappa
                m.bias.data[:]    = lamda
                m.running_var[:]  = 1.
                m.running_mean[:] = 0.
            m.track_running_stats = True
            m.eval()
            if disable_grad:
                m.weight.requires_grad = False
                m.bias.requires_grad = False

def _calibrate_bn_pact(self, calib_dict={}, kappa_bit_default=14, lamda_bit_default=20, kappa_dict=None, lamda_dict=None, range_factor=8, minmax=False, **kwargs):
    r"""Calibrates BN layer quantization for :py:class:`nemo.quant.pact.PACT_QuantizedBatchNormNd` layers.
    Using BN min-max statistics previously calculated, this method calibrates the number of bits used in BN parameters
    so that both the BN output and the `kappa`, `lamda` affine parameters are representable.
    
    :param kappa_bit_default: Default maximum number of bits for BN multiplicative parameter (can be overridden by `kappa_dict`); default 16.
    :type  kappa_bit_default: int
    :param lamda_bit_default: Default maximum number of bits for BN additive parameter (can be overridden by `lamda_dict`); default 32.
    :type  lamda_bit_default: int
    :param kappa_dict: dictionary of maximum number of bits for `kappa` in specific layers; overrides `kappa_bit_default`. Default None.
    :type  kappa_dict: `dict` or `collections.OrderedDict`
    :param lamda_dict: dictionary of maximum number of bits for `lamda` in specific layers; overrides `lamda_bit_default`. Default None.
    :type  lamda_dict: `dict` or `collections.OrderedDict`
    
    """

    if not calib_dict:
        calib_dict = get_calib_dict_from_supernodes(self)

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Act" or \
            m.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
            module_dict[n] = m

    for n,m in self.named_modules():
        if m.__class__.__name__ == "PACT_QuantizedBatchNormNd":
            kappa_max = m.kappa.abs().max()
            lamda_max = m.lamda.abs().max()
            if minmax:
                out_range = max(m.max.abs(), m.min.abs())
            else: # if we do not have minmax statistics, approximate with alpha mult by a range_factor
                out_range = module_dict[calib_dict[n]].alpha * range_factor
            if kappa_dict is not None:
                kappa_bit_lim = kappa_dict[n]
            else:
                kappa_bit_lim = kappa_bit_default
            if lamda_dict is not None:
                lamda_bit_lim = lamda_dict[n]
            else:
                lamda_bit_lim = lamda_bit_default
            eps_lim = max(out_range, lamda_max) / (2**(lamda_bit_lim - 1) - 1)
            eps_kappa_lim = eps_lim / m.eps_in
            kappa_bits = min(int(min(torch.log2(1 + 2*kappa_max / eps_kappa_lim).floor(), kappa_bit_lim)), BN_PRECISION_MAX)
            lamda_bits = min(lamda_bit_lim, BN_PRECISION_MAX)
            m.precision_kappa.set_bits(kappa_bits)
            m.precision_lamda.set_bits(lamda_bits)
            # print("%s %d %d" % (n, kappa_bits, lamda_bits))

def _unfreeze_bn(self):
    r"""Sets :py:class:`torch.nn.BatchNorm2d` layers to collect statistics and update `running_var` and `running_mean`.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "BatchNorm2d":
            m.train()

def _prune_empty_bn_pact(self, bn_dict={}, threshold=None):
    if not bn_dict:
        bn_dict = get_bn_dict_from_supernodes(self)

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" or \
            m.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
            module_dict[n] = m
    for n_before in bn_dict.keys():
        n_after  = bn_dict[n_before]
        m_before = module_dict[n_before]
        m_after  = module_dict[n_after]
        if threshold is None:
            try:
                eps = (m_before.W_alpha + m_before.W_beta) / (2.**m_before.W_precision.get_bits()-1)
            except AttributeError:
                continue
        else:
            eps = threshold
        if m_before.bias is not None:
            continue
        try:
            m_after.kappa.data[m_before.weight.data.flatten(1).abs().max(1)[0] < eps] = 0.
        except AttributeError:
            m_after.weight.data[m_before.weight.data.flatten(1).abs().max(1)[0] < eps] = 0.
            m_after.running_var.data[m_before.weight.data.flatten(1).abs().max(1)[0] < eps] = 1.

def _fold_bn_pact(self, bn_dict={}, bn_inv_dict={}, eps=None, phi_inv=0., reset_alpha=True, remove_bn=False):
    r"""Performs batch-normalization folding following the algorithm presented in
    https://arxiv.org/abs/1905.04166. It performs both normal folding and inverse
    folding using two separate dictionaries `bn_dict` and `bn_inv_dict`.
    
    :param bn_dict: a dictionary of layer names, with the key being the source (linear) and the value the target (batch-norm). If empty (default), uses the graph to fold all BN layers.
    :type  bn_dict: `dict` or `collections.OrderedDict`
    :param bn_inv_dict: a dictionary of layers, with the key being the source (batch-norm) and the value the target (linear).
    :type  bn_inv_dict: `dict` or `collections.OrderedDict`
    :param verbose: if True, prints more information.
    :type  verbose: bool
    :param eps: if not None (the default), overrides numerical `eps` used within batch-norm layer.
    :type  eps: float
    :param phi_inv: parameter added to `gamma` in inverse folding for better numerical stability (default 0).
    :type  phi_inv: float
    :param reset_alpha: if True (default), reset the clipping parameters of weights.
    :type  reset_alpha: bool
    :param remove_bn: if True, replace all BN layers with identity layers to "physically remove" BN from the network (default False).
    :type  remove_bn: bool

    """

    if not bn_dict:
        bn_dict = get_bn_dict_from_supernodes(self)

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" or \
            m.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
            module_dict[n] = m
    param = {}
    bn_list = list(bn_dict.values()) + list(bn_inv_dict.keys())
    for n in bn_list:
        m = module_dict[n]
        # count how many time this occures as a value in the direct map (for bilinear functions e.g. add)
        count = list(bn_dict.values()).count(n)
        if eps is None:
            try:
                eps = m.eps
            except AttributeError:
                eps = 0.
        if m.__class__.__name__ == "PACT_QuantizedBatchNormNd":
            gamma = m.kappa.data[:].clone().detach().cpu().flatten()
            beta  = m.lamda.data[:].clone().detach().cpu().flatten()
            sigma = torch.ones_like(gamma).flatten()
            mu    = torch.zeros_like(beta).flatten()
        else:
            gamma = m.weight.data[:].clone().detach().cpu()
            beta  = m.bias.data[:].clone().detach().cpu()
            sigma = torch.sqrt(m.running_var.data[:] + eps).clone().detach().cpu()
            mu    = m.running_mean.data[:].clone().detach().cpu()
        param[n] = {
            'gamma' : gamma,
            'beta'  : beta, 
            'sigma' : sigma,
            'mu'    : mu,   
            'count' : count
        }
    # direct folding (CONV->BN)
    for n in bn_dict.keys():
        n_bn  = bn_dict[n]
        m     = module_dict[n]
        m_bn  = module_dict[n_bn]
        gamma = param[n_bn]['gamma']
        beta  = param[n_bn]['beta']
        mu    = param[n_bn]['mu']
        sigma = param[n_bn]['sigma']
        count = param[n_bn]['count']
        if count > 1:
            beta  = beta/count
            mu    = mu/count

        th_a = (gamma/sigma).to(m.weight.device)
        th_b = (beta-gamma/sigma * mu).to(m.weight.device)

        m.weight.data[:] = m.weight.data[:] * reshape_before(m, th_a)
        try:
            m.bias.data[:] = th_a * m.bias.data[:] + th_b
        except AttributeError:
            m.bias = torch.nn.Parameter(th_b)
    # inverse folding (BN->CONV)
    for n_bn in bn_inv_dict.keys():
        n    = bn_inv_dict[n_bn]
        m    = module_dict[n]
        m_bn = module_dict[n_bn]
        gamma = param[n_bn]['gamma']
        beta  = param[n_bn]['beta']
        mu    = param[n_bn]['mu']
        sigma = param[n_bn]['sigma']
        count = param[n_bn]['count']

        th_a = sigma/gamma
        shape_w = np.prod(np.asarray(m.weight.data.shape)[3:1])
        th_m_by_w   = (reshape_after(m, mu)*m.weight.data[:]).sum(3).sum(2).sum(1) / shape_w
        th_bsg_by_w = (reshape_after(m, beta*sigma/(gamma+phi_inv))*m.weight.data[:]).sum(3).sum(2).sum(1) / shape_w

        if phi_inv is None:
            phi_inv = m_bn.eps
        m.weight.data[:] = m.weight.data[:] * reshape_after(m, th_a)
        try:
            m.bias.data[:] = m.bias.data[:] + th_m_by_w - th_bsg_by_w
        except AttributeError:
            m.bias = torch.nn.Parameter(th_m_by_w - th_bsg_by_w)
    # neutralize BatchNorm's
    for n in bn_list:
        m = module_dict[n]
        if m.__class__.__name__ == "PACT_QuantizedBatchNormNd":
            m.kappa.data[:] = 1.
            m.lamda.data[:] = 0.
        else:
            m.weight.data[:] = 1.
            m.bias.data[:] = 0.
            m.running_mean.data[:] = 0.
            m.running_var.data[:] = (1. - eps)**2
    if reset_alpha:
        self.reset_alpha_weights()

def _threshold_folding_pact(self, act_dict):
    r"""Performs the folding of batch-normalization layers into threshold-based
    activation layers.
    
    :param act_dict: a dictionary of layer names, with the key being the source (batch-norm) and the value the target (activation).
    :type  act_dict: `dict` or `collections.OrderedDict`

    """

    module_dict = {}
            
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_ThresholdAct" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" ):
            module_dict[n] = m
    for n_before in act_dict.keys():
        n_after = act_dict[n_before]
        m_before = module_dict[n_before]
        m_after  = module_dict[n_after]
        # get BN parameters
        eps = m_before.eps
        gamma = m_before.weight.data[:]
        beta  = m_before.bias.data[:]
        sigma = torch.sqrt(m_before.running_var.data[:] + eps)
        mu    = m_before.running_mean.data[:]
        # setup threshold in PACT_ThresholdAct
        del m_after.kappa, m_after.lamda
        m_after.kappa = torch.nn.Parameter(torch.zeros(gamma.shape[0]).to(m_after.alpha.data.device))
        m_after.lamda = torch.nn.Parameter(torch.zeros(gamma.shape[0]).to(m_after.alpha.data.device))
        m_after.kappa.data[:] = sigma/gamma
        m_after.lamda.data[:] = mu - beta*sigma/gamma
        # remove BN parameters
        m_before.weight.data[:] = 1.
        m_before.bias.data[:] = 0.
        m_before.running_var.data[:] = (1. - eps)**2
        m_before.running_mean.data[:] = 0.
