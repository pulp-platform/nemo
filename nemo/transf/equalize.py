#
# equalize.py
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
from sklearn import linear_model

# Part of the procedure necessary for DFQ as described here https://arxiv.org/pdf/1906.04721.pdf            
def _equalize_weights_dfq_pact(self, equalize_dict={}, act_dict={}, verbose=False, cost_eps=1e-3, max_iter=1000, reset_alpha=True):
    r"""This function implements the cross-layer weight-range equalization procedure proposed in 
    the Data-Free Quantization paper by Qualcomm (https://arxiv.org/pdf/1906.04721.pdf).
    It should be used only after batch-normalization layers have been folded into convolution
    by means of the `fold_bn` or `fold_bn_withinv` methods.
    
    :param equalize_dict: a dictionary of layer names, with the key being a Linear and the value the next Linear layer.
    :type  equalize_dict: `dict` or `collections.OrderedDict`
    :param act_dict: a dictionary of layer names, with the key being a Linear and the value the next Act layer. If empty, activation alpha scaling is not performed unless `equalize_dict` is also empty.
    :type  act_dict: `dict` or `collections.OrderedDict`
    :param verbose: if True, prints more information.
    :type  verbose: bool
    :param cost_eps: equalization will iterate until the cost is less than this threshold or the number of iterations is greater than `max_iter`.
    :type  cost_eps: float
    :param max_iter: maximum number of iterations.
    :type  max_iter: int
    :param reset_alpha: if True, reset the clipping parameters of weights (default True).
    :type  reset_alpha: bool

    """

    if not equalize_dict:
        equalize_dict, act_dict = get_equalize_dict_from_supernodes(self)

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" or \
            m.__class__.__name__ == "PACT_Act"):
            module_dict[n] = m
    it = 0
    cost = 1e10
    while cost > cost_eps and it < max_iter:
        cost = 0.0
        for n_before in equalize_dict.keys():
            n_after = equalize_dict[n_before]
            m_before = module_dict[n_before]
            m_after  = module_dict[n_after]
            range_before = weight_range(m_before, 0)
            range_after  = weight_range(m_after, 1)
            old_prec_before_mean = (weight_range(m_before, 0).abs() / (m_before.weight.max() - m_before.weight.min()).abs()).sum().item()
            old_prec_after_mean  = (weight_range(m_after, 0).abs()  / (m_after.weight.max()  - m_after.weight.min()).abs()).sum().item()

            # this happens when the two layers are across a Flatten operation
            flatten_flag = False
            if range_after.shape[0] != range_before.shape[0]:
                flatten_flag = True
                range_after = range_after.reshape((range_before.shape[0], -1))
                flatten_dim = range_after.shape[1]
                range_after = range_after.max(1)[0]

            s = torch.sqrt(range_after/range_before)
            m_before.weight.data[:] = m_before.weight.data[:] * reshape_before(m_before, s)
            if act_dict:
                # per-layer: has to use s max!
                module_dict[act_dict[n_before]].alpha.data[:] *= s.max()
            try:
                m_before.bias.data[:] = m_before.bias.data[:] * s
            except AttributeError:
                pass

            if flatten_flag:
                s = torch.cat(flatten_dim*(s.unsqueeze(1),),1).flatten()

            m_after.weight.data[:]  = m_after.weight.data[:] / reshape_after(m_after, s)
            new_prec_before_mean = (weight_range(m_before, 0).abs() / (m_before.weight.max() - m_before.weight.min()).abs()).sum().item()
            new_prec_after_mean  = (weight_range(m_after, 0).abs()  / (m_after.weight.max()  - m_after.weight.min()).abs()).sum().item()
            cost += np.abs(new_prec_before_mean*new_prec_after_mean - old_prec_before_mean*old_prec_after_mean)
        it += 1
        if verbose:
            logging.info("[DFQ Equalization] cost=%.5f" % cost)
    logging.info("[DFQ Equalization] terminated after %d iterations" % it)
    if reset_alpha:
        self.reset_alpha_weights()

def _equalize_weights_unfolding_pact(self, bn_dict={}, verbose=False, eps=None):
    r"""Performs in-layer equalization by unfolding of convolution parameters
    into batch-normalization parameters.
    
    :param bn_dict: a dictionary of layer names, with the key being the source (linear) and the value the target (batch-norm).
    :type  bn_dict: `dict` or `collections.OrderedDict`
    :param verbose: if True, prints more information.
    :type  verbose: bool
    :param eps: if not None (the default), overrides numerical `eps` used within batch-norm layer.
    :type  eps: float

    """

    if not bn_dict:
        bn_dict = get_bn_dict_from_supernodes(self)

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" ):
            module_dict[n] = m
    for n_before in bn_dict.keys():
        n_after = bn_dict[n_before]
        m_before = module_dict[n_before]
        m_after  = module_dict[n_after]
        if eps is None:
            eps = m_after.eps
        range_before = weight_range(m_before, 0)
        if verbose:
            logging.info("[Equalization by Unfolding] %s: wrange_min=%.5f wrange_max=%.5f" % (n_before, range_before.min().item(), range_before.max().item()))
        m_before.weight.data[:] = m_before.weight.data[:] / reshape_before(m_before, range_before)
        try:
            m_before.bias.data[:] = m_before.bias.data[:] / range_before
        except AttributeError:
            pass
        m_after.running_mean.data[:] = m_after.running_mean.data[:] / range_before
        m_after.weight.data[:] = m_after.weight.data[:] * reshape_after(m_after, range_before)
        if verbose:
            logging.info("[Equalization by Unfolding] %s: wrange_min=%.5f wrange_max=%.5f" % (n_before, weight_range(m_before, 0).min().item(), weight_range(m_before, 0).max().item()))

def _equalize_weights_lsq_pact(self, bn_dict={}, verbose=False, eps=None):
    r"""Performs in-layer equalization by unfolding of convolution parameters
    into batch-normalization parameters.
    
    :param bn_dict: a dictionary of layer names, with the key being the source (linear) and the value the target (batch-norm).
    :type  bn_dict: `dict` or `collections.OrderedDict`
    :param verbose: if True, prints more information.
    :type  verbose: bool
    :param eps: if not None (the default), overrides numerical `eps` used within batch-norm layer.
    :type  eps: float

    """

    if not bn_dict:
        bn_dict = get_bn_dict_from_supernodes(self)

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" ):
            module_dict[n] = m
    for n_before in bn_dict.keys():
        n_after = bn_dict[n_before]
        m_before = module_dict[n_before]
        m_after  = module_dict[n_after]
        if eps is None:
            eps = m_after.eps
        min_before = weight_min(m_before, 0).cpu().detach().numpy()
        max_before = weight_max(m_before, 0).cpu().detach().numpy()
        if verbose:
            logging.info("[Equalization by Least Squares] %s: wrange_min=%.5f wrange_max=%.5f" % (n_before, weight_range(m_before, 0).min().item(), weight_range(m_before, 0).max().item()))
        X = np.vstack((min_before, max_before))
        y = np.asarray((-1,1))
        coeff = torch.zeros(len(min_before), device=m_before.weight.device)
        regr = linear_model.LinearRegression(fit_intercept=False)
        for i in range(len(min_before)):
            regr.fit(X[:,i].reshape((-1,1)), y)
            coeff[i] = torch.as_tensor(regr.coef_[0], device=m_before.weight.device)
        coeff = 1./coeff
        m_before.weight.data[:] = m_before.weight.data[:] / reshape_before(m_before, coeff)
        try:
            m_before.bias.data[:] = m_before.bias.data[:] / coeff
        except AttributeError:
            pass
        m_after.running_mean.data[:] = m_after.running_mean.data[:] / coeff
        m_after.weight.data[:] = m_after.weight.data[:] * reshape_after(m_after, coeff)
        if verbose:
            logging.info("[Equalization by Least Squares] %s: wrange_min=%.5f wrange_max=%.5f" % (n_before, weight_range(m_before, 0).min().item(), weight_range(m_before, 0).max().item()))

