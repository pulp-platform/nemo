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

# Part of the procedure necessary for DFQ as described here https://arxiv.org/pdf/1906.04721.pdf            
def _equalize_weights_dfq_pact(self, equalize_dict, verbose=False, cost_eps=1e-3, max_iter=1000):
    r"""This function implements the cross-layer weight-range equalization procedure proposed in 
    the Data-Free Quantization paper by Qualcomm (https://arxiv.org/pdf/1906.04721.pdf).
    It should be used only after batch-normalization layers have been folded into convolution
    by means of the `fold_bn` or `fold_bn_withinv` methods.
    
    :param equalize_dict: a dictionary of layer names, with the key being the source and the value the target layer.
    :type  equalize_dict: `dict` or `collections.OrderedDict`
    :param verbose: if True, prints more information.
    :type  verbose: bool
    :param cost_eps: equalization will iterate until the cost is less than this threshold or the number of iterations is greater than `max_iter`.
    :type  cost_eps: float
    :param max_iter: maximum number of iterations.
    :type  max_iter: int

    """

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "BatchNorm2d" or \
            m.__class__.__name__ == "BatchNorm1d" ):
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

            s = torch.sqrt(range_after/range_before)
            m_before.weight.data[:] = m_before.weight.data[:] * reshape_before(m_before, s)
            try:
                m_before.bias.data[:] = m_before.bias.data[:] * s
            except AttributeError:
                pass
            m_after.weight.data[:]  = m_after.weight.data[:] / reshape_after(m_after, s)
            new_prec_before_mean = (weight_range(m_before, 0).abs() / (m_before.weight.max() - m_before.weight.min()).abs()).sum().item()
            new_prec_after_mean  = (weight_range(m_after, 0).abs()  / (m_after.weight.max()  - m_after.weight.min()).abs()).sum().item()
            cost += np.abs(new_prec_before_mean*new_prec_after_mean - old_prec_before_mean*old_prec_after_mean)
        it += 1
        if verbose:
            logging.info("[DFQ Equalization] cost=%.5f" % cost)
    logging.info("[DFQ Equalization] terminated after %d iterations" % it)

def _equalize_weights_unfolding_pact(self, bn_dict={}, verbose=False, eps=None):
    r"""Performs cross-layer equalization by unfolding of convolution parameters
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
