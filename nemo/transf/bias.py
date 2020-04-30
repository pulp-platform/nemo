#
# bias.py
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

def _add_input_bias_pact(self, lin_dict, eps_in=None):
    r"""Adds a bias to compensate the asymmetry of an input activation tensor.
    
    :param lin_dict: a dictionary, where the key is the linear layer name and the value is the related activation translation.
    :type  lin_dict: `dict` or `collections.OrderedDict`

    """

    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" or \
            m.__class__.__name__ == "ConstantPad2d"):
            module_dict[n] = m
    # print(lin_dict)
    for n in lin_dict.keys():
        m = module_dict[n]
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            try:
                m.bias.data[:] = m.bias.data[:] - lin_dict[n] * m.weight.data[:].sum(3).sum(2).sum(1)
            except AttributeError:
                m.bias = torch.nn.Parameter(-lin_dict[n] * m.weight.data[:].sum(3).sum(2).sum(1))
            if eps_in is None:
                m.padding_value = lin_dict[n]
            else:
                m.padding_value = math.floor(lin_dict[n]/eps_in)*eps_in
        elif eps_in is None:
            m.value = lin_dict[n]
        else:
            m.value = math.floor(lin_dict[n]/eps_in)*eps_in
    self.input_bias_dict = lin_dict

def _remove_input_bias_pact(self):
    module_dict = {}
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            module_dict[n] = m
    for n in self.input_bias_dict.keys():
        m = module_dict[n]
        m.bias.data[:] = m.bias.data[:] + self.input_bias_dict[n] * m.weight.data[:].sum(3).sum(2).sum(1)
        m.padding_value = 0
    self.input_bias_dict = None

def _remove_bias_pact(self, bn_dict={}):
    r"""Folds the bias of a linear layer into the parameters of a following batch-norm.
    
    :param bn_dict: a dictionary of layer names, with the key being the source (linear) and the value the target (batch-norm).
    :type  bn_dict: `dict` or `collections.OrderedDict`

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
        m_after.running_mean.data[:] -= m_before.bias.data[:]
        m_before.bias = None

