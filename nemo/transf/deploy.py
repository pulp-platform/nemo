#
# deploy.py
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
import nemo

def _get_eps_at(self, *args, **kwargs):
    if hasattr(self, 'graph'):
        if self.graph is not None:
            return self.graph.get_eps_at(*args, **kwargs)

def _harden_weights_pact(self, **kwargs):
    r"""Harden all weights in the network to their quantized value.

    """

    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            m.train_loop_oldprec = float(m.W_beta.item()+m.W_alpha.item())/(2.0**(m.W_precision.get_bits())-1)
            m.harden_weights(**kwargs)
        if (m.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
            m.harden_weights(**kwargs)

def _round_weights_pact(self, **kwargs):
    r"""Round all weights in the network adding 1/2 an eps.

    """

    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            m.weight.data[:] += (m.W_beta.item()+m.W_alpha.item())/(2.0**(m.W_precision.get_bits())-1) / 2
        if (m.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
            m.kappa.data[:] += m.eps_kappa/2
            m.lamda.data[:] += m.eps_lamda/2

def _set_deployment_pact(self, eps_in, only_activations=False, **kwargs):
    r"""Sets the network in deployment mode, enabling saving it to ONNX format or similar.

    :param eps_in: Input precision quantum.
    :type  eps_in: float

    """

    self.stage = 'qd'
    if not only_activations:
        self.eps_in = eps_in
        self.set_eps_in(eps_in)
    for n,m in self.named_modules():
        if ((not only_activations and m.__class__.__name__ == "PACT_Conv2d") or \
            (not only_activations and m.__class__.__name__ == "PACT_Conv1d") or \
            (not only_activations and m.__class__.__name__ == "PACT_Linear") or \
            (not only_activations and m.__class__.__name__ == "PACT_IntegerAdd") or \
                                      m.__class__.__name__ == "PACT_Act"):
            m.deployment = True
        if (m.__class__.__name__ == "PACT_Act"):
            m.set_static_precision()

def _set_eps_in_pact(self, eps_in):
    r"""Sets the input precision quantum of the network.

    :param eps_in: Input precision quantum.
    :type  eps_in: float

    """

    assert(hasattr(self, 'graph'))
    assert(self.graph is not None)
    self.graph.rebuild_module_dict()
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Act"):
            m.eps_in = torch.tensor(self.get_eps_at(n, eps_in).item(), requires_grad=False)
        if (m.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
            m.eps_in = torch.tensor(self.get_eps_at(n, eps_in).item(), requires_grad=False)
        if (m.__class__.__name__ == "PACT_IntegerAdd"):
            eps_in_tmp = self.get_eps_at(n, eps_in)
            eps_in_list = []
            for eps in eps_in_tmp:
                eps_in_list.append(torch.tensor(eps.item(), requires_grad=False))
            m.eps_in_list = eps_in_list

def _qd_stage(self, eps_in=None, add_input_bias_dict={}, remove_bias_dict={}, prune_empty_bn=True, int_accurate=True, **kwargs):
    if prune_empty_bn:
        self.prune_empty_bn(threshold=1e-9)
    self.round_weights()
    self.harden_weights()
    if add_input_bias_dict:
        self.add_input_bias(add_input_bias_dict)
    if remove_bias_dict:
        self.remove_bias(remove_bias_dict)
    if int_accurate:
        nemo.transform.bn_quantizer(self, **kwargs)
        # harden_weights repeated to harden also BN parameters
        self.harden_weights()
    else:  # this is mainly useful for debug purposes, to identify misalignments FQ/QD
        for n,m in self.named_modules():
            if (m.__class__.__name__ == "PACT_Act"):
                m.precise = True
    self.set_deployment(eps_in=eps_in)

def _id_stage(self, eps_in=None, **kwargs):
    if self.stage == 'fq':
        self.qd_stage(eps_in=eps_in, **kwargs)
    self.stage = 'id'
    if eps_in is None:
        eps_in = self.eps_in
    nemo.transform.integerize_pact(self, eps_in=eps_in)

