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
            m.set_static_precision(**kwargs)

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

def _qd_stage(self, eps_in, add_input_bias_dict={}, remove_bias_dict={}, prune_empty_bn=True, int_accurate=True, bn_calibration_fn=None, bn_calibration_range_factor=8, **kwargs):
    r"""High-level function to move the network from FQ to QD stage.

    :param eps_in: Input precision quantum (required).
    :type  eps_in: float
    :param add_input_bias_dict: dictionary of layers to which an input bias must be added (layer name as key, bias as value).
    :type  add_input_bias_dict: dict or `collections.OrderedDict`
    :param remove_bias_dict: dictionary of Linear->BatchNorm couples where bias must be absorbed by the BatchNorm (Linear name as key, BatchNorm name as value).
    :type  remove_bias_dict: dict or `collections.OrderedDict`
    :param prune_empty_bn: if True (default), BatchNorm channel multiplicative parameters are pruned if smaller than 1e-9.
    :type  prune_empty_bn: bool
    :param int_accurate: if True (default), target an accurate representation of ID numerical dynamics (e.g., with requantization) at QD stage.
    :type  int_accurate: bool
    :param bn_calibration_fn: if not None (default), a function (e.g., calling validation) used to calibrate BatchNorm range.
    :type  bn_calibration_fn: function
    :param bn_calibration_range_factor: if bn_calibration_fn is None, multiply the clipping parameter of the following Activation multiplied by bn_calibration_range_factor to estimate BatchNorm range.
    :type  bn_calibration_range_factor: int

    """

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
    else:  # this is mainly useful for debug purposes, to identify misalignments FQ/QD
        for n,m in self.named_modules():
            if (m.__class__.__name__ == "PACT_Act"):
                m.precise = True
    self.set_deployment(eps_in=eps_in, **kwargs) # with initial BN eps
    if bn_calibration_fn is not None:
        with self.statistics_bn():
            bn_calibration_fn()
        self.calibrate_bn(**kwargs)
    else:
        self.calibrate_bn(minmax=False, range_factor=bn_calibration_range_factor, **kwargs)
    self.set_deployment(eps_in=eps_in, **kwargs) # repeat, to fix BN eps
    self.harden_weights()

def _id_stage(self, eps_in=None, **kwargs):
    r"""High-level function to move the network from QD to ID stage.

    :param eps_in: Input precision quantum, default None (will use the previously saved eps_in).
    :type  eps_in: float

    """

    if self.stage == 'fq':
        self.qd_stage(eps_in=eps_in, **kwargs)
    self.stage = 'id'
    if eps_in is None:
        eps_in = self.eps_in
    nemo.transform.integerize_pact(self, eps_in=eps_in, **kwargs)

