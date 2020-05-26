#
# transform.py
# Francesco Conti <fconti@iis.ee.ethz.ch>
# Alfio Di Mauro <adimauro@iis.ee.ethz.ch>
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
import nemo
from nemo.transf.common import *
from nemo.transf.bias import *
from nemo.transf.bn import *
from nemo.transf.deploy import *
from nemo.transf.equalize import *
from nemo.transf.export import *
from nemo.transf.pruning import *
from nemo.transf.statistics import *
from nemo.transf.utils import *

def quantize_pact(module, W_bits=4, x_bits=4, dummy_input=None, remove_dropout=False, **kwargs):
    r"""Takes a PyTorch module and makes it quantization-aware with PACT, recursively.

    The function follows recursively the data structures containing PyTorch layers (typically as hierarchical lists, e.g.
    block-level :py:class:`torch.nn.Sequential` for networks like ResNet).
    It performs two main kinds of replacements:
    - linear layers like :py:class:`torch.nn.Conv2d`, :py:class:`torch.nn.Conv1d`, :py:class:`torch.nn.Linear` are replaced
    with quantization-aware versions (:py:class:`nemo.quant.pact.PACT_Conv2d`, :py:class:`nemo.quant.pact.PACT_Conv1d`,
    :py:class:`nemo.quant.pact.PACT_Linear`). By default, these layers quantize weights but not (input) activations.
    - activation layers like :py:class:`torch.nn.ReLU`, :py:class:`torch.nn.ReLU6` are replaced
    with a special quantized activation, :py:class:`nemo.quant.pact.PACT_Act` that performs both clipping and quantization.

    The returned layer exposes a series of methods typical of quantization-aware models:
    - `export_precision`, exports the precision of current layers as a dictionary.
    - `change_precision`, changes the target precision of the model.
    - `reset_alpha_weights`, resets the scaling factors for weights.
    - `get_clip_parameters`, returns scaling factor parameters of activations, that can be trained.
    - `get_nonclip_parameters`, returns all parameters except scaling factor parameters of activations.
    - `harden_weights`, hardens the quantized value of weights into their underlying float representation.
    - `prune_weights`, replaces a portion of the weights in the model with 0's.
    - `set_statistics_act`, setup activation layers in statistics collection mode (they run as normal ReLUs collecting
    the maximum value of activations, to calibrate the scaling parameters).
    - `get_statistics_act`, get collected activation layer statistics.
    - `unset_statistics_act`, setup activation layers to act normally as quantization-aware layers.
    - `reset_alpha_act`, uses the collected activation layer statistics to recalibrate the scaling parameters. 
    
    :param module: module to be transformed to use PACT quantization (typically, a container like :py:class:`torch.nn.ModuleList`).
    :type  module: `torch.nn.Module`
    
    :param W_bits: target precision for weights.
    :type  W_bits: float
    
    :param x_bits: target precision for activations.
    :type  x_bits: float
    
    :param dummy_input: dummy input tensor (default None). Used to derive an adjacency map by tracing
    :type  dummy_input: `torch.Tensor`

    :param remove_dropout: if True, removes dropout layers before graph construction.
    :type  remove_dropout: bool
    
    :return: The quantization-aware module.
    :rtype:  same as `module`

    """
    # if given a dummy input, get an adjacency map of the module and other useful things
    module.eval()
    if remove_dropout:
        module = nemo.transform.dropout_to_identity(module)
    if dummy_input is not None:
        module.graph = DeployGraph(module, dummy_input=dummy_input)
    else:
        module.graph = None
    module.stage = 'fq'
    module = _hier_quantizer_pact(module, module.graph, **kwargs)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    module.add_input_bias              = types.MethodType(nemo.transf.bias._add_input_bias_pact, module)
    module.remove_bias                 = types.MethodType(nemo.transf.bias._remove_bias_pact, module)
    module.remove_input_bias           = types.MethodType(nemo.transf.bias._remove_input_bias_pact, module) 
    module.freeze_bn                   = types.MethodType(nemo.transf.bn._freeze_bn, module)
    module.unfreeze_bn                 = types.MethodType(nemo.transf.bn._unfreeze_bn, module)
    module.absorb_affine_bn            = types.MethodType(nemo.transf.bn._absorb_affine_bn, module)
    module.prime_affine_bn             = types.MethodType(nemo.transf.bn._prime_affine_bn, module)
    module.fold_bn                     = types.MethodType(nemo.transf.bn._fold_bn_pact, module)
    module.fold_bn_withinv             = types.MethodType(nemo.transf.bn._fold_bn_pact, module)
    module.fold_thresholds             = types.MethodType(nemo.transf.bn._threshold_folding_pact, module)
    module.prune_empty_bn              = types.MethodType(nemo.transf.bn._prune_empty_bn_pact, module)
    module.calibrate_bn                = types.MethodType(nemo.transf.bn._calibrate_bn_pact, module)
    module.get_eps_at                  = types.MethodType(nemo.transf.deploy._get_eps_at, module)
    module.set_eps_in                  = types.MethodType(nemo.transf.deploy._set_eps_in_pact, module)
    module.round_weights               = types.MethodType(nemo.transf.deploy._round_weights_pact, module)
    module.harden_weights              = types.MethodType(nemo.transf.deploy._harden_weights_pact, module)
    module.set_deployment              = types.MethodType(nemo.transf.deploy._set_deployment_pact, module)
    module.qd_stage                    = types.MethodType(nemo.transf.deploy._qd_stage, module)
    module.id_stage                    = types.MethodType(nemo.transf.deploy._id_stage, module)
    module.export_precision            = types.MethodType(nemo.transf.export._export_precision, module)
    module.export_weights_legacy_int16 = types.MethodType(nemo.transf.export._export_weights_legacy_int16, module)
    module.change_precision            = types.MethodType(nemo.transf.utils._change_precision_pact, module)
    module.reset_alpha_weights         = types.MethodType(nemo.transf.utils._reset_alpha_weights_pact, module)
    module.reset_alpha_act             = types.MethodType(nemo.transf.utils._reset_alpha_act_pact, module)
    module.get_clip_parameters         = types.MethodType(nemo.transf.utils._get_clip_parameters_pact, module)
    module.get_nonclip_parameters      = types.MethodType(nemo.transf.utils._get_nonclip_parameters_pact, module)
    module.set_train_loop              = types.MethodType(nemo.transf.utils._set_train_loop_pact, module)
    module.unset_train_loop            = types.MethodType(nemo.transf.utils._unset_train_loop_pact, module)
    module.prune_weights               = types.MethodType(nemo.transf.pruning._prune_weights_pact, module)
    module.equalize_weights_dfq        = types.MethodType(nemo.transf.equalize._equalize_weights_dfq_pact, module)
    module.equalize_weights_lsq        = types.MethodType(nemo.transf.equalize._equalize_weights_lsq_pact, module)
    module.equalize_weights_unfolding  = types.MethodType(nemo.transf.equalize._equalize_weights_unfolding_pact, module)
    module.statistics_act              = types.MethodType(nemo.transf.statistics._statistics_act_pact, module)
    module.set_statistics_act          = types.MethodType(nemo.transf.statistics._set_statistics_act_pact, module)
    module.get_statistics_act          = types.MethodType(nemo.transf.statistics._get_statistics_act_pact, module)
    module.unset_statistics_act        = types.MethodType(nemo.transf.statistics._unset_statistics_act_pact, module)
    module.statistics_bn               = types.MethodType(nemo.transf.statistics._statistics_bn_pact, module)
    module.set_statistics_bn           = types.MethodType(nemo.transf.statistics._set_statistics_bn_pact, module)
    module.unset_statistics_bn         = types.MethodType(nemo.transf.statistics._unset_statistics_bn_pact, module)
    module.W_precision = Precision(W_bits, None)
    module.x_precision = Precision(x_bits, None)
    return module

def _hier_replacer(module, name, replacement):
    for n,m in module.named_children():
        if n == name:
            module._modules[n] = replacement()
        elif n == name.split('.')[0]:
            module._modules[n] = _hier_replacer(m, '.'.join(name.split('.')[1:]), replacement)
    return module

def _hier_bn_to_identity(module):
    if module.__class__.__name__ == 'BatchNorm2d' or \
       module.__class__.__name__ == 'BatchNorm1d':
        module = PACT_Identity()
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_bn_to_identity(m)
        return module

def _hier_dropout_to_identity(module):
    if module.__class__.__name__ == 'Dropout':
        module = PACT_Identity()
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_dropout_to_identity(m)
        return module

def _hier_bn_quantizer(module, **kwargs):
    if module.__class__.__name__ == 'BatchNorm2d' or \
        module.__class__.__name__ == 'BatchNorm1d':
        gamma = module.weight.data[:].clone().detach()
        beta = module.bias.data[:].clone().detach()
        sigma = torch.sqrt(module.running_var.data[:] + module.eps).clone().detach()
        mu = module.running_mean.data[:].clone().detach()
        dimensions = 1 if module.__class__.__name__ == 'BatchNorm1d' else 2
        module = PACT_QuantizedBatchNormNd(kappa=gamma/sigma, lamda=beta-gamma/sigma*mu, dimensions=dimensions, **kwargs)
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_bn_quantizer(m, **kwargs)
        return module

def _hier_bn_dequantizer(module):
    if module.__class__.__name__ == 'PACT_QuantizedBatchNormNd':
        gamma = module.kappa.data[:].clone().detach().flatten()
        beta = module.lamda.data[:].clone().detach().flatten()
        module = torch.nn.BatchNorm2d(weight=gamma, bias=beta)
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_bn_dequantizer(m)
        return module

def _hier_integerizer(module, **kwargs):
    if (module.__class__.__name__ == "PACT_Conv2d" or \
        module.__class__.__name__ == "PACT_Conv1d" or \
        module.__class__.__name__ == "PACT_Linear"):
        module.integerize_weights(**kwargs)
        return module
    elif (module.__class__.__name__ == "PACT_QuantizedBatchNormNd"):
        module = PACT_IntegerBatchNormNd(kappa=module.kappa, lamda=module.lamda, eps_in=module.eps_in, eps_kappa=module.eps_kappa, eps_lamda=module.eps_lamda)
        module.integerize_weights(**kwargs)
    elif (module.__class__.__name__ == "PACT_Act"):
        module = PACT_IntegerAct(precision=module.precision, eps_in=module.eps_in, eps_out=module.eps_static, alpha=module.alpha_static, **kwargs)
        module.set_output_eps(**kwargs)
    elif (module.__class__.__name__ == "PACT_IntegerAdd"):
        module.integerized = True
    elif (module.__class__.__name__ == "AvgPool2d"):
        module = PACT_IntegerAvgPool2d(module.kernel_size, stride=module.stride, padding=module.padding, ceil_mode=module.ceil_mode,
            count_include_pad=module.count_include_pad)
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_integerizer(m, **kwargs)
    return module

def _hier_thresholdizer_pact(module):
    if module.__class__.__name__ == 'PACT_Act':
        module = PACT_ThresholdAct(precision=module.precision, alpha=module.alpha.data[:])
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_thresholdizer_pact(m)
        return module

def _hier_quantizer_pact(module, graph=None, **kwargs):
    if module.__class__.__name__ == 'Conv2d':
        W = module.weight.data
        try:
            b = module.bias.data
        except AttributeError:
            b = None
        module = PACT_Conv2d(
            module.in_channels,
            module.out_channels,
            _single(module.kernel_size),
            stride=_single(module.stride),
            padding=_single(module.padding),
            dilation=_single(module.dilation),
            groups=module.groups,
            bias=True if module.bias is not None else False
        )
        module.weight.data = W.clone()
        if b is not None:
            module.bias.data = b.clone()
        return module
    if module.__class__.__name__ == 'Conv1d':
        W = module.weight.data
        try:
            b = module.bias.data
        except AttributeError:
            b = None
        module = PACT_Conv1d(
            module.in_channels,
            module.out_channels,
            _single(module.kernel_size),
            stride=_single(module.stride),
            padding=_single(module.padding),
            dilation=_single(module.dilation),
            groups=module.groups,
            bias=True if module.bias is not None else False
        )
        module.weight.data = W.clone()
        if b is not None:
            module.bias.data = b.clone()
        return module
    if module.__class__.__name__ == 'Linear':
        W = module.weight.data
        try:
            b = module.bias.data
        except AttributeError:
            b = None
        module = PACT_Linear(
            module.in_features,
            module.out_features,
            bias=True if module.bias is not None else False
        )
        module.weight.data = W.clone()
        if b is not None:
            module.bias.data = b.clone()
        return module
    elif module.__class__.__name__ == 'ReLU6':
        module = PACT_Act(alpha=6., **kwargs)
        return module
    elif module.__class__.__name__ == 'ReLU':
        module = PACT_Act(**kwargs)
        return module
    elif module.__class__.__name__ == 'LeakyReLU':
        module = PACT_Act(leaky=module.negative_slope, **kwargs)
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_quantizer_pact(m, **kwargs)
        return module

def _hier_dequantizer_pact(module):
    if module.__class__.__name__ == 'PACT_Conv2d':
        W = module.weight.data
        try:
            b = module.bias.data
        except AttributeError:
            b = None
        module = torch.nn.Conv2d(
            module.in_channels,
            module.out_channels,
            _single(module.kernel_size),
            stride=_single(module.stride),
            padding=_single(module.padding),
            dilation=_single(module.dilation),
            groups=module.groups,
            bias=True if module.bias is not None else False
        )
        module.weight.data = W.clone()
        if b is not None:
            module.bias.data = b.clone()
        return module
    if module.__class__.__name__ == 'PACT_Linear':
        W = module.weight.data
        try:
            b = module.bias.data
        except AttributeError:
            b = None
        module = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=True if module.bias is not None else False
        )
        module.weight.data = W.clone()
        if b is not None:
            module.bias.data = b.clone()
        return module
    elif module.__class__.__name__ == 'PACT_Act':
        module = torch.nn.ReLU()
        return module
    else:
        for n,m in module.named_children():
            module._modules[n] = _hier_dequantizer_pact(m)
        return module

def _hier_flat_dict_build(module, name):
    for n,m in module.named_children():
        if n == name:
            return m
        elif n == name.split('.')[0]:
            return _hier_flat_dict_build(m, '.'.join(name.split('.')[1:]))
    return module

def integerize_pact(module, eps_in, **kwargs):
    # r"""Takes a PyTorch module in q-deploy stage and makes it integerized, recursively.

    # :param eps_in: input quantum :math:`\varepsilon_{in}`.
    # :type  eps_in: :py:class:`torch.Tensor`
    # :return: output quantum :math:`\varepsilon_{out}`.
    # :rtype:  :py:class:`torch.Tensor`

    # """
    try:
        net = module.module
    except AttributeError:
        net = module
    # assert(hasattr(net, 'graph'))
    # assert(net.graph is not None)
    net.set_eps_in(eps_in)
    net = _hier_integerizer(net, **kwargs)
    net.graph.rebuild_module_dict()
    # if hasattr(module, 'model'):
    #     module.model = net
    # else:
    #     module = net
    return module

def dequantize_pact(module):
    module = _hier_dequantizer_pact(module)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    return module

def thresholdize_pact(module, act_dict):
    module = _hier_thresholdizer_pact(module)
    module.fold_thresholds(act_dict)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    return module

def bn_to_identity(module):
    module = _hier_bn_to_identity(module)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    return module

def dropout_to_identity(module):
    module = _hier_dropout_to_identity(module)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    return module

def bn_quantizer(module, **kwargs):
    module = _hier_bn_quantizer(module, **kwargs)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    return module

def bn_dequantizer(module):
    module = _hier_bn_dequantizer(module)
    if hasattr(module, 'graph'):
        if module.graph is not None:
            module.graph.rebuild_module_dict()
    return module
