#
# utils.py
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
import nemo
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

def _change_precision_pact(self, bits=4, scale_activations=True, scale_weights=True, verbose=True, reset_alpha=True, min_prec_dict=None, **kwargs):
    r"""Changes the target precision of a PACT quantization-aware layer.

    
    :param bits: target bit-width.
    :type  bits: `int`

    :param scale_activations: if False, do not change precision of activations (default True).
    :type  scale_activations: boolean

    :param scale_weights: if False, do not change precision of weights (default True).
    :type  scale_weights: boolean

    :param verbose: if False, do not log precision information (default True).
    :type  verbose: boolean

    :param reset_alpha: if False, do not reset weight scale parameter upon precision change (default True).
    :type  reset_alpha: boolean

    :param min_prec_dict: dictionary of minimum layer-by-layer precisions (default None).
    :type  min_prec_dict: dictionary

    """
    if scale_activations and bits is not None:
        self.x_precision.bits = bits
    if scale_weights and bits is not None:
        self.W_precision.bits = bits
    for n,m in self.named_modules():
        min_prec_x = copy.deepcopy(self.x_precision)
        min_prec_W = copy.deepcopy(self.W_precision)
        if min_prec_dict is not None:
            try:
                min_prec_x.bits = min_prec_dict[n]['x_bits']
            except KeyError:
                pass
            try:
                min_prec_W.bits = min_prec_dict[n]['W_bits']
            except KeyError:
                pass
        if m.__class__.__name__ == "PACT_Act" and scale_activations:
            m.precision = max(self.x_precision, min_prec_x)
        if scale_weights and (m.__class__.__name__ == "PACT_Conv2d" or \
                              m.__class__.__name__ == "PACT_Conv1d" or \
                              m.__class__.__name__ == "PACT_Linear"):
            m.W_precision = max(self.W_precision, min_prec_W)
            if reset_alpha:
                m.reset_alpha_weights()
        if verbose and (m.__class__.__name__ == "PACT_Act") and scale_activations:
            try:
                logging.info("[Quant]\t\t %s: x_bits=%.2f" % (n, m.precision.get_bits()))
            except AttributeError:
                pass
        if verbose and scale_weights and (m.__class__.__name__ == "PACT_Conv2d" or \
                                          m.__class__.__name__ == "PACT_Conv1d" or \
                                          m.__class__.__name__ == "PACT_Linear"):
            try:
                logging.info("[Quant]\t\t %s: W_bits=%.2f" % (n, m.W_precision.get_bits()))
            except AttributeError:
                pass

def _set_train_loop_pact(self):
    r"""Sets modules so that weights are not treated like hardened (e.g., for training).
    
    """

    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" ):
            m.train_loop = True

def _unset_train_loop_pact(self):
    r"""Sets modules so that weights are treated like hardened (e.g., for evaluation).
    
    """

    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear" ):
            m.train_loop = False
            m.train_loop_oldprec = float(m.W_beta.item()+m.W_alpha.item())/(2.0**(m.W_precision.get_bits())-1)

def _reset_alpha_act_pact(self, **kwargs):
    r"""Resets :py:class:`nemo.quant.PACT_Act` parameter `alpha` the value collected through statistics.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "PACT_Act":
            m.reset_alpha(**kwargs)

def _get_nonclip_parameters_pact(self):
    r"""Yields all parameters except for `alpha` values.
    
    """

    for name, param in self.named_parameters(recurse=True):
        if name[-5:] != 'alpha':
            yield param

def _get_clip_parameters_pact(self):
    r"""Yields all `alpha` parameters.
    
    """

    for name, param in self.named_parameters(recurse=True):
        if name[-5:] == 'alpha':
            yield param

def _reset_alpha_weights_pact(self, method='standard', **kwargs):
    r"""Resets parameter `W_alpha`.
    
    """

    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            m.reset_alpha_weights(**kwargs)

