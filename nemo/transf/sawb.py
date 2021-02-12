#
# sawb.py
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2018-2021 ETH Zurich and University of Bologna
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

# first entry is c1, second is c2 with alpha_w^* = c1 * sqrt(Ew2) - c2 * Ew1
__sawb_asymm_lut = {
    2: [8.356, 7.841],
    3: [4.643, 3.729],
    4: [8.356, 7.841],
    5: [12.522, 12.592],
    6: [15.344, 15.914],
    7: [19.767, 21.306],
    8: [26.294, 29.421]
}

# Disable gradients for alpha,beta params
def _disable_grad_sawb(self, layer_bits={}):

    # Colab with SAWB LUT: https://colab.research.google.com/drive/1UEQnvVcSP3N-QTZLEumbbGCv_oLv-JtL
    module_dict = {}
    use_default = False
    if not layer_bits:
        layer_bits = {}
        use_default = True
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            m.W_alpha.requires_grad = False
            m.W_beta.requires_grad = False

# Set weight clipping parameters according to Statistics-Aware Weight Binning
def _weight_clip_sawb(self, asymmetric=True, layer_bits={}, check_minmax=True, verbose=False):

    # Colab with SAWB LUT: https://colab.research.google.com/drive/1UEQnvVcSP3N-QTZLEumbbGCv_oLv-JtL
    module_dict = {}
    use_default = False
    if not layer_bits:
        layer_bits = {}
        use_default = True
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            if use_default:
                module_dict[n] = m
                layer_bits[n] = m.W_precision.get_bits()
            elif n in layer_bits.keys():
                module_dict[n] = m

    for n in module_dict.keys():
        m = module_dict[n]
        # compute E[|w|]
        Ew1 = m.weight.abs().mean()
        # compute E[w^2]
        Ew2 = (m.weight.abs() ** 2).mean()
        # compute alpha
        alpha = __sawb_asymm_lut[layer_bits[n]][0] * torch.sqrt(Ew2) - __sawb_asymm_lut[layer_bits[n]][1] * Ew1
        # compute beta
        eps = 2*alpha / (2**layer_bits[n])
        if asymmetric:
            beta = alpha + eps * (2**layer_bits[n]-1)
        else:
            beta = alpha + eps * 2**layer_bits[n]
        if check_minmax:
            m.W_alpha.data[:] = min(alpha, m.weight.min().abs())
            m.W_beta.data[:]  = min(beta, m.weight.max().abs())
        else:
            m.W_alpha.data[:] = alpha
            m.W_beta.data[:]  = beta
        if verbose:
            print("[weight clip SAWB] %s: Ew1=%.3e Ew2=%.3e alpha=%.3e beta=%.3e" % (n, Ew1, Ew2, m.W_alpha.data.item(), m.W_beta.data.item()))

