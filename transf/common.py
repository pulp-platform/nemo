#
# common.py
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

__all__ = [ "reshape_before", "reshape_after", "weight_range", "onnx_name_2_pytorch_name" ]

def reshape_before(m, s):
    if   m.__class__.__name__ == "PACT_Conv2d":
        return s.reshape((s.shape[0],1,1,1))
    elif m.__class__.__name__ == "PACT_Conv1d":
        return s.reshape((s.shape[0],1,1))
    elif m.__class__.__name__ == "PACT_Linear":
        return s.reshape((1,s.shape[0]))
    else:
        return s

def reshape_after(m, s):
    if   m.__class__.__name__ == "PACT_Conv2d":
        if m.groups == s.shape[0]:
            # dwc
            return s.reshape((s.shape[0],1,1,1))
        else:
            return s.reshape((1,s.shape[0],1,1))
    elif m.__class__.__name__ == "PACT_Conv1d":
        return s.reshape((1,s.shape[0],1))
    elif m.__class__.__name__ == "PACT_Linear":
        return s.reshape((s.shape[0],1))
    else:
        return s

def weight_range(m, range_idx, symmetric=False):
    if   m.__class__.__name__ == "PACT_Conv2d":
        if m.groups == m.weight.shape[0]:
            range_idx = 0 # for DW-conv, always marginalize idx 1
        if not symmetric:
            return m.weight.max(3)[0].max(2)[0].max(1-range_idx)[0] - m.weight.min(3)[0].min(2)[0].min(1-range_idx)[0]
        else:
            return 2*m.weight.abs().max(3)[0].max(2)[0].max(1-range_idx)[0]
    elif m.__class__.__name__ == "PACT_Conv1d":
        if not symmetric:
            return m.weight.max(2)[0].max(1-range_idx)[0] - m.weight.min(2)[0].min(1-range_idx)[0]
        else:
            return m.weight.abs().max(2)[0].max(1-range_idx)[0]
    elif m.__class__.__name__ == "PACT_Linear":
        if not symmetric:
            return m.weight.max(1-range_idx)[0] - m.weight.min(1-range_idx)[0]
        else:
            return m.weight.abs().max(1-range_idx)[0]
    elif m.__class__.__name__ == "BatchNorm1d" or m.__class__.__name__ == "BatchNorm2d":
        return m.weight.data.abs()

def onnx_name_2_pytorch_name(name):
    name_parts = re.findall('\[.*?\]', name)
    name_parts = [part[1:-1] for part in name_parts]
    return '.'.join(name_parts)
