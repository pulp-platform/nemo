#
# statistics.py
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
from contextlib import contextmanager

@contextmanager
def _statistics_act_pact(self):
    r"""Used with `with net.statistics_act():`, calls `net.set_statistics_act()` on enter
    and `net.unset_statistics_act()` on exit.    

    """
    self.set_statistics_act()
    try:
        yield
    finally:
        self.unset_statistics_act()

@contextmanager
def _statistics_bn_pact(self):
    r"""Used with `with net.statistics_bn():`, calls `net.set_statistics_bn()` on enter
    and `net.unset_statistics_bn()` on exit.    

    """
    self.set_statistics_bn()
    try:
        yield
    finally:
        self.unset_statistics_bn()

def _set_statistics_act_pact(self):
    r"""Sets :py:class:`nemo.quant.PACT_Act` layers to collect statistics and work like ReLU's.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "PACT_Act":
            m.statistics_only = True

def _get_statistics_act_pact(self):
    r"""Returns the statistics collected by :py:class:`nemo.quant.PACT_Act` layers.
    
    """

    d = OrderedDict([])
    for n,m in self.named_modules():
        d[n] = OrderedDict([])
        if m.__class__.__name__ == "PACT_Act":
            d[n]['max']          = m.get_statistics()[0]
            d[n]['running_mean'] = m.get_statistics()[1]
            d[n]['running_var']  = m.get_statistics()[2]
            d[n]['active']       = m.statistics_only
    return d

def _unset_statistics_act_pact(self):
    r"""Sets :py:class:`nemo.quant.PACT_Act` layers to act normally and stop statistics collection.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "PACT_Act":
            m.statistics_only = False

def _set_statistics_bn_pact(self):
    r"""Sets :py:class:nemo.quant.PACT_QuantizedBatchNormNd` layers to collect statistics.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "PACT_QuantizedBatchNormNd":
            m.statistics_only = True

def _unset_statistics_bn_pact(self):
    r"""Sets :py:class:`nemo.quant.PACT_QuantizedBatchNormNd` layers to act normally and stop statistics collection.
    
    """

    for n,m in self.named_modules():
        if m.__class__.__name__ == "PACT_QuantizedBatchNormNd":
            m.statistics_only = False
