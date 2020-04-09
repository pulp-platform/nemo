#
# export.py
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

def _export_precision(self):
    r"""Returns a dictionary of precisions for each layer.

    """

    d = OrderedDict([])
    for n,m in self.named_modules():
        d[n] = OrderedDict([])
        try:
            d[n]['x_bits']  = m.precision.get_bits()
            if not "PACT" in m.__class__.__name__:
                d[n]['x_scale'] = m.precision.get_scale()
        except AttributeError:
            pass
        try:
            d[n]['W_bits']  = m.W_precision.get_bits()
            if not "PACT" in m.__class__.__name__:
                d[n]['W_scale'] = m.W_precision.get_scale()
        except AttributeError:
            pass
        if len(d[n].keys()) == 0 or n == "":
            d.pop(n, None)
    return d

def _export_weights_legacy_int16(self, header_name='weights.h', save_binary=False, folder_name='.', x_alpha_safety_factor=1.):
    r"""Exports weights and bias values with the legacy strategies used e.g. in PULP-DroNet,
    towards INT-16. Quantization is fully symmetric and aligned to power-of-two `alpha` so that
    there is no need to propagate :math:`\varepsilon` values.
    
    :param header_name: name of a header file.
    :type  header_name: string
    :param save_binary: if True, saves also a binary version.
    :type  save_binary: bool
    :param folder_name: name of the folder where to save binaries.
    :type  folder_name: string

    """

    weight_dict = {}
    bias_dict = {}
    qi_dict = {} # actually qi-1, as 1 bit is for sign
    x_alpha = 0.001
    W_alpha = {}
    bigstr = "/* weights & biases */\n\n#include <stdint.h>\n\n\n"
    checkstr_w = ""
    checkstr_b = ""
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Act"):
            x_alpha = max(x_alpha, m.alpha.item())
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Act"):
            m.alpha.data[:] = 2.**int(math.ceil(math.log2(x_alpha)))
    x_alpha *= x_alpha_safety_factor
    for n,m in self.named_modules():
        if (m.__class__.__name__ == "PACT_Conv2d" or \
            m.__class__.__name__ == "PACT_Conv1d" or \
            m.__class__.__name__ == "PACT_Linear"):
            W_alpha[n] = max(-m.W_alpha.item(), m.W_beta.item())
            qi_dict[n] = int(math.ceil(math.log2(W_alpha[n])))
            W_eps = 2.**-(16-qi_dict[n]-1)
            m.W_beta.data[:]  = 2.**qi_dict[n]
            m.W_alpha.data[:] = 2.**qi_dict[n]
            # performs also weight hardening, destructive!
            m.harden_weights()
            weight_dict[n] = np.int16(((m.weight.data.clone().detach().to('cpu').numpy()) / W_eps))
            m.weight.data[:] = torch.tensor(weight_dict[n] * W_eps)
            x_eps = 2.**-(16-int(math.ceil(math.log2(x_alpha)))-1)
            try:
                bias_dict[n]   = np.int16(m.bias.data.clone().detach().to('cpu').numpy() / x_eps)
                m.bias.data[:] = torch.tensor(bias_dict[n] * x_eps)
            except AttributeError:
                bias_dict[n]   = np.int16(np.zeros(weight_dict[n].shape[0]))
            import re
            n_str = re.sub('[^0-9a-zA-Z_]+', '_', n)
            bigstr += "// %s weights [shape=%s, qi=%d, qf=%d]\n" %  (n, weight_dict[n].shape, qi_dict[n]+1, 16-qi_dict[n]-1)
            bigstr += "int16_t w_%s[] = {\n  " % n_str
            for i in range(len(weight_dict[n].flatten())-1):
                bigstr += "0x%04x,\n  " % np.uint16(weight_dict[n].flatten()[i])
            bigstr += "0x%04x\n};\n\n" % np.uint16(weight_dict[n].flatten()[-1])
            bigstr += "// %s bias [shape=%s, qi=%d, qf=%d]\n" %  (n, bias_dict[n].shape, int(math.ceil(math.log2(x_alpha)))+1, 16-int(math.ceil(math.log2(x_alpha)))-1)
            bigstr += "int16_t b_%s[] = {\n  " % n_str
            for i in range(len(bias_dict[n].flatten())-1):
                bigstr += "0x%04x,\n  " % np.uint16(bias_dict[n].flatten()[i])
            bigstr += "0x%04x\n};\n\n\n" % np.uint16(bias_dict[n].flatten()[-1])
            if save_binary:
                with open("%s/weights_%s.hex" % (folder_name, n_str), "w") as file:
                    weight_dict[n].flatten().tofile(file)
                with open("%s/bias_%s.hex" % (folder_name, n_str), "w") as file:
                    bias_dict[n].flatten().tofile(file)
                checkstr_w += "Checksum weights_%s:\t%s\n" % (n_str, sum(weight_dict[n].flatten()))
                checkstr_b += "Checksum bias_%s:\t%s\n" % (n_str, sum(bias_dict[n].flatten()))
    print("Export procedure completed, qi=%d qf=%d for activations" % (int(math.ceil(math.log2(x_alpha)))+1, 16-int(math.ceil(math.log2(x_alpha)))-1))
    with open("%s/%s" % (folder_name, header_name), "w") as file:
        file.write(bigstr)
    with open("%s/checksum.txt" % (folder_name), "w") as file:
        file.write(checkstr_w)
        file.write(checkstr_b)