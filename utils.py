#
# utils.py
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
import logging
import os
import json
import re
import nemo
import numpy as np
from collections import OrderedDict

def precision_dict_to_json(d, filename=None):
    s = json.dumps(d, indent=4)
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(s)
    else:
        return s

def precision_dict_from_json(filename):
    with open(filename, "r") as f:
        rr = json.load(f)
    return rr

def process_json(json, args):
    args = vars(args)
    regime = {}
    if args.regime is not None:
        with open(args.regime, "r") as f:
            rr = json.load(f)
        for k in rr.keys():
            try:
                regime[int(k)] = rr[k]
            except ValueError:
                regime[k] = rr[k]

def save_checkpoint(net, optimizer, epoch, acc=0.0, checkpoint_name='net_', checkpoint_suffix=''):
    checkpoint_name = checkpoint_name + checkpoint_suffix
    logging.info('Saving checkpoint %s' % checkpoint_name)
    try:
        optimizer_state = optimizer.state_dict()
    except AttributeError:
        optimizer_state = None
    try:
        precision = net.export_precision()
    except AttributeError:
        precision = None
    state = {
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'precision': precision,
        'acc': acc,
        'optimizer' : optimizer_state,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/%s.pth' % (checkpoint_name))

def export_onnx(file_name, net, net_inner, input_shape, round_params=True, perm=None, redefine_names=False, batch_size=1, verbose=False):
    if perm is None:
        perm = lambda x : x
    pattern = re.compile('[\W_]+')
    dummy_input = perm(torch.randn(batch_size, *input_shape, device='cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()
    # rounding of parameters to avoid strange numerical errors on writeout
    if round_params:
        for param in net_inner.parameters(recurse=True):
            if param.dtype is torch.float32:
                param[:] = torch.round(param)
    if redefine_names:
        input_names  = [ 'input' ] + [ pattern.sub('_', n) for n,_ in net_inner.named_parameters() ]
        output_names = [ 'output' ]
        torch.onnx.export(net_inner, dummy_input, file_name, verbose=verbose, do_constant_folding=True, input_names=input_names, output_names=output_names, export_params=True)
    else:
        torch.onnx.export(net_inner, dummy_input, file_name, verbose=verbose, do_constant_folding=True, export_params=True)

PRECISION_RULE_KEYS_REQUIRED = {
    "for_epochs": 1,
    "for_epochs_no_abs_bound": 3,
    "delta_loss_less_than": 0.01,
    "running_avg_memory": 5,
    "delta_loss_running_std_stale": 1.5,
    "abs_loss_stale": 1.4,
    "scale_lr": True,
    "lr_scaler": 4.0,
    "divergence_abs_threshold": 1e9
}
PRECISION_RULE_KEYS_ALLOWED = [
    "custom_scaler",
    "bit_scaler",
    "bit_stop_condition"
]
 
def parse_precision_rule(rule):
    required = list(PRECISION_RULE_KEYS_REQUIRED.keys())
    allowed  = PRECISION_RULE_KEYS_ALLOWED + required
    for k in required:
        if not k in rule:
            rule[k] = PRECISION_RULE_KEYS_REQUIRED[k]
    flag = False
    for k in rule.keys():
        if not k in allowed and not k.isdigit():
            print("[ERROR] %s is not a key allowed in the relaxation rule", k)
            flag = True
    if "bit_scaler" in rule and not "W_bit_scaler" in rule:
        rule["W_bit_scaler"] = rule["bit_scaler"]
    if "bit_scaler" in rule and not "x_bit_scaler" in rule:
        rule["x_bit_scaler"] = rule["bit_scaler"]
    if "bit_stop_condition" in rule and not "W_bit_stop_condition" in rule:
        rule["W_bit_stop_condition"] = rule["bit_stop_condition"]
    if "bit_stop_condition" in rule and not "x_bit_stop_condition" in rule:
        rule["x_bit_stop_condition"] = rule["bit_stop_condition"]
    if flag:
        import sys; sys.exit(1)
    print(list(rule.keys()))
    return rule
    
# see https://github.com/sksq96/pytorch-summary
def get_summary(net, input_size, batch_size=1, device="cuda", verbose=False):
    s = ""
    mdict = {}
    for n,m in net.named_modules():
        mdict[n] = m
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = next(n for n,m in mdict.items() if m==module)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                try:
                    params += torch.prod(torch.LongTensor(list(module.weight.size()))) / module.group
                except AttributeError:
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            
            if hasattr(module, "W_precision"):
                summary[m_key]['W_bits'] = module.W_precision.get_bits()
            
            if hasattr(module, "precision"):
                summary[m_key]['bits'] = module.precision.get_bits()

        if (
            not isinstance(module, torch.nn.Sequential)
            and not isinstance(module, torch.nn.ModuleList)
            and not (module == net)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    net.apply(register_hook)

    # make a forward pass
    net(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    s += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    s += line_new + "\n"
    s += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    params_size = 0
    output_size = 0
    input_size = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        try:
            params_size += abs(summary[layer]["nb_params"] / (1024) * summary[layer]["W_bits"] / 8.)
        except KeyError:
            params_size += abs(summary[layer]["nb_params"] / (1024) * 32. / 8.)
        total_output += np.prod(summary[layer]["output_shape"])
        try:
            output_size = max(output_size, np.prod(summary[layer]["output_shape"]) / (1024) * summary[layer]["bits"] / 8)
        except KeyError:
            output_size = max(output_size, np.prod(summary[layer]["output_shape"]) / (1024) * 32 / 8)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        s += line_new + "\n"

    s += "================================================================" + "\n"
    s += "Total params: {0:,}".format(total_params) + "\n"
    s += "Trainable params: {0:,}".format(trainable_params) + "\n"
    s += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    s += "----------------------------------------------------------------" + "\n"
    s += "Biggest activation tensor size (kB): %0.2f" % output_size + "\n"
    s += "Params size (kB): %0.2f" % params_size + "\n"
    s += "----------------------------------------------------------------" + "\n"
    if verbose:
        logging.info(s)
    return summary, s

def get_intermediate_activations(net, test_fn, *test_args, **test_kwargs):
    l = len(list(net.named_modules()))
    buffer_in  = OrderedDict([])
    buffer_out = OrderedDict([])
    hooks = OrderedDict([])
    def get_hk(n):
        def hk(module, input, output):
            buffer_in  [n] = input
            buffer_out [n] = output
        return hk
    for i,(n,l) in enumerate(net.named_modules()):
        hk = get_hk(n)
        hooks[n] = l.register_forward_hook(hk)
    ret = test_fn(*test_args, **test_kwargs)
    for n,l in net.named_modules():
        hooks[n].remove()
    return buffer_in, buffer_out, ret
        
def get_intermediate_eps(net, eps_in):
    l = len(list(net.named_modules()))
    eps = OrderedDict([])
    for i,(n,l) in enumerate(net.named_modules()):
        eps[n] = net.get_eps_at(n, eps_in)
    return eps

def get_integer_activations(buf, eps, net=None):
    if type(eps) is float and net is None:
        return buf
    elif type(eps) is float:
        eps_in = eps
        eps = OrderedDict([])
        for n,m in net.named_modules():
            try:
                eps[n] = m.get_output_eps(eps_in)
            except AttributeError:
                pass
    buf_ = OrderedDict([])
    for n in buf.keys():
        b = buf.get(n, None)
        e = eps.get(n, None)
        if b is None or e is None:
            continue
        if type(buf[n]) is tuple or type(buf[n]) is list:
            buf_[n] = []
            for b in buf[n]:
                buf_[n].append((b / eps[n]).floor()) # FIXME
        else:
            buf_[n] = (buf[n] / eps[n]).floor()
    return buf_
    
