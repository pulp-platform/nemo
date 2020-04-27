#
# graph.py
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
import torch.onnx.utils
from nemo.precision import Precision
from nemo.quant.pact import *
from torch.nn.modules.utils import _single,_pair
from collections import OrderedDict
import types
import logging
import numpy as np
import copy
import math
import torchvision.models
import re
from packaging import version

# necessary for PyTorch >= 1.4! See https://github.com/pytorch/pytorch/issues/33463#issuecomment-606399944
class scope_name_workaround(object):
    def __init__(self):
        self.backup = None

    def __enter__(self):
        def _tracing_name(self_, tracing_state):
            if not tracing_state._traced_module_stack:
                return None
            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_children():
                if child is self_:
                    return name
            return None

        def _slow_forward(self_, *input, **kwargs):
            tracing_state = torch._C._get_tracing_state()
            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)
            if not hasattr(tracing_state, '_traced_module_stack'):
                tracing_state._traced_module_stack = []
            name = _tracing_name(self_, tracing_state)
            if name:
                tracing_state.push_scope('%s[%s]' % (self_._get_name(), name))
            else:
                tracing_state.push_scope(self_._get_name())
            tracing_state._traced_module_stack.append(self_)
            try:
                result = self_.forward(*input, **kwargs)
            finally:
                tracing_state.pop_scope()
                tracing_state._traced_module_stack.pop()
            return result
        
        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, '_slow_forward', _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, '_slow_forward', self.backup)

def onnx_name_2_pytorch_name(name):
    name_parts = re.findall('\[.*?\]', name)
    name_parts = [part[1:-1] for part in name_parts]
    return '.'.join(name_parts)

def _hier_flat_dict_build(module, name):
    for n,m in module.named_children():
        if n == name:
            return m
        elif n == name.split('.')[0]:
            return _hier_flat_dict_build(m, '.'.join(name.split('.')[1:]))
    return module

class DeployNode(object):
    def __init__(self, key="", incoming=None, outgoing=None):
        if incoming is None:
            self.incoming = []
        else:
            self.incoming = incoming
        if outgoing is None:
            self.outgoing = []
        else:
            self.outgoing = outgoing
        self.key = key
        self.input_node = False

    def is_input(self):
        return True if len(self.incoming)==0 else False

    def is_output(self):
        return True if len(self.outgoing)==0 else False

    def _traverse_forward(self, fn=None, recurse_max="inf", reduc_fn=lambda ret,x: ret+x, ret_default=0, **kwargs):
        ret = ret_default
        recurse_max = "inf" if recurse_max == "inf" else recurse_max-1
        for o in self.outgoing:
            ret = reduc_fn(ret, fn(o, **kwargs))
            if recurse_max == "inf" or recurse_max > 0:
                ret = reduc_fn(ret, o._traverse_forward(fn, recurse_max=recurse_max, reduc_fn=reduc_fn, **kwargs))
        return ret

    def _traverse_backward(self, fn=None, recurse_max="inf", reduc_fn=lambda ret,x: ret+x, ret_default=0, **kwargs):
        ret = ret_default
        recurse_max = "inf" if recurse_max == "inf" else recurse_max-1
        for o in self.incoming:
            ret = reduc_fn(ret, fn(o, **kwargs))
            if recurse_max == "inf" or recurse_max > 0:
                ret = reduc_fn(ret, o._traverse_backward(fn, recurse_max=recurse_max, reduc_fn=reduc_fn, ret_default=ret_default, **kwargs))
        return ret

class DeployGraph(object):
    def __init__(self, module, dummy_input):
        if version.parse(torch.__version__) < version.parse('1.4.0'):
            trace, _, _ = torch.jit.get_trace_graph(module, dummy_input, _force_outplace=True, _return_inputs_states=True)
            torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
            graph = trace.graph()
        else:
            with scope_name_workaround():
                graph, _params_dict, _torch_out = torch.onnx.utils._model_to_graph(module, dummy_input, propagate=True, _retain_param_name=True)
        input_dict = {}
        output_dict = {}
        self.non_unique_names_dict = {}
        self.module = module

        # build list of inputs/outputs
        for i,node in enumerate(graph.nodes()):
            op_name = node.scopeName()
            module_name = onnx_name_2_pytorch_name(op_name) + "/" + node.kind().lstrip('::onnx') + "_" + str(i)
            self.non_unique_names_dict[module_name] = onnx_name_2_pytorch_name(op_name)
            for i in node.inputs():
                try:
                    input_dict[i.debugName()].append(module_name)
                except KeyError:
                    input_dict[i.debugName()] = [module_name,]
            for o in node.outputs():
                output_dict[o.debugName()] = module_name

        # build a flat dictionary of modules -- this is centralized in the DeployGraph to ease nemo.transform
        all_nodes = list(set([i for i in output_dict.values()]) | set([i for l in input_dict.values() for i in l]))
        self.module_nodes = OrderedDict([])
        for n in all_nodes:
            nn = n.split("/")[0]
            self.module_nodes[n] = _hier_flat_dict_build(module, nn)
        
        # build a flat dictionary of DeployNodes
        self.nodes = OrderedDict([])
        for n in all_nodes:
            self.nodes[n] = DeployNode(key=n)
            
        # populate outgoing connections
        for ok in output_dict.keys():
            out = output_dict[ok]
            try:
                ils = input_dict[ok]
            except KeyError:
                # unused outputs
                ils = None
            if ils is not None:
                dnl = [self.nodes[i] for i in ils]
                self.nodes[out].outgoing.extend(dnl)

        # populate incoming connections
        nodes_copy = copy.copy(self.nodes) # shallow copy
        for k,n in nodes_copy.items():
            for i,m in enumerate(n.outgoing):
                self.nodes[k].outgoing[i].incoming.append(n)
                
        # identify input nodes (only 1!) FIXME
        self.input_nodes = []
        for k,n in self.nodes.items():
            if len(n.incoming) == 0:
                n.input_node = True
                self.input_nodes.append(n)

        self.jit_graph = graph

    def rebuild_module_dict(self):
        # build a flat dictionary of modules -- this is centralized in the DeployGraph to ease nemo.transform
        for n in self.module_nodes.keys():
            nn = n.split("/")[0]
            self.module_nodes[n] = _hier_flat_dict_build(self.module, nn)

    def print_modules(self):
        for k,n in self.nodes.items():
            print (k, self.module_nodes[k])

    def print_forward_edges(self):
        for k,n in self.nodes.items():
            print (k, [m.key for m in n.outgoing])

    def print_backward_edges(self):
        for k,n in self.nodes.items():
            print (k, [m.key for m in n.incoming])

    def print_jit_graph(self):
        print(self.jit_graph)

    def get_eps_at(self, key, eps_in, use_non_unique_name=True, verbose=False):
        # back-track route to input
        # the procedure is repeated for each incoming edge to the target node
        target = None
        if use_non_unique_name:
            for kk,el in list(self.non_unique_names_dict.items()):
                if el == key:
                    target = self.nodes[kk]
                    break
        else:
            target = self.nodes.get(key, None)
        if target is None:
            print("[nemo-graph] Warning: %s is not a module name" % key)
            # import IPython
            # IPython.embed()
            return None
        eps_list = []
        for incoming_idx in range(len(target.incoming)):
            curr = target
            route = []
            while not curr.is_input():
                if verbose:
                    print("[nemo-graph] backward %s" % (curr.key))
                k = curr.key
                # if current node is the target, use the incoming_idx route
                if curr == target:
                    curr = curr.incoming[incoming_idx]
                else:
                    curr = curr.incoming[0]
                route.append(curr.outgoing.index(self.nodes[k]))
            # forward-track route to node, computing eps
            route = route[::-1]
            eps = eps_in
            for idx in route:
                if verbose:
                    print("[nemo-graph] forward %s %d" % (curr.key, idx))
                if hasattr(self.module_nodes[curr.key], 'get_output_eps'):
                    eps = self.module_nodes[curr.key].get_output_eps(eps)
                try:
                    curr = curr.outgoing[idx]
                except IndexError:
                    print("[nemo-graph] Warning: %s has no outgoing edge" % (curr.key))
                    break
            if type(eps) is float:
                eps_list.append(torch.tensor(eps))
            eps_list.append(eps)
        if len(eps_list) == 1:
            return eps_list[0]
        else:
            return eps_list

    def get_supernodes(self, verbose=False):
        # collect all activation nodes
        actnodes = []
        for k,n in self.nodes.items():
            if isinstance(self.module_nodes[n.key], PACT_Act) or \
               isinstance(self.module_nodes[n.key], PACT_ThresholdAct) or \
               isinstance(self.module_nodes[n.key], PACT_IntegerAct):
                actnodes.append(n)
        supernodes = OrderedDict([])
        # for each activation node, backtrack until another activation node is found
        for target in actnodes:
            # here we assume all activation nodes have only one incoming path, which should be reasonable
            curr = target.incoming[0]
            route = []
            while not isinstance(self.module_nodes[curr.key], PACT_Act) or \
                      isinstance(self.module_nodes[curr.key], PACT_ThresholdAct) or \
                      isinstance(self.module_nodes[curr.key], PACT_IntegerAct):
                route.append((self.non_unique_names_dict[curr.key], self.module_nodes[curr.key]))
                if verbose:
                    print("[nemo-graph] backward %s" % (curr.key))
                try:
                    curr = curr.incoming[0]
                except IndexError:
                    break
            # forward-track route to node
            supernodes[self.non_unique_names_dict[target.key]] = {'supernode': route[::-1], 'previous': self.non_unique_names_dict[curr.key]}
        return supernodes
