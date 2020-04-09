#
# pact.py
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

# coding=utf-8
import math
import torch
from torch.nn.modules.utils import _single,_pair
from nemo.precision import Precision
import numpy as np

import logging

# Create custom symbolic function
from torch.onnx.symbolic_helper import parse_args

__all__ = ["PACT_Conv1d", "PACT_Conv2d", "PACT_Linear", "PACT_Act", "PACT_ThresholdAct", "PACT_IntegerAct", "PACT_IntegerAvgPool2d", "PACT_Identity", "PACT_QuantizedBatchNorm2d", "PACT_IntegerBatchNorm2d"]

# re-quantize from a lower precision (larger eps_in) to a higher precision (lower eps_out)
def pact_integer_requantize(t, eps_in, eps_out, D=1):
    eps_ratio = (D*eps_in/eps_out).round()
    device = t.device
    return torch.as_tensor((torch.as_tensor(t, dtype=torch.int64) * torch.as_tensor(eps_ratio, dtype=torch.int64) / torch.as_tensor(D, dtype=torch.int64)), dtype=torch.float32, device=device)

# re-quantize from a lower precision (larger eps_in) to a higher precision (lower eps_out)
def pact_integer_requantize_add(*t, eps_in_list, eps_out, D=1):
    # unrolling the first iteration of the loop instead of using torch.zeros to init y is necessary for correct ONNX export
    eps_in = eps_in_list[0]
    eps_ratio = (D*eps_in/eps_out).round()
    y = t[0] * eps_ratio
    for i in range(1,len(eps_in_list)):
        eps_in = eps_in_list[i]
        eps_ratio = (D*eps_in/eps_out).round()
        y += t[i] * eps_ratio
    y = (y / D).floor()
    return y

# PACT quantization for inference
def pact_quantize_inference(W, eps, clip):
    W_quant = W.clone().detach()
    W_quant.data[:] = (W_quant.data[:] / eps).floor()*eps
    W_quant.clamp_(0, clip.item())
    return W_quant

def pact_quantize_deploy(W, eps, clip):
    W = (W / eps).floor()*eps
    W = W.clamp(0, clip)
    return W

# PACT signed quantization for inference (workaround for pact_quantize_signed not functional in inference)
def pact_quantize_signed_inference(W, eps, clip):
    W_quant = W.clone().detach()
    W_quant.data[:] = (W_quant.data[:] / eps).floor()*eps
    W_quant.clamp_(-clip.item(), clip.item())
    return W_quant

# PACT asymmetric quantization for inference (workaround for pact_quantize_asymm not functional in inference)
def pact_quantize_asymm_inference(W, eps, alpha, beta, train_loop=True, train_loop_oldprec=None):
    # for numerical reasons, W_quant should be put at a "small_eps" of its "pure" value to enable
    # running againt the weights through pact_quantize_asymm_inference (the torch.floor function
    # won't return the correct value otherwise)
    # we choose small_eps = eps/2
    if not train_loop and train_loop_oldprec is not None:
        W_quant = W.clone().detach()
        W_quant.data[:] = (W_quant.data[:] / train_loop_oldprec).floor()*train_loop_oldprec + eps/2
    elif not train_loop:
        W_quant = W.clone().detach() + eps/2
    else:
        W_quant = W.clone().detach()
    W_quant.data[:] = (W_quant.data[:] / eps).floor()*eps
    W_quant.clamp_(-alpha.item(), beta.item() + eps.item())
    return W_quant

def pact_pwl(x, eps, alpha, beta, q0=0):
    beta = beta.abs()
    beta_cumsum = beta.cumsum(0)
    m = eps / beta # at initial PWL step, should be 1!
    # compute step, extended with -alpha at the beginning
    step = torch.cat((torch.as_tensor((-np.infty,), device=alpha.device), -alpha, -alpha + beta_cumsum[:-1], torch.as_tensor((+np.infty,), device=alpha.device)))
    sr = torch.arange(len(step.shape)+len(x.shape))
    sr_m1  = sr[-1].clone().detach()
    sr[1:] = sr[:-1].clone().detach()
    sr[0]  = sr_m1
    step = step.repeat((*x.shape,1)).permute(tuple(sr.numpy()))
    # compute q offset
    q = torch.zeros_like(beta)
    q[1:] = q0 + eps * (torch.arange(1, beta.shape[0], dtype=torch.float32, device=alpha.device) - 1 - (-alpha + beta_cumsum[:-1])/beta[1:])
    # extend m with a final 0
    m = torch.cat((torch.ones(1, device=alpha.device), m))
    q = torch.cat((torch.zeros(1, device=alpha.device), q))
    y = torch.zeros_like(x)
    # compare
    inside = ((x >= step[:-1]) * (x < step[1:]))
    for i in range(len(inside.shape)-1):
        m = torch.unsqueeze(m, -1)
        q = torch.unsqueeze(q, -1)
    x = torch.unsqueeze(x, 0)
    y = torch.where(inside, m*x+q, y).sum(0)
    del inside, m, q, beta_cumsum
    return y.clamp(-alpha.item(), alpha.item()-eps.item())

# PACT activation: https://arxiv.org/pdf/1805.06085.pdf
class PACT_QuantFunc(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function for activations.

        Implements a :py:class:`torch.autograd.Function` for quantizing activations in :math:`Q` bits using the PACT strategy.
        In forward propagation, the function is defined as 
        
        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [0,\alpha) } (\mathbf{x})\right\rfloor \cdot \varepsilon
        
        where :math:`\varepsilon` is the quantization precision:
        
        .. math::
            \varepsilon = \alpha / (2^Q - 1)
        
        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as
        
        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}
        
        It can be applied by using its static `.apply` method:
    
    :param input: the tensor containing :math:`x`, the activations to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param alpha: the value of :math:`\alpha`.
    :type  alpha: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default 1e-9).
    :type  delta: `torch.Tensor` or float
    
    :return: The quantized input activations tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, alpha, delta=1e-9):
        where_input_nonclipped = (input >= 0) * (input < alpha)
        where_input_gtalpha = (input >= alpha)
        ctx.save_for_backward(where_input_nonclipped, where_input_gtalpha)
        return (input.clamp(0., alpha.data[0]) / (eps+delta)).floor() * eps

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_gtalpha = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        grad_alpha = torch.where(where_input_gtalpha, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_alpha

pact_quantize = PACT_QuantFunc.apply

class PACT_QuantFunc_Signed(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function for weights (simmetric).

        Implements a :py:class:`torch.autograd.Function` for quantizing weights in :math:`Q` bits using a symmetric PACT-like strategy (original
        PACT is applied only to activations, using DoReFa-style weights).
        In forward propagation, the function is defined as 
        
        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [-\alpha,+\alpha) } (\mathbf{x})\right\rfloor \cdot \varepsilon
        
        where :math:`\varepsilon` is the quantization precision:
        
        .. math::
            \varepsilon = 2\cdot\alpha / (2^Q - 1)
        
        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as
        
        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}
        
        It can be applied by using its static `.apply` method:
    
    :param input: the tensor containing :math:`x`, the weights to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param alpha: the value of :math:`\alpha`.
    :type  alpha: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default 1e-9).
    :type  delta: `torch.Tensor` or float
    
    :return: The quantized weights tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, alpha, delta=1e-9):
        where_input_nonclipped = (input >= -alpha) * (input < alpha)
        where_input_gtalpha = (input >= alpha) + (input < -alpha)
        ctx.save_for_backward(where_input_nonclipped, where_input_gtalpha)
        return (input.clamp(-alpha.data[0], alpha.data[0]) / (eps+delta)).floor() * eps

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_gtalpha = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        grad_alpha = torch.where(where_input_gtalpha, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_alpha

pact_quantize_signed = PACT_QuantFunc_Signed.apply

class PACT_QuantFunc_Asymm(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function (asymmetric).

        Implements a :py:class:`torch.autograd.Function` for quantizing weights in :math:`Q` bits using an asymmetric PACT-like strategy (original
        PACT is applied only to activations, using DoReFa-style weights).
        In forward propagation, the function is defined as 
        
        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [-\alpha,+\beta) } (\mathbf{x})\right\rfloor \cdot \varepsilon
        
        where :math:`\varepsilon` is the quantization precision:
        
        .. math::
            \varepsilon = (\alpha+\beta) / (2^Q - 1)
        
        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as
        
        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}
        
        It can be applied by using its static `.apply` method:
    
    :param input: the tensor containing :math:`x`, the weights to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param alpha: the value of :math:`\alpha`.
    :type  alpha: `torch.Tensor` or float
    :param beta: the value of :math:`\beta`.
    :type  beta: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default 1e-9).
    :type  delta: `torch.Tensor` or float
    
    :return: The quantized weights tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, alpha, beta, delta=1e-9):
        where_input_nonclipped = (input >= -alpha) * (input < beta)
        where_input_ltalpha = (input < -alpha)
        where_input_gtbeta = (input >= beta)
        ctx.save_for_backward(where_input_nonclipped, where_input_ltalpha, where_input_gtbeta)
        return (input.clamp(-alpha.data[0], beta.data[0]) / (eps+delta)).floor() * eps

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_ltalpha, where_input_gtbeta = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        grad_alpha = torch.where(where_input_ltalpha, grad_output, zero).sum().expand(1)
        grad_beta = torch.where(where_input_gtbeta, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_alpha, grad_beta

pact_quantize_asymm = PACT_QuantFunc_Asymm.apply

class PACT_Act(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`torch.nn.ReLU`, :py:class:`torch.nn.ReLU6` and
    similar activations in a PACT-quantized network.

    This layer can also operate in a special mode, defined by the `statistics_only` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9

    """

    def __init__(self, precision=None, alpha=1., backprop_alpha=True, statistics_only=False, leaky=None):
        r"""Constructor. Initializes a :py:class:`torch.nn.Parameter` for :math:`\alpha` and sets
            up the initial value of the `statistics_only` member.

        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha`.
        :type  alpha: `torch.Tensor` or float
        :param backprop_alpha: default `True`; if `False`, do not update the value of `\alpha` with backpropagation.
        :type  backprop_alpha: bool
        :param statistics_only: initialization value of `statistics_only` member.
        :type  statistics_only: bool

        """

        super(PACT_Act, self).__init__()
        if precision is None:
            self.precision = Precision()
        else:
            self.precision = precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = torch.nn.Parameter(torch.Tensor((alpha,)).to(device), requires_grad=backprop_alpha)
        self.alpha_p = alpha
        self.statistics_only = statistics_only
        self.deployment = False
        self.eps_in = None
        self.leaky = leaky

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.alpha.data).to(device),  requires_grad=False)

    def set_static_precision(self):
        r"""Sets static parameters used only for deployment.

        """
        self.eps_static   = self.alpha.item()/(2.0**(self.precision.get_bits())-1)
        self.alpha_static = self.alpha.item()

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """

        return self.alpha/(2.0**(self.precision.get_bits())-1)

    def reset_alpha(self, use_max=True, nb_std=5.):
        r"""Reset the value of :math:`\alpha`. If `use_max` is `True`, then the highest tensor-wise value collected
            in the statistics collection phase is used. If `False`, the collected standard deviation multiplied by
            `nb_std` is used as a parameter

        :param use_max: if True, use the tensor-wise maximum value collected in the statistics run as new :math:`\alpha` (default True).
        :type  use_max: bool
        :param nb_std: number of standard deviations to be used to initialize :math:`\alpha` if `use_max` is False.
        :type  nb_std: float

        """

        if use_max:
            self.alpha.data[0] = self.max.item()
        else:
            self.alpha.data[0] = nb_std * torch.sqrt(self.var).item()

    def get_statistics(self):
        r"""Returns the statistics collected up to now.
    
        :return: The collected statistics (maximum, running average, running variance).
        :rtype:  tuple of floats

        """
        return self.max.item(), self.running_mean.item(), self.running_var.item()
    
    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.
        
        See :py:class:`nemo.quant.pact_quant.PACT_QuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        if self.deployment:
            return pact_quantize_deploy(x, self.eps_static, self.alpha_static)
        elif self.statistics_only:
            if self.leaky is None:
                x = torch.nn.functional.relu(x)
            else:
                x = torch.nn.functional.leaky_relu(x, self.leaky)
            with torch.no_grad():
                self.max[:] = max(self.max.item(), x.max())
                self.min[:] = min(self.min.item(), x.min())
                self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x.mean()
                self.running_var[:]  = 0.9 * self.running_var.item()  + 0.1 * x.std()*x.std()
            return x
        else:
            return pact_quantize(x, self.alpha/(2.0**(self.precision.get_bits())-1), self.alpha)

class PACT_IntegerAdd(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation for integer images.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`nemo.quant.pact_quant.PACT_Act`
    in a integer network. It is not meant to be trained but rather generated by a NeMO transformation. 

    """

    def __init__(self, alpha=1., precision=None, requantization_factor=256):
        r"""Constructor. Initializes a :py:class:`torch.nn.Parameter` for :math:`\alpha`.

        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha`.
        :type  alpha: `torch.Tensor` or float
        :param requantization_factor: minimum target ratio in requantization fraction (default 32).
        :type  requantization_factor: int

        """
        super(PACT_IntegerAdd, self).__init__()
        if precision is None:
            self.precision = Precision(bits=8)
        else:
            self.precision = precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.requantization_factor = requantization_factor
        self.deployment = False
        self.integerized = False
        self.eps_in_list = []
    
    def get_output_eps(self, eps_in_list):
        if type(eps_in_list) is list:
            self.eps_in_list = eps_in_list
        # for now, select the biggest eps_out as the target one
        self.eps_out = max(self.eps_in_list)
        self.alpha_out = 2.0**(self.precision.get_bits())-1
        # D is selected as a power-of-two
        self.D = 2.0**torch.ceil(torch.log2(self.requantization_factor * self.eps_out / min(self.eps_in_list)))
        return self.eps_out

    def forward(self, *x):
        if not self.deployment or not self.integerized:
            y = x[0]
            for i in range(1,len(x)):
                y += x[i]
            return y
        else:
            return pact_integer_requantize_add(*x, eps_in_list=self.eps_in_list, eps_out=self.eps_out, D=self.D)

class PACT_IntegerAvgPool2d(torch.nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
            count_include_pad=True, divisor_override=None, requantization_factor=16):
        super(PACT_IntegerAvgPool2d, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
            count_include_pad=count_include_pad, divisor_override=divisor_override)
        self.requantization_factor = requantization_factor
        if type(self.kernel_size) is not tuple and type(self.kernel_size) is not list:
            ks = (self.kernel_size, self.kernel_size)
        else:
            ks = self.kernel_size
        self.kernel_size = ks
        self.D = 2.0**torch.ceil(torch.log2(self.requantization_factor*torch.as_tensor(1.0*self.kernel_size[0]*self.kernel_size[1])))

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """
        return eps_in

    def forward(self, input):
        y = torch.nn.functional.avg_pool2d(input, self.kernel_size, self.stride,
                                           self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        if self.divisor_override is not None:
            y *= self.divisor_override
        else:
            y *= self.kernel_size[0] * self.kernel_size[1]
        ratio = (self.D/(self.kernel_size[0]*self.kernel_size[1])).round()
        return (y * ratio / self.D).floor()

class PACT_IntegerAct(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation for integer images.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`nemo.quant.pact_quant.PACT_Act`
    in a integer network. It is not meant to be trained but rather generated by a NeMO transformation. 

    """

    def __init__(self, eps_in, alpha=1., precision=None, requantization_factor=16):
        r"""Constructor. Initializes a :py:class:`torch.nn.Parameter` for :math:`\alpha`.

        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha`.
        :type  alpha: `torch.Tensor` or float
        :param requantization_factor: minimum target ratio in requantization fraction (default 32).
        :type  requantization_factor: int

        """
        super(PACT_IntegerAct, self).__init__()
        if precision is None:
            self.precision = Precision(bits=16)
        else:
            self.precision = precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha = torch.nn.Parameter(alpha)

        self.eps_in = eps_in
        self.requantization_factor = requantization_factor

    def set_output_eps(self):
        r"""Sets static parameters used only for deployment.

        """
        self.eps_out   = self.alpha.item()/(2.0**(self.precision.get_bits())-1)
        self.alpha_out = 2.0**(self.precision.get_bits())-1
        # D is selected as a power-of-two
        self.D = 2.0**torch.ceil(torch.log2(self.requantization_factor * self.eps_out / self.eps_in))

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """

        return self.eps_out

    def forward(self, x):
        r"""Forward-prop function for integer PACT activations.
        
        In forward propagation, the function is defined as 
        
        .. math::
            \mathbf{y} = \mathrm{clip}_{[0,\alpha/\varepsilon_\mathbf{y}]}\left( \left\lfloor\frac{\varepsilon_\mathbf{x}\cdot 2^d}{\varepsilon_\mathbf{y}}\right\rfloor \cdot {q}_\mathbf{x}(\mathbf{x}) \gg d \right)
        
        It performs a requantization from :math:`\varepsilon_\mathbf{y}` to :math:`\varepsilon_\mathbf{y}`.
        
        This layer is not meant to be trained directly, and as such using it in backward is not recommended.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        x_rq = pact_integer_requantize(x, self.eps_in, self.eps_out, self.D)
        return x_rq.clamp(0, self.alpha_out)

class PACT_ThresholdAct(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation with thresholding.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`nemo.quant.pact_quant.PACT_Act`
    if BatchNorms are merged with PACT_Act activations. It works both in fake-quantized and integer mode.
    It is not meant to be trained but rather generated by a NeMO transformation.
    Thresholds are not directly memorized; instead, they are split in two parameters :math:`\kappa` and :math:`\lambda`. If :math:`p` is an iterator
    on the number of thresholds,

    .. math::
        \tau_p &= p \cdot \sigma/\gamma\varepsilon - \sigma/\gamma*\beta + \mu \\
        \tau_p &= p \cdot \kappa\varepsilon + \lambda \\
        \kappa &= \sigma/\gamma, \lambda = \mu - \sigma/\gamma*\beta
    
    After integerization:

    .. math::
        \tau_p = 1/\varepsilon_x/\varepsilon_W \cdot (p \cdot \kappa\varepsilon + \lambda)

    """

    def __init__(self, precision=None, alpha=1., nb_channels=1):
        r"""Constructor. Initializes a :py:class:`torch.nn.Parameter` for :math:`\alpha` and the target
        number of channels.

        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha`.
        :type  alpha: `torch.Tensor` or float
        :param requantization_factor: minimum target ratio in requantization fraction (default 100).
        :type  requantization_factor: int

        """
        super(PACT_ThresholdAct, self).__init__()
        if precision is None:
            self.precision = Precision()
        else:
            self.precision = precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha = torch.nn.Parameter(alpha)

        # thresholds are memorized in the format:
        #   tau_hat(p) = sigma/gamma*eps * p - sigma/gamma*beta + mu
        #   tau_hat(p) = kappa*eps * p + lamda
        #   kappa = sigma/gamma, lamda = mu - sigma/gamma*beta
        # with p in [0,1,...,2**precision.bits()-1]
        # INT-Q thresholds are:
        #   tau(p) = 1/eps_x/eps_W * (kappa * eps * p + lambda)
        self.kappa = torch.nn.Parameter(torch.zeros(nb_channels).to(device))
        self.lamda = torch.nn.Parameter(torch.zeros(nb_channels).to(device))

    def forward(self, x):
        r"""Forward-prop function for threshold-based PACT activations.
        
        This layer is not meant to be trained directly, and as such using it in backward is not recommended.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """
        kappa = self.kappa.reshape((1,self.kappa.shape[0],1,1))
        lamda = self.lamda.reshape((1,self.lamda.shape[0],1,1))
        eps = self.alpha/(2**self.precision.get_bits()-1)
        return pact_quantize_inference((x.data[:] - lamda) / kappa, eps, self.alpha)

class PACT_QuantizedBatchNorm2d(torch.nn.Module):
    r"""PACT-quantized 2d BatchNorm.

    Implements a :py:class:`torch.nn.Module` to implement a :py:class:`torch.nn.BatchNorm2d` with quantized parameters.

    """

    def __init__(self, precision=None, kappa=None, lamda=None, nb_channels=1, statistics_only=False):
        r"""Constructor.

        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: :py:class:`nemo.precision.Precision`
        :param kappa: the value of :math:`\kappa`.
        :type  kappa: `torch.Tensor` or float
        :param lamda: the value of :math:`\lambda`.
        :type  lamda: `torch.Tensor` or float
        :param nb_channels: number of channels to batch-normalize.
        :type  nb_channels: int
        :param statistics_only: initialization value of `statistics_only` member.
        :type  statistics_only: bool

        """

        super(PACT_QuantizedBatchNorm2d, self).__init__()
        if precision is None:
            self.precision_kappa = Precision(bits=16)
            self.precision_lamda = Precision(bits=16)
        else:
            self.precision_kappa = precision
            self.precision_lamda = precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if kappa is None:
            self.kappa = torch.nn.Parameter(torch.zeros(1,nb_channels,1,1).to(device), requires_grad=False)
        else:
            self.kappa = torch.nn.Parameter(kappa.to(device), requires_grad=False).reshape((1,kappa.shape[0],1,1))
        if lamda is None:
            self.lamda = torch.nn.Parameter(torch.zeros(1,nb_channels,1,1).to(device), requires_grad=False)
        else:
            self.lamda = torch.nn.Parameter(lamda.to(device), requires_grad=False).reshape((1,lamda.shape[0],1,1))

        self.statistics_only = statistics_only

        self.min = torch.nn.Parameter(torch.zeros(1).to(device), requires_grad=False)
        self.max = torch.nn.Parameter(torch.zeros(1).to(device), requires_grad=False)

        self.deployment = False
        self.eps_kappa = None
        self.eps_lamda = None
        self.eps_in = None
        self.eps_lamda_min = 1e-6
        
    def harden_weights(self):
        r"""Replaces the current value of weight tensors (full-precision, quantized on forward-prop) with the quantized value.

        """

        kappa_int = self.kappa.abs().max()
        if self.eps_kappa is None:
            eps_kappa = 2*kappa_int/(2**self.precision_kappa.get_bits()-1)
        else:
            eps_kappa = self.eps_kappa
        lamda_int = self.lamda.abs().max()
        if self.eps_lamda is None:
            eps_lamda = 2*lamda_int/(2**self.precision_lamda.get_bits()-1)
        else:
            eps_lamda = self.eps_lamda
        # this is because lamda might be 0!
        if eps_lamda < self.eps_lamda_min:
            eps_lamda = self.eps_lamda_min

        self.kappa.data[:] = pact_quantize_signed_inference(self.kappa.data[:], eps_kappa, kappa_int)
        self.lamda.data[:] = pact_quantize_signed_inference(self.lamda.data[:], eps_lamda, lamda_int)

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """

        if self.eps_kappa is None or self.eps_lamda is None:
            kappa_int = self.kappa.abs().max()
            eps_kappa = 2*kappa_int/(2**self.precision_kappa.get_bits()-1)
        else:
            eps_kappa = self.eps_kappa
        # eps_lambda must be == eps_kappa * eps_in
        self.eps_kappa = eps_kappa
        self.eps_lamda = (eps_kappa * eps_in).clone().detach()
        # check whether lamda can be represented in this many bits, otherwise, requantize it in a lossy fashion
        n = torch.log2(self.lamda.abs().max()/self.eps_lamda)
        if n > self.precision_lamda.get_bits():
            lamda_int = self.lamda.abs().max()
            self.eps_lamda = 2*lamda_int/(2**self.precision_lamda.get_bits()-1)
        return eps_kappa * eps_in

    def forward(self, x):
        r"""Forward-prop function for PACT-quantized batch-norm.
        
        In forward propagation, the function is defined as 
        
        .. math::
            \mathbf{y} = \kappa\cdot\varphi + \lambda

        This layer is not meant to be trained directly, and as such using it in backward is not recommended.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        kappa = self.kappa
        lamda = self.lamda
        if self.deployment:
            if self.eps_kappa is None:
                with torch.no_grad():
                    kappa_int = self.kappa.abs().max()
                    eps_kappa = 2*kappa_int/(2**self.precision_kappa.get_bits()-1)
            else:
                eps_kappa = self.eps_kappa
            if self.eps_lamda is None:
                with torch.no_grad():
                    lamda_int = self.lamda.abs().max()
                    eps_lamda = 2*lamda_int/(2**self.precision_lamda.get_bits()-1)
            else:
                eps_lamda = self.eps_lamda
            y = kappa * x + lamda
            return y
        if not self.statistics_only:
            kappa_int = self.kappa.abs().max()
            lamda_int = self.lamda.abs().max()
            eps_kappa = 2*kappa_int/(2**self.precision_kappa.get_bits()-1)
            eps_lamda = 2*lamda_int/(2**self.precision_lamda.get_bits()-1)
            # this is because lamda might be 0!
            if eps_lamda < self.eps_lamda_min:
                eps_lamda = self.eps_lamda_min
            kappa = pact_quantize_signed_inference(kappa, eps_kappa, kappa_int)
            lamda = pact_quantize_signed_inference(lamda, eps_lamda, lamda_int)
        out = kappa*x + lamda
        if self.statistics_only:
            with torch.no_grad():
                self.min[:] = min(self.min.item(), out.min())
                self.max[:] = min(self.max.item(), out.max())
        return out

class PACT_IntegerBatchNorm2d(torch.nn.Module):
    r"""Integer 2d BatchNorm.

    Implements a :py:class:`torch.nn.Module` to implement a :py:class:`torch.nn.BatchNorm2d` with integer parameters.

    """

    def __init__(self, kappa, lamda, eps_in, eps_kappa, eps_lamda):
        r"""Constructor.

        :param kappa: the value of :math:`\kappa`.
        :type  kappa: `torch.Tensor` or float
        :param lamda: the value of :math:`\lambda`.
        :type  lamda: `torch.Tensor` or float
        :param eps_in: the value of :math:`\varepsilon_{in}`.
        :type  eps_in: `torch.Tensor` or float
        :param eps_kappa: the value of :math:`\varepsilon_\kappa`.
        :type  eps_kappa: `torch.Tensor` or float
        :param eps_lamda: the value of :math:`\varepsilon_\lambda`.
        :type  eps_lamda: `torch.Tensor` or float

        """

        super(PACT_IntegerBatchNorm2d, self).__init__()
        self.kappa = kappa
        self.lamda = lamda
        self.eps_kappa = eps_kappa
        self.eps_lamda = eps_lamda
        self.eps_in = eps_in
        
    def integerize_weights(self):
        r"""Replaces the current value of weight tensors with the integer weights (i.e., the weight's quantized image).

        """

        kappa_int = self.kappa.abs().max()
        lamda_int = self.lamda.abs().max()
        self.kappa.data[:] = torch.round(pact_quantize_signed_inference(self.kappa.data[:], self.eps_kappa, kappa_int) / self.eps_kappa)
        self.lamda.data[:] = torch.round(pact_quantize_signed_inference(self.lamda.data[:], self.eps_lamda, lamda_int) / self.eps_lamda)

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """

        return self.eps_kappa * eps_in

    def forward(self, x):
        r"""Forward-prop function for integer batch-norm.
        
        In forward propagation, the function is defined as 
        
        .. math::
            q_\mathbf{y}(\mathbf{y}) = {q}_\kappa(\kappa)\cdot{q}_\varphi(\varphi) + {q}_{\kappa\varphi}(\lambda)
        
        where :math:`q_{\kappa\varphi}(\lambda)` is obtained by requantization from :math:`\varepsilon_\lambda` to
        :math:`\varepsilon_\kappa\cdot\varepsilon_\varphi`. 

        This layer is not meant to be trained directly, and as such using it in backward is not recommended.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        # requantize lamda to eps_kappa*eps_in (which is the output precision)
        lamda_rq = pact_integer_requantize(self.lamda, self.eps_lamda, self.eps_kappa*self.eps_in)
        return self.kappa*x + lamda_rq

class PACT_Identity(torch.nn.Module):
    r"""Identity module.

    Implements a :py:class:`torch.nn.Module` returning its own input unchanged.

    """

    def __init__(self):
        r"""Constructor.

        """

        super(PACT_Identity, self).__init__()
        
    def forward(self, x):
        r"""Forward-prop function for identity.
        
        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """
        return x

class PACT_Conv2d(torch.nn.Conv2d):
    r"""PACT (PArametrized Clipping acTivation) 2d convolution, extending :py:class:`torch.nn.Conv2d`.

    Implements a :py:class:`torch.nn.Module` to implement PACT-like convolution. It uses PACT-like quantized weights and can be configured to quantize also input activations.
    It assumes full-precision storage of partial results.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        quantize_x=False,
        quantize_W=True,
        quantize_y=False,
        W_precision=None,
        x_precision=None,
        alpha=1.,
        quant_asymm=True,
        **kwargs
    ):
        r"""Constructor. Supports all arguments supported by :py:class:`torch.nn.Conv2d` plus additional ones.

        :param quantize_x: if True, quantize input activations (default False).
        :type  quantize_x: bool
        :param quantize_W: if True, quantize weights (default True).
        :type  quantize_W: bool
        :param x_precision: precision to be used for quantization of input activations (default None).
        :type  x_precision: :py:class:`nemo.precision.Precision`
        :param W_precision: precision to be used for quantization of weights (default None).
        :type  W_precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha` (default 1.0).
        :type  alpha: `torch.Tensor` or float
        :param quant_asymm: use asymmetric quantization for weights (default `True`).
        :type  quant_asymm: bool
        """
        if W_precision is None:
            self.W_precision = Precision()
        else:
            self.W_precision = W_precision
        if x_precision is None:
            self.x_precision = Precision()
        else:
            self.x_precision = x_precision

        self.quantize_x = quantize_x
        self.quantize_W = quantize_W

        super(PACT_Conv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W_alpha = torch.nn.Parameter(torch.Tensor((alpha,)).to(device))
        if quant_asymm:
            self.W_beta  = torch.nn.Parameter(torch.Tensor((alpha,)).to(device))

        self.x_alpha = torch.nn.Parameter(torch.Tensor((2.0,)).to(device))
        self.weight.data.uniform_(-1., 1.)

        self.quant_asymm = quant_asymm

        self.train_loop = True
        self.deployment = False
        self.train_loop_oldprec = None

        self.padding_value = 0

    def reset_alpha_weights(self, use_max=True, nb_std=5., verbose=False, **kwargs):
        r"""Resets :math:`\alpha` and :math:`\beta` parameters for weights.

        """

        if not self.quant_asymm:
            self.W_alpha.data[0] = self.weight.data.abs().max()
        elif use_max:
            self.W_alpha.data[0] = -self.weight.data.min()
            self.W_beta.data [0] =  self.weight.data.max()
        else:
            self.W_alpha.data[0] = -self.weight.data.mean() + nb_std*self.weight.data.std()
            self.W_beta.data[0]  =  self.weight.data.mean() + nb_std*self.weight.data.std()
        if verbose:
            logging.info("[Quant] W_alpha = %.5f" % self.W_alpha.data[0])
            logging.info("[Quant] W_beta  = %.5f" % self.W_beta.data[0])

    def harden_weights(self):
        r"""Replaces the current value of weight tensors (full-precision, quantized on forward-prop) with the quantized value.

        """

        if self.quant_asymm:
            self.weight.data = pact_quantize_asymm_inference(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta, train_loop=False, train_loop_oldprec=self.train_loop_oldprec)
        else: 
            self.weight.data = pact_quantize_signed_inference(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)

    def integerize_weights(self):
        r"""Replaces the current value of weight tensors with the integer weights (i.e., the weight's quantized image).

        """

        if self.quant_asymm:
            eps = (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1)
            self.weight.data = pact_quantize_asymm_inference(self.weight, eps, self.W_alpha, self.W_beta, train_loop=False) / eps
        else:
            eps = 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1)
            self.weight.data = pact_quantize_signed_inference(self.weight, eps, self.W_alpha) / eps

    def prune_weights(self, threshold=0.1, eps=2**-9.):
        r"""Prunes the weights of the layer.

        The pruning is performed channel-wise by replacing all weights that are "near" to the channel-wise mean with the mean itself.
        "Near" weights are those that differ with the mean by less than `threshold` times the standard deviation.

        :param threshold: threshold used for pruning (default 0.1).
        :type  threshold: float
        :param eps: parameter used to compare the difference with 0 (default 2^-9).
        :type  eps: float

        """

        logging.info("[Pruning] tau=%.1e", threshold)
        stdev_per_chan = self.weight.data.std ((2,3), keepdim=True)
        mean_per_chan  = self.weight.data.mean((2,3), keepdim=True) # + eps
        self.weight.data = torch.where((self.weight.data - mean_per_chan).abs() < stdev_per_chan*threshold, mean_per_chan, self.weight.data)
        wc = self.weight.data.clone().detach().to('cpu').numpy().flatten()
        logging.info("[Pruning] Pruned %d" % np.count_nonzero(wc < eps))
        return np.count_nonzero(wc < eps)

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """

        if self.quant_asymm:
            eps_W = (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1)
        else:
            eps_W = 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1)
        return eps_W * eps_in
    def forward(self, input):
        r"""Forward-prop function for PACT-quantized 2d-convolution.

        :param input: input activations tensor.
        :type  input: `torch.Tensor`

        """

        if self.training and self.quantize_x and not self.deployment:
            x_quant = pact_quantize_signed(input, self.x_alpha/(2.0**(self.x_precision.get_bits())-1), self.x_alpha)
        else:
            x_quant = input
        if self.training and self.quantize_W and not self.deployment:
            if self.quant_asymm:
                W_quant = pact_quantize_asymm(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta)
            else:
                W_quant = pact_quantize_signed(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)
        elif self.quantize_W and not self.deployment:
            if self.quant_asymm:
                W_quant = pact_quantize_asymm_inference(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta, train_loop=self.train_loop, train_loop_oldprec=self.train_loop_oldprec)
            else:
                W_quant = pact_quantize_signed_inference(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)
        else:
            W_quant = self.weight
        # if input bias is present, padding should be performed using the input bias as padding value instead of 0
        if self.deployment and self.padding is not None and self.bias is not None:
            if type(self.padding) is not tuple and type(self.padding) is not list:
                pad = (self.padding, self.padding, self.padding, self.padding)
            elif len(self.padding) == 2:
                pad = (*self.padding, *self.padding)
            else:
                pad = self.padding
            x_quant = torch.nn.functional.pad(x_quant, pad, 'constant', self.padding_value)
        y = torch.nn.functional.conv2d(
            x_quant,
            W_quant,
            self.bias, # typically nil
            self.stride,
            self.padding if not self.deployment or self.bias is None else 0,
            self.dilation,
            self.groups
        )
        if not self.training and self.quantize_W:
            del W_quant
        # y is returned non-quantized, as it is assumed to be quantized after BN
        return y

class PACT_Conv1d(torch.nn.Conv1d):
    r"""PACT (PArametrized Clipping acTivation) 1d convolution, extending :py:class:`torch.nn.Conv1d`.

    Implements a :py:class:`torch.nn.Module` to implement PACT-like convolution. It uses PACT-like quantized weights and can be configured to quantize also input activations.
    It assumes full-precision storage of partial results.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        quantize_x=False,
        quantize_W=True,
        W_precision=None,
        x_precision=None,
        alpha=1.,
        quant_asymm=True,
        **kwargs
    ):
        r"""Constructor. Supports all arguments supported by :py:class:`torch.nn.Conv2d` plus additional ones.

        :param quantize_x: if True, quantize input activations (default False).
        :type  quantize_x: bool
        :param quantize_W: if True, quantize weights (default True).
        :type  quantize_W: bool
        :param x_precision: precision to be used for quantization of input activations (default None).
        :type  x_precision: :py:class:`nemo.precision.Precision`
        :param W_precision: precision to be used for quantization of weights (default None).
        :type  W_precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha` (default 1.0).
        :type  alpha: `torch.Tensor` or float
        :param quant_asymm: use asymmetric quantization for weights (default `True`).
        :type  quant_asymm: bool
        """

        if W_precision is None:
            self.W_precision = Precision()
        else:
            self.W_precision = W_precision
        if x_precision is None:
            self.x_precision = Precision()
        else:
            self.x_precision = x_precision

        self.quantize_x = quantize_x
        self.quantize_W = quantize_W
        super(PACT_Conv1d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W_alpha = torch.nn.Parameter(torch.Tensor((alpha,)).to(device))
        if quant_asymm:
            self.W_beta  = torch.nn.Parameter(torch.Tensor((alpha,)).to(device))

        self.x_alpha = torch.nn.Parameter(torch.Tensor((2.0,)).to(device))
        self.weight.data.uniform_(-1., 1.)

        # FIXME to implement: fix alpha,beta scaling factors for "beautiful" quantization in INT4,6,8

        self.quant_asymm = quant_asymm

        self.train_loop = True
        self.deployment = False
        self.train_loop_oldprec = None

    # this is for ONNX export -- right now, ignore quantization (FIXME)
    def symbolic(g, self, input):
        return g.op("Conv", self)

    def reset_alpha_weights(self, use_max=True, nb_std=5., verbose=False, **kwargs):
        r"""Resets :math:`\alpha` and :math:`\beta` parameters for weights.

        """

        if not self.quant_asymm:
            self.W_alpha.data[0] = self.weight.data.abs().max()
        elif use_max:
            self.W_alpha.data[0] = -self.weight.data.min()
            self.W_beta.data [0] =  self.weight.data.max()
        else:
            self.W_alpha.data[0] = -self.weight.data.mean() + nb_std*self.weight.data.std()
            self.W_beta.data[0]  =  self.weight.data.mean() + nb_std*self.weight.data.std()
        if verbose:
            logging.info("[Quant] W_alpha = %.5f" % self.W_alpha.data[0])
            logging.info("[Quant] W_beta  = %.5f" % self.W_beta.data[0])

    def harden_weights(self):
        r"""Replaces the current value of weight tensors (full-precision, quantized on forward-prop) with the quantized value.

        """

        if self.quant_asymm:
            self.weight.data = pact_quantize_asymm_inference(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta, train_loop=False, train_loop_oldprec=self.train_loop_oldprec)
        else:
            self.weight.data = pact_quantize_signed_inference(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)

    def integerize_weights(self):
        r"""Replaces the current value of weight tensors with the integer weights (i.e., the weight's quantized image).

        """

        if self.quant_asymm:
            eps = (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1)
            self.weight.data = pact_quantize_asymm_inference(self.weight, eps, self.W_alpha, self.W_beta, train_loop=False) / eps
        else:
            eps = 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1)
            self.weight.data = pact_quantize_signed_inference(self.weight, eps, self.W_alpha) / eps

    def prune_weights(self, threshold=0.1, eps=2**-9.):
        # logging.info("[Pruning] tau=%.1e", threshold)
        # stdev_per_chan = self.weight.data.std ((2), keepdim=True)
        # mean_per_chan  = self.weight.data.mean((2), keepdim=True) # + eps
        # self.weight.data = torch.where((self.weight.data - mean_per_chan).abs() < stdev_per_chan*threshold, mean_per_chan, self.weight.data)
        # wc = self.weight.data.clone().detach().to('cpu').numpy().flatten()
        # logging.info("[Pruning] Pruned %d" % np.count_nonzero(wc < eps))
        # return np.count_nonzero(wc < eps)
        return 0

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """
        if self.quant_asymm:
            eps_W = (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1)
        else:
            eps_W = 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1)
        return eps_W * eps_in

    def forward(self, input):
        r"""Forward-prop function for PACT-quantized 1d-convolution.

        :param input: input activations tensor.
        :type  input: `torch.Tensor`

        """

        if self.training and self.quantize_W and not self.deployment:
            if self.quant_asymm:
                W_quant = pact_quantize_asymm(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta)
            else:
                W_quant = pact_quantize_signed(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)
        elif self.quantize_W and not self.deployment:
            if self.quant_asymm:
                W_quant = pact_quantize_asymm_inference(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta, train_loop=self.train_loop)
            else:
                W_quant = pact_quantize_signed_inference(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)
        else:
            W_quant = self.weight
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return torch.nn.functional.conv1d(torch.nn.functional.pad(input, expanded_padding, mode='circular'),
                            W_quant, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return torch.nn.functional.conv1d(input, W_quant, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class PACT_Linear(torch.nn.Linear):
    r"""PACT (PArametrized Clipping acTivation) fully connected layer, extending :py:class:`torch.nn.Linear`.

    Implements a :py:class:`torch.nn.Module` to implement PACT-like matrix multiplication. It uses PACT-like quantized weights and can be configured to quantize also input activations.
    It assumes full-precision storage of partial results.

    """

    def __init__(
        self,
        in_features,
        out_features,
        quantize_x=False,
        quantize_W=True,
        W_precision=None,
        x_precision=None,
        alpha=1.,
        quant_asymm=True,
        quant_pc=False,
        **kwargs
    ):
        if W_precision is None:
            self.W_precision = Precision()
        else:
            self.W_precision = W_precision
        if x_precision is None:
            self.x_precision = Precision()
        else:
            self.x_precision = x_precision

        self.quantize_x = quantize_x
        self.quantize_W = quantize_W
        super(PACT_Linear, self).__init__(in_features, out_features, **kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W_alpha = torch.nn.Parameter(torch.Tensor((alpha,)).to(device))
        if quant_asymm:
            self.W_beta  = torch.nn.Parameter(torch.Tensor((alpha,)).to(device))
        self.x_alpha = torch.nn.Parameter(torch.Tensor((2.0,)).to(device))
        self.weight.data.uniform_(-1., 1.)

        self.quant_asymm = quant_asymm
        self.quant_pc    = quant_pc
        
        self.train_loop = True
        self.deployment = False
        self.train_loop_oldprec = None

    def symbolic(g, self, input):
        return g.op("MatMul", self)

    def reset_alpha_weights(self, use_max=True, nb_std=5., verbose=False, **kwargs):
        r"""Resets :math:`\alpha` and :math:`\beta` parameters for weights.

        """

        if not self.quant_asymm:
            self.W_alpha.data[0] = self.weight.data.abs().max()
        elif use_max:
            self.W_alpha.data[0] = -self.weight.data.min()
            self.W_beta.data [0] =  self.weight.data.max()
        else:
            self.W_alpha.data[0] = -self.weight.data.mean() + nb_std*self.weight.data.std()
            self.W_beta.data[0]  =  self.weight.data.mean() + nb_std*self.weight.data.std()
        if verbose:
            logging.info("[Quant] W_alpha = %.5f" % self.W_alpha.data[0])
            logging.info("[Quant] W_beta  = %.5f" % self.W_beta.data[0])

    def harden_weights(self):
        r"""Replaces the current value of weight tensors (full-precision, quantized on forward-prop) with the quantized value.

        """

        if self.quant_asymm:
            self.weight.data = pact_quantize_asymm_inference(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta, train_loop=False, train_loop_oldprec=self.train_loop_oldprec)
        else:
            self.weight.data = pact_quantize_signed_inference(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)

    def integerize_weights(self):
        r"""Replaces the current value of weight tensors with the integer weights (i.e., the weight's quantized image).

        """

        if self.quant_asymm:
            eps = (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1)
            self.weight.data = pact_quantize_asymm_inference(self.weight, eps, self.W_alpha, self.W_beta, train_loop=False) / eps
        else:
            eps = 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1)
            self.weight.data = pact_quantize_signed_inference(self.weight, eps, self.W_alpha) / eps

    def prune_weights(self, threshold=0.1, eps=2**-9.):
        r"""Prunes the weights of the layer.

        The pruning is performed by replacing all weights that are "near" to the mean of the weights with the mean itself.
        "Near" weights are those that differ with the mean by less than `threshold` times the standard deviation.

        :param threshold: threshold used for pruning (default 0.1).
        :type  threshold: float
        :param eps: parameter used to compare the difference with 0 (default 2^-9).
        :type  eps: float

        """

        logging.info("[Pruning] tau=%.1e", threshold)
        stdev_per_chan = self.weight.data.std ()
        mean_per_chan  = self.weight.data.mean() # + eps
        self.weight.data = torch.where((self.weight.data - mean_per_chan).abs() < stdev_per_chan*threshold, mean_per_chan, self.weight.data)
        wc = self.weight.data.clone().detach().to('cpu').numpy().flatten()
        logging.info("[Pruning] Pruned %d" % np.count_nonzero(wc < eps))
        return np.count_nonzero(wc < eps)

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.

        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`

        """

        if self.quant_asymm:
            eps_W = (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1)
        else:
            eps_W = 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1)
        return eps_W * eps_in

    def forward(self, input):
        r"""Forward-prop function for PACT-quantized linear layer.

        :param input: input activations tensor.
        :type  input: `torch.Tensor`

        """

        if self.training and self.quantize_x and not self.deployment:
            x_quant = pact_quantize_signed(input, self.x_alpha/(2.0**(self.x_precision.get_bits())-1), self.x_alpha)
        else:
            x_quant = input
        if self.training and self.quantize_W and not self.deployment:
            if self.quant_asymm:
                W_quant = pact_quantize_asymm(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta)
            else:
                W_quant = pact_quantize_signed(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)
        elif self.quantize_W and not self.deployment:
            if self.quant_asymm:
                W_quant = pact_quantize_asymm_inference(self.weight, (self.W_beta+self.W_alpha)/(2.0**(self.W_precision.get_bits())-1), self.W_alpha, self.W_beta, train_loop=self.train_loop)
            else:
                W_quant = pact_quantize_signed_inference(self.weight, 2*self.W_alpha/(2.0**(self.W_precision.get_bits())-1), self.W_alpha)
        else:
            W_quant = self.weight
        y = torch.nn.functional.linear(x_quant, W_quant, self.bias)
        if not self.training and self.quantize_W:
            del W_quant
        # y is returned non-quantized, as it is assumed to be quantized after BN
        return y
