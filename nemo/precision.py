#
# precision.py
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

import numpy as np

MIN_REPR_PRECISION = 1e-15
MAX_REPR_SCALE     = 1e+15
MAX_NB_BITS        = 32

class Precision():
    def __init__(self, bits=None, scale=None, positive=False):
        super(Precision, self).__init__()
        self.bits  = bits
        self.scale = scale
        self.positive = positive

    def __gt__(self, other):
        try:
            if self.bits > other.bits:
                return True
            else:
                return False
        except AttributeError:
            if self.bits > other:
                return True
            else:
                return False

    def __lt__(self, other):
        try:
            if self.bits < other.bits:
                return True
            else:
                return False
        except AttributeError:
            if self.bits < other:
                return True
            else:
                return False

    def __ge__(self, other):
        try:
            if self.bits >= other.bits:
                return True
            else:
                return False
        except AttributeError:
            if self.bits >= other:
                return True
            else:
                return False

    def __le__(self, other):
        try:
            if self.bits <= other.bits:
                return True
            else:
                return False
        except AttributeError:
            if self.bits <= other:
                return True
            else:
                return False

    def __eq__(self, other):
        try:
            if self.bits == other.bits:
                return True
            else:
                return False
        except AttributeError:
            if self.bits == other:
                return True
            else:
                return False

    def __ne__(self, other):
        try:
            if self.bits != other.bits:
                return True
            else:
                return False
        except AttributeError:
            if self.bits != other:
                return True
            else:
                return False

    def set_eps(self, eps):
        self.bits = np.log2(-eps)

    def set_bits(self, bits):
        self.bits = bits

    def set_clip(self, clip):
        self.scale = clip

    def set_scale(self, scale):
        self.scale = scale
    
    def get_eps(self):
        if self.bits is None or self.scale is None:
            return MIN_REPR_PRECISION
        if not self.positive:
            return 2.0**(-(self.bits-1)) * self.scale
        else:
            return 2.0**(-self.bits) * self.scale

    def get_clip(self):
        return self.get_scale()

    def get_scale(self):
        if self.scale is None:
            return MAX_REPR_SCALE
        return self.scale

    def get_bits(self):
        if self.bits is None:
            return MAX_NB_BITS
        return self.bits
