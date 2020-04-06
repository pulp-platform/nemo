#
# evaluation.py
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
import logging
import nemo
import torch
from collections import OrderedDict
from tqdm import tqdm

class EvaluationEngine():
    def __init__(self, net, validate_fn=None, validate_data=None, precision_rule=None, min_prec_dict=None):
        super(EvaluationEngine, self).__init__()

        self.precision_rule = precision_rule
        if self.precision_rule is None:
            self.scale_activations = True
            self.scale_weights = True
        self.min_prec_dict = min_prec_dict
        self.loss_prev = 1e3
        self.net = net
        self.validate_fn = validate_fn
        self.validate_data = validate_data
        self.reset_grids()

    def reset_grids(self, W_start=None, W_stop=None, W_step=None, x_start=None, x_stop=None, x_step=None):
        first_W = self.precision_rule['0']['W_bits'] if W_start is None else W_start
        first_x = self.precision_rule['0']['x_bits'] if x_start is None else x_start
        step_W = float(self.precision_rule['W_bit_scaler']) if W_step is None else W_step
        step_x = float(self.precision_rule['x_bit_scaler']) if x_step is None else x_step
        last_W = float(self.precision_rule['W_bit_stop_condition'])+step_W if W_stop is None else W_stop
        last_x = float(self.precision_rule['x_bit_stop_condition'])+step_x if x_stop is None else x_stop
        W = np.arange(first_W, last_W, step_W)
        x = np.arange(first_x, last_x, step_x)
        if len(W) == 0:
            W = np.asarray([first_W,])
        if len(x) == 0:
            x = np.asarray([first_x,])
        logging.info("[Evaluation]\t Setting up new grid: W=%s x=%s" % (W, x))
        wgrid, xgrid = np.meshgrid(W, x)
        self.acc = np.zeros_like(wgrid, dtype='float32')
        self.wgrid = wgrid.flatten()
        self.xgrid = xgrid.flatten()
        self.idx = -1

    def report(self, acc):
        sh = self.acc.shape
        self.acc = self.acc.flatten()
        self.acc[self.idx] = acc.item()
        self.acc = self.acc.reshape(sh)

    def __suffix(self):
        return "_%.1fx%.1fb" % (self.net.W_precision.get_bits(), self.net.x_precision.get_bits())

    def step(self, checkpoint_name='checkpoint', verbose=False):
        if self.precision_rule is None:
            return
        try:
            curr_regime = self.precision_rule[0]
        except KeyError:
            try:
               curr_regime = self.precision_rule[str(0)]
            except KeyError:
               curr_regime = None

        if self.idx == self.wgrid.shape[0] - 1:
            return False

        self.idx += 1
        self.net.change_precision(bits=self.wgrid[self.idx], scale_activations=False, scale_weights=True, reset_alpha=False, verbose=verbose, min_prec_dict=self.min_prec_dict)
        self.net.change_precision(bits=self.xgrid[self.idx], scale_activations=True, scale_weights=False, reset_alpha=False, verbose=verbose, min_prec_dict=self.min_prec_dict)
        return True

    # the rationale here is to define frontiers using a certain threshold, e.g. 90% of the top accuracy
    # reached by pure evaluation. configurations within the frontier are considered 'good enough'
    # and not worth being considered for retraining (only minor fine-tuning is necessary).
    def get_next_config(self, upper_threshold=0.9, strategy='min_precision', verbose=False, Wbits_curr=None, xbits_curr=None, timeout=25):
        def create_bins(upper_threshold):
            bins = np.asarray([0, 1.0-upper_threshold, upper_threshold]) * self.acc.max()
            acc_idx = np.digitize(self.acc, bins)
            return acc_idx
        acc_idx = create_bins(upper_threshold)
        for i in range(timeout):
            if len(acc_idx[acc_idx == 2]) == 0:
                # exponentially increase threshold
                upper_threshold /= upper_threshold
                if verbose:
                    logging.info("[Evaluation]\t No middle-bin element, changing threshold to %.3e." % upper_threshold)
                acc_idx = create_bins(upper_threshold)
        if len(acc_idx[acc_idx == 2]) == 0:
            return self.wgrid.reshape(self.acc.shape)[0,0], self.xgrid.reshape(self.acc.shape)[0,0]
        if verbose:
            logging.info("[Evaluation]\t Top-bin:    %s" % (np.dstack((self.wgrid.reshape(self.acc.shape)[acc_idx == 3], self.xgrid.reshape(self.acc.shape)[acc_idx == 3], self.acc[acc_idx == 3]))))
            logging.info("[Evaluation]\t Middle-bin: %s" % (np.dstack((self.wgrid.reshape(self.acc.shape)[acc_idx == 2], self.xgrid.reshape(self.acc.shape)[acc_idx == 2], self.acc[acc_idx == 2]))))
            logging.info("[Evaluation]\t Bottom-bin: %s" % (np.dstack((self.wgrid.reshape(self.acc.shape)[acc_idx == 1], self.xgrid.reshape(self.acc.shape)[acc_idx == 1], self.acc[acc_idx == 1]))))
        if strategy == 'max_accuracy':
            if verbose:
                logging.info("[Evaluation]\t Select the most accurate one from the middle-bin.")
            idxs = np.unravel_index(self.acc[acc_idx == 2].argmax(), self.acc.shape)
        elif strategy == 'min_precision': 
            if verbose:
                logging.info("[Evaluation]\t Select the lowest precision one from the middle-bin.")
            xmin = self.xgrid.reshape(self.acc.shape)[acc_idx == 2].min()
            mask = np.logical_and(acc_idx == 2, self.xgrid.reshape(self.acc.shape)==xmin)
            wtmpgrid = np.copy(self.wgrid.reshape(self.acc.shape))
            wtmpgrid[np.logical_not(mask)] = 1e6
            idxs = np.unravel_index(wtmpgrid.argmin(), self.acc.shape)
        Widx = idxs[1]
        xidx = idxs[0]
        if verbose:
            logging.info("[Evaluation]\t Returning idxs: %d,%d prec %d,%d" % (Widx, xidx, self.wgrid[Widx], self.xgrid[xidx]))
        return self.wgrid.reshape(self.acc.shape)[0,Widx], self.xgrid.reshape(self.acc.shape)[xidx,0]
