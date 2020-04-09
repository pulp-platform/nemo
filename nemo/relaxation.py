#
# relaxation.py
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

class RelaxationEngine():
    r"""Main engine class for weight/activation precision relaxation procedure.

        :param net: Module or network on which the relaxation procedure will be performed (must have a `change_precision` method).
        :type  net: `torch.nn.Module`

        :param optimizer: A pointer to the optimizer being used for the training.
        :type  optimizer: `torch.optim.Optimizer`

        :param criterion: Loss function being used as a quality metric.
        :type  criterion: `torch.autograd.Function`

        :param trainloader: Loader used for training values (used for relaxation evaluation).
        :type  trainloader: `torch.utils.data.DataLoader`

        :param precision_rule: A dictionary describing the rules to be used for the precision relaxation.
        :type  precision_rule: `dict` or `collections.OrderedDict`

        :param tbx_writer: TensorBoardX writer for data logging. Defaults to `None`.
        :type  tbx_writer: `tensorboardx.SummaryWriter`

        :param reset_alpha_weights: If True, reset the W_alpha and W_beta parameters at precision change.
        :type  reset_alpha_weights: `bool`

        :param min_prec_dict: Dictionary of minimum allowed precision for all parameters.
        :type  min_prec_dict: `dict` or `collections.OrderedDict`

        :param evaluator: Evaluation engine for precision selection heuristics.
        :type  evaluator: `nemo.evaluation.EvaluationEngine`

        :param evaluator_threshold: Threshold to be used for precision binning (default 0.9).
        :type  evaluator_threshold: `float`

        :param evaluator_verbose: If True, print more information from evaluation engine.
        :type  evaluator_verbose: `bool`

        :param evaluator_strategy: Can be 'max_accuracy' (default) or 'min_precision'.
        :type  evaluator_strategy: `string`

        :param divergence_policy: Can be 'change_precision' (default) or 'lower_lr'.
        :type  divergence_policy: `string`

        :param divergence_lr_scaler: LR scaling factor for 'lower_lr' divergence policy.
        :type  divergence_lr_scaler: `float`

        The :py:class:`RelaxationEngine` represents a precision-scaling procedure utilizing the relaxation
        heuristic.
        The main parameters of the relaxation heuristic are passed through the `precision_rule`
        dictionary when initializing the class.

        Assuming `p` is the `precision_rule`, the relaxation heuristic keeps an internal memory
        of the loss and its relative variation (`delta_loss`) over the last `p[running_avg_memory]` epochs.
        The mean and standard deviation of `delta_loss` over `p[running_avg_memory]` epochs are compared with 
        `p[delta_loss_less_than]` and `p[delta_loss_running_std_stale]`. Moreover the absolute loss value
        is compared with `p[abs_loss_stale]`.
        
        Two counters are updated:
        - for each consecutive epoch in which all the three values compared are less than the respective
        parameters from `p`, an `abs_bound` counter is updated
        - for each consecutive epoch in which the two delta values compared are less than the respective
        parameters from `p`, a `no_abs_bound` counter is updated

        After the counters are updated, their respective value is compared with `p[for_epochs]` and
        `p[for_epochs_no_abs_bound]`: if any of the counters is higher than the respective parameter, the
        training is considered "stale" for the current quantization value.

        When this happens, precision is scaled down by a factor of `p[bit_scaler]` bits, up to the point
        when `p[bit_stop_condition]` precision is reached. If `p[scale_lr]` is set to true, the learning
        rate is also downscaled by a factor of `p[lr_scaler]`.
    """

    def __init__(self, net, optimizer, criterion, trainloader, precision_rule=None, tbx_writer=None, reset_alpha_weights=True, min_prec_dict=None, evaluator=None, evaluator_threshold=0.9, evaluator_verbose=False, evaluator_strategy='min_precision', divergence_policy='change_precision', divergence_lr_scaler=0.2, evaluate=None, log_start=None, log_stop=None, log_step=None, validate_on_train_fn=None, reset_alpha_below_6bits=False):
        super(RelaxationEngine, self).__init__()

        self.precision_rule = precision_rule
        if self.precision_rule is None:
            self.scale_activations = True
            self.scale_weights = True
        else:
            try:
                self.scale_activations = self.precision_rule['scale_activations']
            except KeyError:
                self.scale_activations = True
            try:
                self.scale_weights = self.precision_rule['scale_weights']
            except KeyError:
                self.scale_weights = True

        if self.precision_rule is not None:
            self.delta_loss_running_avg = 1000.0
            self.delta_loss_running_std = 0.0
            self.delta_loss_memory_curr_size = 0
            self.delta_loss_memory_size = self.precision_rule['running_avg_memory']
            self.delta_loss_memory = np.zeros(self.precision_rule['running_avg_memory'])
            
            try:
                cs = self.precision_rule['custom_scaler']
            except KeyError:
                self.precision_rule['custom_scaler'] = None

        self.loss_best = 1e3
        self.loss_prev = 1e3
        self.net = net
        self.optimizer = optimizer
        self.precision_abs_bound_counter = 0
        self.precision_no_abs_bound_counter = 0
        self.criterion = criterion
        self.trainloader = trainloader

        self.tbx_writer = tbx_writer
        self.reset_alpha_weights = reset_alpha_weights

        self.min_prec_dict = min_prec_dict
        self.evaluator = evaluator
        self.evaluator_threshold = evaluator_threshold
        self.evaluator_verbose = evaluator_verbose
        self.evaluator_strategy = evaluator_strategy

        self.divergence_policy = divergence_policy
        self.divergence_lr_scaler = divergence_lr_scaler
        self.divergence_cnt = 0
        self.divergence_lrscaling_limit = 3

        self.validate_on_train_fn = validate_on_train_fn

        self.reset_alpha_below_6bits = reset_alpha_below_6bits

        self.relaxation_ended = False

    def step(self, loss, epoch=0, checkpoint_name='checkpoint', previous_loss=None):
        r"""Iterate a step over the relaxation engine, checking the current convergence rate and updating precision and LR.

            :param loss: Current value of the training loss function.
            :type  loss: `torch.Tensor`

            :param epoch: Epoch of training.
            :type  epoch: `int`

            :param checkpoint_name: String to be used as a name for the checkpoint file.
            :type  checkpoint_name: `str`

            The `step` method iterates the `RelaxationEngine` using as input the current value of the training loss function,
            whose convergence is evaluated.
        """

        reset_alpha_weights = self.reset_alpha_weights
        if self.precision_rule is None:
            return

        # save best result in case of catastrophic failure
        if loss < self.loss_best:
            self.loss_best = loss
            if checkpoint_name is not None:
                nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name=checkpoint_name, checkpoint_suffix="_current_best")
            else:
                nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name="checkpoint", checkpoint_suffix="_current_best")

        try:
            curr_regime = self.precision_rule[epoch]
        except KeyError:
            try:
               curr_regime = self.precision_rule[str(epoch)]
            except KeyError:
               curr_regime = None

        for i in range(self.delta_loss_memory_size-1, 0, -1):
            self.delta_loss_memory[i] = self.delta_loss_memory[i-1]
        self.delta_loss_memory[0] = self.loss_prev-loss
        if self.delta_loss_memory_curr_size < self.delta_loss_memory_size:
            self.delta_loss_memory_curr_size += 1
        if self.delta_loss_memory_curr_size > 1:
            delta_loss_running_avg = self.delta_loss_memory[:self.delta_loss_memory_curr_size-1].mean()
            delta_loss_running_std = self.delta_loss_memory[:self.delta_loss_memory_curr_size-1].std()
        else:
            delta_loss_running_avg = np.Inf
            delta_loss_running_std = np.Inf
        logging.info("[Relax @%d]\t delta_loss_running_avg=%.3e loss_epoch_m1=%.3e delta_loss=%.3e" % (epoch-1, delta_loss_running_avg, loss, self.loss_prev-loss))
        logging.info("[Relax @%d]\t delta_loss_memory=%s" % (epoch-1, self.delta_loss_memory[:self.delta_loss_memory_curr_size-1]))

        if self.tbx_writer is not None:
            self.tbx_writer.add_scalars('train', { 'delta_loss_avg': delta_loss_running_avg, 'delta_loss_std': delta_loss_running_std }, epoch+1)

        # staleness happens when 1) delta_loss has bounded mean and std and absolute loss is bounded for for_epochs, or 2) delta_loss has bounded mean and std for for_epochs_no_abs_bound
        if delta_loss_running_avg < self.precision_rule['delta_loss_less_than'] and delta_loss_running_std < self.precision_rule['delta_loss_running_std_stale'] and loss < self.precision_rule['abs_loss_stale']:
            self.precision_abs_bound_counter += 1
        else:
            self.precision_abs_bound_counter = 0
        if delta_loss_running_avg < self.precision_rule['delta_loss_less_than'] and delta_loss_running_std < self.precision_rule['delta_loss_running_std_stale']:
            self.precision_no_abs_bound_counter += 1
        else:
            self.precision_no_abs_bound_counter = 0

        divergence_chprec_flag = False
        divergence_lowlr_flag = False
        # catastrophic failure occurs when delta_loss_running_avg is negative
        try:
            divergence_abs_threshold = self.precision_rule['divergence_abs_threshold']
        except KeyError:
            divergence_abs_threshold = 1e9
        if delta_loss_running_avg < 0 or loss > self.precision_rule['divergence_abs_threshold']:
            # recover previous state
            state = torch.load("checkpoint/"+checkpoint_name+"_current_best.pth")['state_dict']
            self.net.load_state_dict(state, strict=True)
            logging.info("[Relax @%d]\t Detected divergent training, restoring previous best state." % (epoch-1))

            if (self.divergence_policy == 'lower_lr' and self.divergence_cnt < self.divergence_lrscaling_limit) or self.relaxation_ended:
                # reset delta loss memory (with a small delta)
                self.precision_abs_bound_counter = 0
                self.precision_no_abs_bound_counter = 0
                loss = self.loss_best
                self.delta_loss_memory[:] = 0
                self.delta_loss_memory_curr_size = 1
                self.divergence_cnt += 1
                if self.divergence_cnt == 1:
                    self.divergence_saved_lr = list(self.optimizer.param_groups)[0]['lr']

                # scale LR together with W bits
                for p in self.optimizer.param_groups:
                    p['lr'] *= self.divergence_lr_scaler
                    lr = p['lr']
                logging.info("[Relax @%d]\t Using 'lower_lr' policy (iter %d); scaled LR to %.3e" % (epoch-1, self.divergence_cnt, lr))

                # report this as a precision change
                divergence_lowlr_flag = True
            elif self.divergence_policy == 'change_precision' or self.divergence_cnt >= self.divergence_lrscaling_limit:
                divergence_chprec_flag = True
                self.divergence_cnt = 0
                logging.info("[Relax @%d]\t Using 'change_precision' policy." % (epoch-1))
            else:
                raise NotImplementedError

        # if relaxation has already ended, exit here
        if self.relaxation_ended:
            return divergence_lowlr_flag, True

        change_precision = divergence_lowlr_flag # False in normal cases

        if curr_regime is not None:
            self.precision_abs_bound_counter = 0
            self.precision_no_abs_bound_counter = 0
            if loss is None:
                loss = 1e3
            self.loss_prev = 1e3
            suffix  = '_' + "%.1f" % (self.net.W_precision.get_bits()) + 'b'
            suffix += 'x' + "%.1f" % (self.net.x_precision.get_bits()) + 'b'
            if checkpoint_name is not None:
                nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name=checkpoint_name, checkpoint_suffix=suffix)
            self.net.change_precision(bits=curr_regime['W_bits'], scale_activations=False, scale_weights=True, reset_alpha=reset_alpha_weights, min_prec_dict=self.min_prec_dict)
            self.net.change_precision(bits=curr_regime['x_bits'], scale_activations=True, scale_weights=False, reset_alpha=reset_alpha_weights, min_prec_dict=self.min_prec_dict)

            # save checkpoint for catastrophic failure case
            self.loss_best = 1e3
            if checkpoint_name is None:
                nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name='checkpoint', checkpoint_suffix="_current_best")
            else:
                nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name=checkpoint_name, checkpoint_suffix="_current_best")

        elif self.precision_abs_bound_counter == self.precision_rule['for_epochs'] or self.precision_no_abs_bound_counter == self.precision_rule['for_epochs_no_abs_bound'] or divergence_chprec_flag:

            if self.precision_abs_bound_counter == self.precision_rule['for_epochs']:
                logging.info("[Relax @%d]\t precision_abs_bound_counter=%d: Triggering precision change below absolute loss threshold" % (epoch-1, self.precision_abs_bound_counter))
            elif self.precision_no_abs_bound_counter == self.precision_rule['for_epochs_no_abs_bound']:
                logging.info("[Relax @%d]\t precision_no_abs_bound_counter=%d: Triggering precision change above absolute loss threshold" % (epoch-1, self.precision_no_abs_bound_counter))

            self.precision_abs_bound_counter = 0
            self.precision_no_abs_bound_counter = 0
            loss = 1e3
            self.loss_prev = 2e3
            self.delta_loss_memory[:] = 1e3
            self.delta_loss_memory_curr_size = 1

            try:
                scale_x = self.precision_rule['scale_x']
            except KeyError:
                scale_x = True
            try:
                scale_W = self.precision_rule['scale_W']
            except KeyError:
                scale_W = True

            # stop condition is currently measured against W_precision.bits
            if (self.net.W_precision.get_bits() >= self.precision_rule['W_bit_stop_condition']) or \
               (self.net.x_precision.get_bits() >= self.precision_rule['x_bit_stop_condition']):

                # save checkpoint
                if checkpoint_name is not None:
                    suffix  = '_' + "%.1f" % (self.net.W_precision.get_bits()) + 'b'
                    suffix += 'x' + "%.1f" % (self.net.x_precision.get_bits()) + 'b'
                    nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name=checkpoint_name, checkpoint_suffix=suffix)

                # if there is an EvaluationEngine available, use it
                if self.evaluator is not None:
                    W_start = self.net.W_precision.get_bits()
                    x_start = self.net.x_precision.get_bits()
                    # pure heuristics...
                    if W_start > 8:
                        W_step = -2
                        W_stop = max(6, self.precision_rule['W_bit_stop_condition']-2) if scale_W else W_start+W_step
                    elif W_start > 6:                      
                        W_step = -1
                        W_stop = max(5, self.precision_rule['W_bit_stop_condition']-1) if scale_W else W_start+W_step
                    else:
                        W_step = -0.5
                        W_stop = max(3.5, self.precision_rule['W_bit_stop_condition']-0.5) if scale_W else W_start+W_step
                    if x_start > 8:
                        x_step = -2
                        x_stop = max(6, self.precision_rule['x_bit_stop_condition']-2) if scale_x else x_start+x_step
                    elif x_start > 6:
                        x_step = -1
                        x_stop = max(5, self.precision_rule['x_bit_stop_condition']-1) if scale_x else x_start+x_step
                    else:
                        x_step = -0.5
                        x_stop = max(3.5, self.precision_rule['x_bit_stop_condition']-0.5) if scale_x else x_start+x_step
                    self.net.unset_train_loop() # this is a "soft" weight hardening
                    self.evaluator.reset_grids(W_start, W_stop, W_step, x_start, x_stop, x_step)
                    while self.evaluator.step():
                        acc = self.evaluator.validate_fn(0, val_loader=self.evaluator.validate_data)
                        self.evaluator.report(acc)
                        logging.info("[Relax @%d]\t %.1f-bit W, %.1f-bit x %.2f%%" % (epoch-1, self.evaluator.wgrid[self.evaluator.idx], self.evaluator.xgrid[self.evaluator.idx], 100*acc.item()))
                    self.net.set_train_loop() # this removes the "soft" hardening
                    Wbits, xbits = self.evaluator.get_next_config(upper_threshold=self.evaluator_threshold, verbose=self.evaluator_verbose, strategy=self.evaluator_strategy)
                    Wdiff = W_start - Wbits
                elif self.precision_rule['custom_scaler'] is not None:
                    scaler = self.precision_rule['custom_scaler']
                    if len(scaler) == 0:
                        return True, True
                    Wbits, xbits, lrscaled, divpol = scaler.pop(0)
                    self.divergence_policy = divpol # update divergence policy
                    W_diff = self.net.W_precision.get_bits() - Wbits
                else:
                    Wdiff = -self.precision_rule['W_bit_scaler']
                    Wbits = self.net.W_precision.get_bits()+self.precision_rule['W_bit_scaler']
                    xbits = self.net.x_precision.get_bits()+self.precision_rule['x_bit_scaler']
                    
                logging.info("[Relax @%d]\t Choosing %.1f-bit W, %.1f-bit x for next step" % (epoch-1, Wbits, xbits))
                if scale_W:
                    self.net.change_precision(bits=Wbits, scale=self.net.W_precision.get_scale(), scale_activations=False, scale_weights=self.scale_weights,     reset_alpha=reset_alpha_weights, min_prec_dict=self.min_prec_dict)
                    # this will reset alpha,beta PACT parameters to 5 standard deviations upon precision change, to avoid wasting dynamic range to represent irrealistic weights
                    if Wbits < 6 and self.reset_alpha_below_6bits:
                        self.net.reset_alpha_weights(stdev=5.)
                        logging.info("[Relax @%d]\t Setting alpha,beta params of weights to %.1f std deviations" % (epoch-1, 5))
                if scale_x:
                    self.net.change_precision(bits=xbits, scale=self.net.x_precision.get_scale(), scale_activations=self.scale_activations, scale_weights=False, reset_alpha=reset_alpha_weights, min_prec_dict=self.min_prec_dict)
                change_precision = True
                if self.divergence_cnt > 0:
                    for p in self.optimizer.param_groups:
                        p['lr'] = self.divergence_saved_lr
                self.divergence_cnt = 0
                try:
                    if self.evaluator is not None:
                        # scale LR together with W bits
                        for p in self.optimizer.param_groups:
                            p['lr'] *= (2**Wdiff)
                            lr = p['lr']
                        logging.info("[Relax @%d]\t Scaled LR to %.3e" % (epoch-1, lr))
                    elif self.precision_rule['custom_scaler'] is not None:
                        for p in self.optimizer.param_groups:
                            p['lr'] = lrscaled
                            lr = p['lr']
                        logging.info("[Relax @%d]\t Scaled LR to %.3e" % (epoch-1, lr))
                    elif self.precision_rule['scale_lr']:
                        for p in self.optimizer.param_groups:
                            p['lr'] *= self.precision_rule['lr_scaler']
                            lr = p['lr']
                        logging.info("[Relax @%d]\t Scaled LR to %.3e" % (epoch-1, lr))
                except KeyError:
                    pass
                if self.validate_on_train_fn is not None:
                    loss = self.validate_on_train_fn(epoch)
                    logging.info("[Relax @%d]\t validate_on_train loss=%.3e" % (epoch-1, loss.item()))

                # save checkpoint for catastrophic failure case
                self.loss_best = loss
                nemo.utils.save_checkpoint(self.net, self.optimizer, epoch-1, checkpoint_name=checkpoint_name, checkpoint_suffix="_current_best")

            else:
                self.relaxation_ended = True
                logging.info("[Relax @%d]\t Precision relaxation procedure ended" % (epoch-1))
                return True, True
            
        if self.tbx_writer is not None:
            self.tbx_writer.add_scalars('train', { 'abs_bound_counter': self.precision_abs_bound_counter, 'no_abs_bound_counter': self.precision_no_abs_bound_counter, }, epoch+1)
            lr_save = list(self.optimizer.param_groups)[0]['lr']
            self.tbx_writer.add_scalars('train', { 'lr': lr_save }, epoch+1)
        logging.info("[Relax @%d]\t delta_loss_running_avg=%.3e loss_epoch_m1=%.3e delta_loss=%.3e" % (epoch-1, delta_loss_running_avg, loss, self.loss_prev-loss))
        logging.info("[Relax @%d]\t delta_loss_memory=%s" % (epoch-1, self.delta_loss_memory[:self.delta_loss_memory_curr_size-1]))
        logging.info("[Relax @%d]\t precision_abs_bound_counter=%d precision_no_abs_bound_counter=%d" % (epoch-1, self.precision_abs_bound_counter, self.precision_no_abs_bound_counter))

        self.loss_prev = loss

        # if precision has changed, signal this upstream
        return change_precision, False
