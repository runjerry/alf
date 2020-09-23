# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Encoding algorithm."""

import gin

from absl import logging
import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.networks import EncodingNetwork, ParallelEncodingNetwork

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score

from alf.algorithms.config import TrainerConfig
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time


NetworkLossInfo = namedtuple("NetworkLossInfo", ["loss", "extra"])


def classification_loss(output, target):
    pred = output.max(-1)[1]
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    loss = F.cross_entropy(output.transpose(1, 2), target)
    return NetworkLossInfo(loss=loss, extra=avg_acc)


def regression_loss(output, target):
    out_shape = output.shape[-1]
    assert (target.shape[-1] == out_shape), (
        "feature dimension of output and target does not match.")
    loss = 0.5 * F.mse_loss(
        output.reshape(-1, out_shape),
        target.reshape(-1, out_shape),
        reduction='sum')
    return NetworkLossInfo(loss=loss, extra=())


def neglogprob(inputs, net, ensemble_size, loss_type):
    if loss_type == 'regression':
        loss_func = regression_loss
    elif loss_type == 'classification':
        loss_func = classification_loss
    else:
        raise ValueError("Unsupported loss_type: %s" % loss_type)

    data, target = inputs
    output, _ = net(data)  # [B, N, D]
    if ensemble_size == 1:
        output = output.unsqueeze(1)
    target = target.unsqueeze(1).expand(*target.shape[:1], ensemble_size,
                                        *target.shape[1:])
    return loss_func(output, target), output


@gin.configurable
class ForwardNetwork(Algorithm):
    """Basic network training algorithm.

    This is just meant to support supervised learning algorithms by providing
    a standard interface to train forward models. 
    It can also be used as a generic way to train additional modules required
    by various RL algorithms or particle VI algorithms, such as Minmax 
    amortized SVGD. 
    """

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 use_fc_bn=False,
                 use_conv_bn=False,
                 ensemble_size=1,
                 ensemble_vote=None,
                 loss_type="classification",
                 optimizer=None,
                 ignore_module=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 debug_summaries=False,
                 config: TrainerConfig = None,
                 name="NetworkAlgorithm"):
        """
        Args:
            Args for the network to be trained
            ===================================================================
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            use_fn_bn (bool): whether to use Batch Norm on the fc layers
            use_fn_conv (bool): whether to use Batch Norm on the conv layers
            ensemble_size (int): how many independant models to train 
                concurrently

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the training network,
                types are [``classification``, ``regression``]
            voting (str): types of voting to make predictions from ensemble.
                if ``ensemble_size`` > 1, then voting is used,
                types are [``soft``, ``hard``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):

            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        
        assert (ensemble_size >= 1 and isinstance(ensemble_size, int)), " "\
            "``ensemble_size`` must be an int greater than or equal to 1 " \
            "got {}".format(ensemble_size)

        if ensemble_size == 1:
            ensemble_vote = None
        
        if last_layer_param is not None and isinstance(last_layer_param, tuple):
            last_layer_param = last_layer_param[0]

        if ensemble_size > 1:
            net = ParallelEncodingNetwork(
                input_tensor_spec,
                ensemble_size,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                last_layer_size=last_layer_param,
                last_activation=last_activation,
                name='Network')
            logging.info("Initialized ensemble of {} models".format(
                ensemble_size))
        else:
            net = EncodingNetwork(
                input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                last_layer_size=last_layer_param,
                last_activation=last_activation,
                name='Network')
            logging.info("Initialized single model")
        
        if logging_network:
            logger.info(net)

        self._net = net
        self._train_loader = None
        self._test_loader = None
        self._use_conv_bn = use_conv_bn
        self._use_fc_bn = use_fc_bn
        self._ensemble_size = ensemble_size
        assert ensemble_vote in ['soft', 'hard', None], "voting only supports "\
            "\"soft\", \"hard\", None"
        
        self._ensemble_vote = ensemble_vote
        if loss_type == "classification":
            self._vote = self._classification_vote
        elif loss_type == "regression":
            self._vote = self.regression_vote
        else:
            raise ValueError("Unsupported loss_type: {}".format(loss_type))
        self._loss_type = loss_type
        self._logging_training=logging_training
        self._logging_evaluate=logging_evaluate
        self._config = config
        self._ignore_module = ignore_module

    def _trainable_attributes_to_ignore(self):
        return ['_ignore_module']

    def set_data_loader(self, train_loader, test_loader=None, outlier=None):
        """ Set data loader for training and testing. """
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._outlier_train = outlier[0]
        self._outlier_test = outlier[1]
    
    def predict_with_update(self, inputs, loss_func, state=None):
        """ Does one step for prediction with update, used for 
            minmax asvgd critic training
        """
        outputs, _ = self._net(inputs)
        loss, loss_propagated = self._minmax_critic_grad(
            inputs,
            outputs,
            loss_func)
        alg_step = AlgStep(
            output=outputs,
            state=state,
            info=LossInfo(
                loss=loss_propagated,
                extra=loss))
        self.update_with_gradient(alg_step.info)
        return alg_step

    def train_iter(self, state=None):
        """ Predict ensemble outputs for inputs """
        assert self._train_loader is not None, "Must set data_loader first"
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            if self._loss_type == 'classification':
                avg_acc = []
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target), state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                loss += loss_info.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra)
        acc = None
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._loss_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
        self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)
        return batch_idx + 1

    def train_step(self, inputs, state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        loss, outputs = neglogprob(
            inputs,
            self._net,
            self._ensemble_size,
            self._loss_type,)
        return AlgStep(
            output=outputs,
            state=(),
            info=LossInfo(
                loss=loss.loss,
                extra=loss.extra))
    
    def evaluate(self):
        """Evaluate the ensemeble. """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._net(data)  # [B, N, D]
                if self._ensemble_size == 1:
                    output = output.unsqueeze(1)
                loss, extra = self._vote(output, target)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            alf.summary.scalar(name='eval/test_acc', data=test_acc * 100)
        if self._logging_evaluate:
            if self._loss_type == 'classification':
                logging.info("Test acc: {}".format(test_acc * 100))
            logging.info("Test loss: {}".format(test_loss))
        alf.summary.scalar(name='eval/test_loss', data=test_loss)


    def _classification_vote(self, output, target):
        """ensmeble the outputs from classifiers."""
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._ensemble_vote == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._ensemble_vote == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = pred.mode(1)[0]  # [B, 1]
        else:
            vote = probs.argmax(-1).cpu()
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1],
                                            self._ensemble_size,
                                            *target.shape[1:])
        loss = classification_loss(output, target)
        return loss, correct

    def _regression_vote(self, output, target):
        """ensemble the outputs from regressors."""
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1],
                                            self._ensemble_size,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def _auc_score(self, inliers, outliers):
        y_true = np.array([0] * len(inliers) + [1] * len(outliers))
        y_score = np.concatenate([inliers, outliers])
        return roc_auc_score(y_true, y_score)
    
    def _predict_dataset(self, testset):
        cls = len(testset.dataset.dataset.classes)
        model_outputs = torch.zeros(
            self._ensemble_size,
            len(testset.dataset),
            cls)
        for batch, (data, target) in enumerate(testset):
            data = data.to(alf.get_default_device())
            target = target.to(alf.get_default_device())
            output, _ = self._net(data)
            if self._ensemble_size == 1:
                output = output.unsqueeze(1)
            output = output.transpose(0, 1)
            model_outputs[:, batch*len(data): (batch+1)*len(data), :] = output
        return model_outputs

    def eval_uncertainty(self):
        # Soft voting for now
        with torch.no_grad():
            outputs = self._predict_dataset(self._test_loader)
        probs = F.softmax(outputs, -1).mean(0)
        entropy = entropy_fn(probs.T.cpu().detach().numpy())
        with torch.no_grad():
            outputs_outlier = self._predict_dataset(self._outlier_test)
        probs_outlier = F.softmax(outputs_outlier, -1).mean(0)
        entropy_outlier = entropy_fn(probs_outlier.T.cpu().detach().numpy())
        auroc_entropy = self._auc_score(entropy, entropy_outlier)
        logging.info("AUROC score: {}".format(auroc_entropy))

    def summarize_train(self, loss_info, params, cum_loss=None, avg_acc=None):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.

        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._config.debug_summaries:
            summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/neglogprob', data=cum_loss)
        if avg_acc is not None:
            alf.summary.scalar(name='train_epoch/avg_acc', data=avg_acc)
     
    def _approx_jacobian_trace(self, fx, x):
        """Hutchinson's trace Jacobian estimator O(1) call to autograd,
            used by "\"minmax\" method"""
        eps = torch.randn_like(fx)
        jvp = torch.autograd.grad(
                fx,
                x,
                grad_outputs=eps,
                retain_graph=True,
                create_graph=True)[0]
        if eps.shape[-1] == jvp.shape[-1]:
            tr_jvp = torch.einsum('bi,bi->b', jvp, eps)
        else:
            tr_jvp = torch.einsum('bi,bj->b', jvp, eps)
        return tr_jvp
    """
    def _minmax_critic_grad(self, net_outputs, critic_outputs, loss_func):
        
        loss_inputs = net_outputs
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), net_outputs)[0]  # [N, D]
        log_p_f = (loss_grad * critic_outputs).sum(1) # [N, D]
        tr_critic = self._approx_jacobian_trace(critic_outputs, net_outputs) # [N]
        lamb = 10
        stein_pq = log_p_f - tr_critic.unsqueeze(1) # [n x 1]
        l2_penalty = (critic_outputs * critic_outputs).sum(1).mean() * lamb
        adv_grad =  -1 * stein_pq.mean() + l2_penalty
        loss_propagated = adv_grad
        
        return loss, loss_propagated
    """
    def _minmax_critic_grad(self, net_outputs, critic_outputs, loss_func):
        """update direction \phi^*(x) for minmax amortized svgd"""
        
        loss_inputs = net_outputs
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), net_outputs)[0]  # [N, D]
        log_p_f = (loss_grad * critic_outputs).sum(1) # [N, D]
        tr_critic = self._approx_jacobian_trace(critic_outputs, net_outputs) # [N]
        lamb = 10
        stein_pq = log_p_f + tr_critic.unsqueeze(1) # [n x 1]
        l2_penalty = (critic_outputs * critic_outputs).sum(1).mean() * lamb
        adv_grad =  stein_pq.mean() - l2_penalty
        loss_propagated = -adv_grad
        
        return loss, loss_propagated
    
