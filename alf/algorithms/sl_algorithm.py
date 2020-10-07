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
from alf.networks import Network

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


@gin.configurable
class SLAlgorithm(Algorithm):
    """Base class for supervised learning algorithms.

    ``SLAlgorithm`` is meant to provide the basic and essential functions for 
    training a predictor, or an ensemble of predictors on a fixed dataset. 

    Most of the key functions are inherited from the ``Algorithm`` base class. 
    But are further expanded for use in the supervised learning context. 
    The key interface methods are defined as follows:

    1. ``train_iter()`` executes a single epoch of training, performing one
        pass over the dataset. 
    2. ``train_step()`` performs one step of evaluation, with respect to 
        a single batch of data
    3. ``evaluate()`` produces an aggregate score such as loss value or
        accuracy, for the entire testing dataset. 
    4. ``predict_dataset()`` returns model outputs given an input dataset. 
        ``predict_dataset()`` is useful when no gradient updates need to
        be performed, yet model predictions are needed without computing
        statistics like loss or classification, as in ``evaluate``. 
    5. ``evaluate_uncertainty()`` calcuates the ability of the model to 
        differentiate between two given datasets, as given by the Area Under
        the Reciever Operating Curve (AUCROC) score. The AUROC score is 
        used as a default statistic, and is computed from the predictive
        entropies w.r.t an inlier dataset and an outlier dataset. 
        But users can choose to implement their own uncertainty quantification
        methods. 

    ``SLAlgorithm`` also can also be used to instantiate and train additional 
        modules required by other algorithms. Right now we explicitly support
        training of the critic network used by the particle VI method Minmax
        Amortized SVGD with the ``minmax_critic_grad()`` function. 
    """

    def __init__(self,
                 input_tensor_spec=(),
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 net=None,
                 use_fc_bn=False,
                 predictor_size=1,
                 predictor_vote=None,
                 loss_type="classification",
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
                 name="SLAlgorithm"):
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
            net (Network): a default predictor, if None is provided then 
                a new one is constructed. 
            use_fn_bn (bool): whether to use Batch Norm on the fc layers
            predictor_size (int): how many independant models to train 
                concurrently in an ensemble
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

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the training network,
                types are [``classification``, ``regression``]
            predictor_vote (str): types of voting to make predictions from 
                ensemble. if ``predictor_size`` > 1, then voting is used.
                Acceptable voting types are [``soft``, ``hard``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        
        assert (predictor_size >= 1 and isinstance(predictor_size, int)), " "\
            "``predictor_size`` must be an int greater than or equal to 1 " \
            "got {}".format(predictor_size)

        if last_layer_param is not None and isinstance(last_layer_param, tuple):
            last_layer_param = last_layer_param[0]
        
        if net is None:
            if predictor_size > 1:
                net = ParallelEncodingNetwork(
                    input_tensor_spec,
                    predictor_size,
                    conv_layer_params=conv_layer_params,
                    fc_layer_params=fc_layer_params,
                    use_fc_bn=use_fc_bn,
                    activation=activation,
                    last_layer_size=last_layer_param,
                    last_activation=last_activation,
                    name='Network')
                logging.info("Initialized ensemble of {} models".format(
                    predictor_size))
            else:
                net = EncodingNetwork(
                    input_tensor_spec,
                    conv_layer_params=conv_layer_params,
                    fc_layer_params=fc_layer_params,
                    use_fc_bn=use_fc_bn,
                    activation=activation,
                    last_layer_size=last_layer_param,
                    last_activation=last_activation,
                    last_use_fc_bn=use_fc_bn,
                    name='Network')
                logging.info("Initialized single model")
        
        if logging_network:
            logger.info(net)

        self._net = net
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._predictor_size = predictor_size
        assert predictor_vote in ['soft', 'hard', None], "voting only supports "\
            "\"soft\", \"hard\", None"
        
        self._predictor_vote = predictor_vote
        if loss_type == "classification":
            self._loss_func = self._classification_loss
            self._vote = self._classification_vote
        elif loss_type == "regression":
            self._loss_func = self._regression_loss
            self._vote = self._regression_vote
        else:
            raise ValueError("Unsupported loss_type: {}".format(loss_type))
        self._loss_type = loss_type
        self._logging_training=logging_training
        self._logging_evaluate=logging_evaluate
        self._config = config

    def set_data_loader(self, train_loader, test_loader=None, outlier=None):
        """ Set data loaders for training, testing, and uncertainty
            quantification 
        """
        self._train_loader = train_loader
        self._test_loader = test_loader
        if outlier is not None:
            assert isinstance(outlier, tuple), "outlier dataset must be " \
                "provided in the format (outlier_train, outlier_test)"
            self._outlier_train = outlier[0]
            self._outlier_test = outlier[1]
        else: 
            self._outlier_train = self._outlier_test = None
    
    def train_iter(self, state=None):
        """ Perform a single (iteration) epoch of training"""
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
                if hasattr(loss_info.extra, 'generator'):
                    loss += loss_info.extra.generator.loss
                else:
                    loss += loss_info.loss
                if self._loss_type == 'classification':
                    if hasattr(alg_step.info.extra, 'generator'):
                        avg_acc.append(alg_step.info.extra.generator.extra)
                    else:
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
        loss, outputs = self._neglogprob(inputs)
        return AlgStep(
            output=outputs,
            state=(),
            info=LossInfo(
                loss=loss.loss,
                extra=loss.extra))
    
    def evaluate(self):
        """Evaluate the network/ensemeble on the test dataset. """
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
                if self._predictor_size == 1:
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
        predictors = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._predictor_vote == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._predictor_vote == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = pred.mode(1)[0]  # [B, 1]
        else:
            vote = probs.argmax(-1).cpu()
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1],
                                            predictors,
                                            *target.shape[1:])
        loss = self._classification_loss(output, target)
        return loss, correct

    def _regression_vote(self, output, target):
        """ensemble the outputs from regressors."""
        predictors = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1],
                                            predictors,
                                            *target.shape[1:])
        total_loss = self._regression_loss(output, target)
        return loss, total_loss
    
    def _classification_loss(self, output, target):
        if output.dim() == 2:
            output = output.reshape(target.shape[0], target.shape[1], -1)
        pred = output.max(-1)[1]
        acc = pred.eq(target).float().mean(0)
        avg_acc = acc.mean()
        loss = F.cross_entropy(output.transpose(1, 2), target)
        return NetworkLossInfo(loss=loss, extra=avg_acc)

    def _regression_loss(self, output, target):
        out_shape = output.shape[-1]
        assert (target.shape[-1] == out_shape), (
            "feature dimension of output and target does not match.")
        loss = 0.5 * F.mse_loss(
            output.reshape(-1, out_shape),
            target.reshape(-1, out_shape),
            reduction='sum')
        return NetworkLossInfo(loss=loss, extra=())

    def _neglogprob(self, inputs):
        """
        Computes the negative log probability for network outputs.
        
        Args: 
            inputs (torch.tensor): (data, target) of training batch.

        Returns:
            negative log_prob for params evaluated on current training batch.
        """
        data, target = inputs
        output, _ = self._net(data)  # [B, N, D]
        if output.dim() == 2:
            output = output.unsqueeze(1)
        target = target.unsqueeze(1).expand(*target.shape[:1],
                                            self._predictor_size,
                                            *target.shape[1:])
        return self._loss_func(output, target), output

    def _auc_score(self, inliers, outliers):
        """
        Computes the AUROC score w.r.t network outputs. the ROC (curve) plots
        true positive rate against false positive rate. Thus the area under
        this curve gives the degree of separability between two dataset. 
        An AUROC score of 1.0 means that the classifier means that the
        classifier can totally discriminate between the two input datasets
        
        Args: 
            inliers (np.array): set of predictions on inlier (training) data
            outliers (np.array): set of predictions on outlier data
        
        Returns:
            AUROC score 
        """
        y_true = np.array([0] * len(inliers) + [1] * len(outliers))
        y_score = np.concatenate([inliers, outliers])
        return roc_auc_score(y_true, y_score)
    
    def _predict_dataset(self, testset, predictor_size=None):
        """
        Computes predictions for an input dataset. 

        Args: 
            testset (iterable): dataset for which to get predictions
            predictor_size (int): optional parameter indicating how many 
                predictors are used in an ensemble. Useful for nonstandard
                implementations.
        Returns:
            model_outputs (torch.tensor): a tensor of shape [N, S, D] where
            N refers to the number of predictors, S is the number of data
            points, and D is the output dimensionality. 
        """
        if hasattr(testset.dataset, 'dataset'):
            cls = len(testset.dataset.dataset.classes)
        else:
            cls = len(testset.dataset.classes)
        outputs = []
        for batch, (data, target) in enumerate(testset):
            data = data.to(alf.get_default_device())
            target = target.to(alf.get_default_device())
            output, _ = self._net(data)
            if output.dim() == 2:
                output = output.unsqueeze(1)
            output = output.transpose(0, 1)
            outputs.append(output)
        model_outputs = torch.cat(outputs, dim=1)  # [N, B, D]
        return model_outputs

    def eval_uncertainty(self):
        """ Utility function to compute the ability of the model to capture
            epistemic uncertainty. By default this function computes the 
            predictive entropy across a dataset of inlier samples, as well
            as the predictive entropy for a dataset of outlier samples. To 
            measure separability, the AUROC score is computed.

        Args:
            None

        Returns:
            AUROC score
        """
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
        return auroc_entropy

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
            used by "\"minmax\" critic training"""
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
    
    def _critic_train_step(self,
                           inputs,
                           loss_func,
                           transform_func=None,
                           entropy_regularization=1.,
                           state=()):
        if transform_func is not None:
            inputs, density_outputs = transform_func(inputs)
            n_perturbed = density_outputs.shape[1]
            density_outputs = torch.cat((
                inputs[:, :-n_perturbed], density_outputs), dim=-1)
            outputs, _ = self._net(density_outputs)
            density_outputs = None
        else:
            density_outputs = None
            outputs, _ = self._net(inputs)
        
        critic_loss, loss_propagated = self._minmax_critic_grad(
            inputs, outputs, loss_func, entropy_regularization, density_outputs)
        
        self.update_with_gradient(LossInfo(loss=loss_propagated))

    def _minmax_critic_grad(self,
                            net_outputs,
                            critic_outputs,
                            loss_func,
                            entropy_regularization,
                            density_outputs=None):
        """update direction \phi^*(x) for minmax amortized svgd"""
        loss_inputs = net_outputs
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), net_outputs)[0]  # [N, D]
        neglogp_f = (loss_grad.detach() * critic_outputs).sum(1) # [N]
        if density_outputs is not None:
            net_outputs = density_outputs
        
        tr_critic = self._approx_jacobian_trace(critic_outputs, net_outputs) # [N]
        lamb = 10.
        stein_pq = neglogp_f - entropy_regularization * tr_critic#.unsqueeze(1) # [n x 1]
        l2_penalty = (critic_outputs * critic_outputs).sum(1).mean() * lamb
        loss_propagated = stein_pq.mean() + l2_penalty
        
        return loss, loss_propagated
    
