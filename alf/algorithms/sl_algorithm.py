# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""An example of ParVI training of a BatchEnsemble model on open category MNIST.

You can use the following command to run it:

.. code-block: batch

    python -m alf.bin.train --conf svgd_be_mnist_ood_conf --root_dir ~/tmp/mnist_ood/svgd_be/

Note: You need to first install torchtext using ``pip install torchtext``

"""

from absl import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

import alf
from alf import layers, networks
from alf.algorithms.config import TrainerConfig
from alf.algorithms.algorithm import Algorithm, LossInfo
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.networks.batchensemble_networks import BatchEnsembleNetwork
from alf.networks.encoding_networks import EncodingNetwork
from alf.networks.networks import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.sl_utils import classification_loss, regression_loss, auc_score
from alf.utils.sl_utils import predict_dataset
from alf.utils.summary_utils import record_time


@alf.configurable
class SLAlgorithm(Algorithm):
    def __init__(self,
                 data_creator=None,
                 data_creator_outlier=None,
                 train_loader=None,
                 test_loader=None,
                 outlier_loaders=None,
                 input_tensor_spec=None,
                 output_dim=None,
                 ensemble_size=None,
                 net: Network = None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 last_activation=alf.math.identity,
                 last_kernel_initializer=None,
                 last_use_fc_bn=False,
                 optimizer=None,
                 sl_type="classification",
                 voting="soft",
                 config: TrainerConfig = None,
                 logging_training=False,
                 logging_evaluate=False,
                 debug_summaries=False,
                 name="SLAlgorithm"):
        """
        Args:
            data_creator (Callable): called as ``data_creator()`` to get a tuple
                of ``(train_dataloader, test_dataloader)``
            data_creator_outlier (Callable): called as ``data_creator()`` to get
                a tuple of ``(outlier_train_dataloader, outlier_test_dataloader)``
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None. It must be provided if ``data_creator`` is not provided.
            output_dim (int): dimension of the output of the generated network.
                It must be provided if ``data_creator`` is not provided.
            ensemble_size (int): ensemble size
            net (Network): network for performing the supervised learning task. 
                If None, a default one with conv_layer_params and fc_layer_params
                will be created.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If None, a variance_scaling_initializer will be
                used.
            use_fc_bn (bool): whether use Batch Normalization for fc layers.
            last_activation (nn.functional): activation function of the last layer.
            last_kernel_initializer (Callable): initializer for the the
                additional layer specified by ``last_layer_size``.
                If None, it will be the same with ``kernel_initializer``. If
                ``last_layer_size`` is None, ``last_kernel_initializer`` will
                not be used.
            last_use_fc_bn (bool): whether use bias for the last layer
            optimizer (alf.Optimizer): The optimizer for training the net.
            sl_type (str): loglikelihood type for the supervised learning task,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            config (TrainerConfig): configuration for training
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            debug_summaries (bool): True if debug summaries should be created.
        """
        if train_loader is None or test_loader is None:
            assert data_creator is not None, (
                "Either train_loader + test_loader or data_creator need to be privided"
            )
            train_loader, test_loader = data_creator()
            if data_creator_outlier is not None:
                outlier_loaders = data_creator_outlier()
            else:
                outlier_loaders = None
        self.set_data_loader(train_loader, test_loader, outlier_loaders)

        super().__init__(
            optimizer=optimizer,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if input_tensor_spec is None:
            input_tensor_spec = TensorSpec(
                shape=train_loader.dataset[0][0].shape)
        if output_dim is None:
            if sl_type == 'classification':
                if hasattr(train_loader.dataset, 'classes'):
                    output_dim = len(train_loader.dataset.classes)
                else:
                    assert output_dim is not None, (
                        "output_dim needs to be specified if trainset.dataset does "
                        "not have attribute 'classes'.")
            elif sl_type == 'regression':
                output_dim = train_loader[1].shape[-1]

        self._input_tensor_spec = input_tensor_spec
        if net is None:
            net = EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                use_fc_bn=use_fc_bn,
                last_layer_size=output_dim,
                last_activation=last_activation,
                last_kernel_initializer=last_kernel_initializer,
                last_use_fc_bn=last_use_fc_bn)
            ensemble_size = 1
        else:
            assert ensemble_size is not None, (
                "If net is provided, ensemble_size needs to be provided too.")

        self._net = net
        self._ensemble_size = ensemble_size

        self._sl_type = sl_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        assert (voting in ['soft',
                           'hard']), ('voting only supports "soft" and "hard"')
        self._voting = voting
        if sl_type == 'classification':
            self._loss_func = classification_loss
            self._vote = self._classification_vote
        elif sl_type == 'regression':
            self._loss_func = regression_loss
            self._vote = self._regression_vote
        else:
            raise ValueError("Unsupported sl_type: %s" % sl_type)

    def set_data_loader(self,
                        train_loader,
                        test_loader=None,
                        outlier_data_loaders=None):
        """Set data loadder for training and testing.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            test_loader (torch.utils.data.DataLoader): testing data loader
            outlier_data_loaders (tuple[torch.utils.data.DataLoader): 
                (trainloader, testloader) for outlier datasets
        """
        self._train_loader = train_loader
        self._test_loader = test_loader

        if outlier_data_loaders is not None:
            assert isinstance(outlier_data_loaders, tuple), "outlier dataset "\
                "must be provided in the format (outlier_train, outlier_test)"
            self._outlier_train_loader = outlier_data_loaders[0]
            self._outlier_test_loader = outlier_data_loaders[1]
        else:
            self._outlier_train_loader = None
            self._outlier_test_loader = None

    def predict_step(self, inputs, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.

        Args:
            inputs (Tensor): inputs to the ensemble of networks. Its shape 
                can be ``[B, n, ...]`` or ``[B, ...]``.
            state (None): not used.

        Returns:
            AlgStep:
            - output (Tensor): predictions with shape ``[B, n, ...]``.
            - state (None): not used
        """
        outputs, _ = self._net(inputs)
        return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, state=None):
        """Perform one epoch (iteration) of training.

        Args:
            state (None): not used

        Return:
            mini_batch number
        """

        assert self._train_loader is not None, "Must set data_loader first."
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            if self._sl_type == 'classification':
                avg_acc = []
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target), state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                loss += loss_info.loss
                if self._sl_type == 'classification':
                    avg_acc.append(alg_step.info.extra)
        acc = None
        if self._sl_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._sl_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
        self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)
        return batch_idx + 1

    def train_step(self, inputs, state=None):
        """Perform one batch of training computation.

        Args:
            inputs (tuple of Tensors): training data tuple (data, targets). 
                data shape can be ``[B, n, ...]`` or ``[B, ...]``.
                targets shape can be ``[B]`` or ``[B, n]``.
            state (None): not used

        Returns:
            ``train_step`` of ``self._generator``
        """
        data, targets = inputs
        outer_rank = alf.nest.utils.get_outer_rank(data,
                                                   self._input_tensor_spec)
        assert 1 <= outer_rank <= 2, ("inputs should have shape [B, %d, ...] "
                                      " or [B, ...]" % self._ensemble_size)
        if outer_rank == 2:
            assert data.shape[1] == self._ensemble_size, (
                "inputs replica "
                "does not match ensemble_size %d" % self._ensemble_size)
        if outer_rank == 1 and self._ensemble_size > 1:
            targets = targets.unsqueeze(1).expand(targets.shape[0],
                                                  self._ensemble_size)
        outputs = self.predict_step(data).output
        loss_info = self._loss_func(outputs, targets)

        return AlgStep(output=outputs, state=(), info=loss_info)

    def _classification_vote(self, output, target):
        """Ensemble the outputs from sampled classifiers."""
        ensemble_size = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = []
            for i in range(pred.shape[0]):
                values, counts = torch.unique(
                    pred[i], sorted=False, return_counts=True)
                modes = (counts == counts.max()).nonzero()
                label = values[torch.randint(len(modes), (1, ))]
                vote.append(label)
            vote = torch.as_tensor(vote, device='cpu')
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], ensemble_size,
                                            *target.shape[1:])
        loss = classification_loss(output, target)
        return loss, correct

    def _regression_vote(self, output, target):
        """Ensemble the outputs for sampled regressors."""
        ensemble_size = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], ensemble_size,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def eval_func(self, inputs):
        return self._net(inputs)

    def evaluate(self):
        """Evaluatation on test dataset.
        self._test_loader needs to be specified first.
        """
        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin evaluating")
        with record_time("time/test"):
            if self._sl_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self.eval_func(data)  # [B, N, D]
                if self._ensemble_size == 1:
                    loss_info = self._loss_func(output, target)
                    loss = loss_info.loss
                    extra = loss_info.extra
                else:
                    loss, extra = self._vote(output, target)
                if self._sl_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.loss.item()

        if self._sl_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            alf.summary.scalar(name='eval/test_acc', data=test_acc * 100)
        if self._logging_evaluate:
            if self._sl_type == 'classification':
                logging.info("Test acc: {}".format(test_acc * 100))
            logging.info("Test loss: {}".format(test_loss))
        alf.summary.scalar(name='eval/test_loss', data=test_loss)

    def eval_uncertainty(self):
        """Function to evaluate the epistemic uncertainty of a sampled ensemble.
            This method computes the following metrics:
        
        * AUROC (AUC): AUC is computed with respect to the entropy in the 
          averaged softmax probabilities, as well as the sum of the
          variance of the softmax probabilities over the ensemble. 

        Returns:
            auroc_entropy (float): auroc of inlier-outlier predictive entropy
            auroc_variance (float): auroc of inlier-outlier predictice variance
        """
        assert self._test_loader is not None, "Must set test_loader first."
        assert self._outlier_test_loader is not None, (
            "Must set outlier_test_loader first.")

        with torch.no_grad():
            outputs = predict_dataset(self.eval_func, self._test_loader)
            outputs_outlier = predict_dataset(self.eval_func,
                                              self._outlier_test_loader)

        probs = F.softmax(outputs, -1)
        probs_outlier = F.softmax(outputs_outlier, -1)

        mean_probs = probs.mean(0)
        mean_probs_outlier = probs_outlier.mean(0)

        entropy = torch.distributions.Categorical(mean_probs).entropy()
        entropy_outlier = torch.distributions.Categorical(
            mean_probs_outlier).entropy()

        variance = F.softmax(outputs, -1).var(0).sum(-1)
        variance_outlier = F.softmax(outputs_outlier, -1).var(0).sum(-1)

        auroc_entropy = auc_score(entropy, entropy_outlier)
        auroc_variance = auc_score(variance, variance_outlier)
        logging.info("AUROC score (entropy): {}".format(auroc_entropy))
        logging.info("AUROC score (variance): {}".format(auroc_variance))
        alf.summary.scalar(name='eval/auroc_entropy', data=auroc_entropy)
        alf.summary.scalar(name='eval/auroc_variance', data=auroc_variance)
        return auroc_entropy, auroc_variance

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
            loss_info (LossInfo): loss. 
            params (list[Parameter]): list of parameters with gradients. 
            cum_loss (float): cumulative training loss of epoch. 
            avg_acc (float): average accuracy across batches in epoch. 
        """
        if self._config is not None:
            if self._config.summarize_grads_and_vars:
                summary_utils.summarize_variables(params)
                summary_utils.summarize_gradients(params)
            if self._config.debug_summaries:
                summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/neglogprob', data=cum_loss)
        if avg_acc is not None:
            alf.summary.scalar(name='train_epoch/avg_acc', data=avg_acc)
