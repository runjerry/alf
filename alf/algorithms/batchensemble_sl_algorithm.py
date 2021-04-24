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
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

import alf
from alf import layers, networks
from alf.algorithms.config import TrainerConfig
from alf.algorithms.algorithm import Algorithm, LossInfo
from alf.algorithms.sl_algorithm import SLAlgorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.networks.batchensemble_networks import BatchEnsembleNetwork
from alf.networks.networks import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.sl_utils import classification_loss, regression_loss, auc_score
from alf.utils.sl_utils import predict_dataset
from alf.utils.summary_utils import record_time


@alf.configurable
class BatchEnsembleSLAlgorithm(SLAlgorithm):
    def __init__(self,
                 train_loader=None,
                 test_loader=None,
                 outlier_loaders=None,
                 input_tensor_spec=None,
                 output_dim=None,
                 ensemble_size=10,
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
                 full_ensemble_train=False,
                 entropy_regularization=1.,
                 parvi=None,
                 sl_type="classification",
                 voting="soft",
                 config: TrainerConfig = None,
                 logging_training=False,
                 logging_evaluate=False,
                 debug_summaries=False,
                 name="BatchEnsembleSLAlgorithm"):
        """
        Args:
            data_creator (Callable): called as ``data_creator()`` to get a tuple
                of ``(train_dataloader, test_dataloader)``
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
            entropy_regularization (float): weight for par_vi repulsive term. 
            parvi (string): if not ``None``, paramters with attribute
                ``ensemble_group`` will be updated by particle-based vi algorithm
                specified by ``parvi``, options are [``svgd``, ``gfsf``],

                * Stein Variational Gradient Descent (SVGD)

                  Liu, Qiang, and Dilin Wang. "Stein Variational Gradient Descent: 
                  A General Purpose Bayesian Inference Algorithm." NIPS. 2016.

                * Wasserstein Gradient Flow with Smoothed Functions (GFSF)
                
                  Liu, Chang, et al. "Understanding and accelerating particle-based 
                  variational inference." ICML. 2019.

            sl_type (str): loglikelihood type for the supervised learning task,
                types are [``classification``, ``regression``]
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            config (TrainerConfig): configuration for training
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            debug_summaries (bool): True if debug summaries should be created.
        """
        assert train_loader is not None and test_loader is not None, (
            "train_loader and test_loader need to be provided")

        input_tensor_spec = TensorSpec(shape=train_loader.dataset[0][0].shape)
        if sl_type == 'classification':
            if hasattr(train_loader.dataset, 'classes'):
                output_dim = len(train_loader.dataset.classes)
            else:
                assert output_dim is not None, (
                    "output_dim needs to be specified if trainset.dataset does "
                    "not have attribute 'classes'.")
        elif sl_type == 'regression':
            output_dim = train_loader[1].shape[-1]

        if optimizer is None:
            assert parvi is not None, (
                "parvi needs to be provided if optimizer is None.")
            batch_size = train_loader.batch_size
            entropy_regularization = batch_size / len(train_loader.dataset)
            optimizer = alf.optimizers.Adam(
                lr=1e-3,
                weight_decay=1e-4,
                parvi=parvi,
                repulsive_weight=entropy_regularization)
        self._full_ensemble_train = full_ensemble_train

        if net is None:
            net = BatchEnsembleNetwork(
                input_tensor_spec=input_tensor_spec,
                ensemble_size=ensemble_size,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                use_fc_bn=use_fc_bn,
                last_layer_size=output_dim,
                last_activation=last_activation,
                last_kernel_initializer=last_kernel_initializer,
                last_use_fc_bn=last_use_fc_bn)

        super().__init__(
            train_loader=train_loader,
            test_loader=test_loader,
            outlier_loaders=outlier_loaders,
            input_tensor_spec=input_tensor_spec,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
            net=net,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            last_activation=last_activation,
            last_kernel_initializer=last_kernel_initializer,
            last_use_fc_bn=last_use_fc_bn,
            optimizer=optimizer,
            sl_type=sl_type,
            voting=voting,
            config=config,
            logging_training=logging_training,
            logging_evaluate=logging_evaluate,
            debug_summaries=debug_summaries,
            name=name)

    def predict_step(self, inputs, state=None, full_ensemble=False):
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
        outputs, _ = self._net(inputs, full_ensemble=full_ensemble)
        return AlgStep(output=outputs, state=(), info=())

    def train_step(self, inputs, state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            state (None): not used

        Returns:
            ``train_step`` of ``self._generator``
        """
        data, targets = inputs
        outputs = self.predict_step(
            data, full_ensemble=self._full_ensemble_train).output
        loss_info = self._loss_func(outputs, targets)

        return AlgStep(output=outputs, state=(), info=loss_info)

    def eval_func(self, inputs):
        return self._net(inputs, full_ensemble=True)
