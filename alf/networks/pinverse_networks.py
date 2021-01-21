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
"""DynamicsNetwork"""

import gin
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec

from .network import Network
from .encoding_networks import EncodingNetwork
from .projection_networks import NormalProjectionNetwork


@gin.configurable
class PinverseNetwork(Network):
    """Create an instance of PinverseNetwork."""

    def __init__(self,
                 input_tensor_spec,
                 output_dim,
                 hidden_size,
                 joint_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="PinverseNetwork"):
        r"""Creates an instance of `PinverseNetwork` for predicting 
        :math:`x=J^{-1}*eps` given eps for the purpose of optimizing a 
        downstream objective :math:`Jx - eps = 0`. 

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            output_dim (int): total output dimension of the pinverse net, will differ
                between ``svgd3`` and ``minmax`` methods
            hidden_size (int): base hidden width for pinverse net
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            name (str):
        """
        super().__init__(input_tensor_spec, name=name)

        z_spec, eps_spec = input_tensor_spec
        self._z_dim = z_spec.shape[0]
        self._eps_dim = eps_spec.shape[0]
        self._output_dim = output_dim
        self._hidden_size = hidden_size
        self._activation = activation
        assert self._z_dim <= output_dim and (
            self._eps_dim == output_dim or self._eps_dim == self._z_dim), (
                "input_tensor_spec and output_dim does not match!")
        if self._eps_dim == output_dim:
            self._fullrank = True
        else:
            self._fullrank = False

        # if kernel_initializer is None:
        #     kernel_initializer = functools.partial(
        #         variance_scaling_init,
        #         gain=1.0 / 2.0,
        #         mode='fan_in',
        #         distribution='truncated_normal',
        #         nonlinearity=math_ops.identity)

        self._z_encoder = torch.nn.Linear(self._z_dim, hidden_size)
        self._eps_encoder = torch.nn.Linear(self._eps_dim, hidden_size)
        if joint_fc_layer_params is None:
            joint_fc_layer_params = (hidden_size * 2, hidden_size * 2)
        self._joint_encoder = EncodingNetwork(
            TensorSpec(shape=(hidden_size * 2, )),
            fc_layer_params=joint_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=output_dim,
            last_activation=math_ops.identity)

    def forward(self, inputs, state=()):
        """Computes prediction given inputs.

        Args:
            inputs:  A tuple of Tensors consistent with (z, eps) 
                z (torch.tensor): size [B', K] or [B', D], represents z' quantity 
                    in the ``svgd3`` case, where K is the input dimension to the 
                    generator, which is less or equal to the output dimension D.
                eps (torch.tensor):  size [B', B, K] or [B', B, D] for ``svgd3``, 
                    [B, K, D] for ``minmax``. K equals D when the density transform
                    function is fullrank, e.g., :math:`x = f(z) + z`, otherwise 
                    K is necessarily less than D.
            state: empty for API consistency

        Returns:
            out (torch.Tensor): of size [B2, B, D] for ``svgd3`` method, or
                [B, K, D] for ``minmax``.
            state: empty
        """
        z, eps = inputs
        assert z.ndim == 2 and z.shape[-1] >= self._z_dim, (
            "the input z has wrong shape!")
        assert eps.ndim == 3 and eps.shape[-1] == self._eps_dim, (
            "the input eps has wrong shape!")
        batch_size_z = z.shape[0]
        assert eps.shape[0] == batch_size_z, (
            "batch sizes of input z and eps do not match!")

        if z.shape[-1] > self._z_dim:
            z = z[:, :self._z_dim]
        z = torch.repeat_interleave(z, eps.shape[1], dim=0)
        eps = eps.reshape(batch_size_z * eps.shape[1], -1)

        encoded_z = self._activation(self._z_encoder(z))
        encoded_eps = self._activation(self._eps_encoder(eps))
        joint = torch.cat([encoded_z, encoded_eps], -1)
        out, _ = self._joint_encoder(joint)
        out = out.reshape(batch_size_z, -1, self._output_dim)

        return out, state
