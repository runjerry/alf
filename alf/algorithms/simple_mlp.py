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

import gin
import torch

import alf
from alf.layers import FC
from alf.networks import Network
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import identity


@gin.configurable
class SimpleMLP(Network):
    """Creates an instance of ``SimpleMLP`` with one bottleneck layer.
    """

    def __init__(self,
                 input_tensor_spec,
                 hidden_layer_size=64,
                 activation=torch.relu_,
                 name="SimpleMLP"):
        r"""Create a SimpleMLP.

        Args:
            input_tensor_spec (TensorSpec):
            hidden_layer_size (int):
            activation (nn.functional):
            name (str):
        """
        assert len(input_tensor_spec.shape) == 1, \
            ("The input shape {} should be a 1-d vector!".format(
                input_tensor_spec.shape))

        super().__init__(input_tensor_spec, name=name)

        self._input_size = input_tensor_spec.shape[0]
        self._output_size = self._input_size
        self._hidden_layer_size = hidden_layer_size
        self._encoder = FC(
            input_size=self._input_size,
            output_size=hidden_layer_size,
            activation=identity,
            use_bias=False,
            use_bn=False)
        self._decoder = FC(
            input_size=hidden_layer_size,
            output_size=self._output_size,
            activation=identity,
            use_bias=False,
            use_bn=False)
        self._hidden_activation = activation

    def forward(self, inputs, state=(), requires_ntk=False):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
            requires_ntk (bool): whether compute ntk
        """
        self._encodes = self._encoder(inputs)
        self._hidden_neurons = self._hidden_activation(self._encodes)
        outputs = self._decoder(self._hidden_neurons)
        if requires_ntk:
            outputs = outputs, self._compute_ntk(inputs)
        return outputs, state

    @property
    def encodes(self):
        return self._encodes

    @property
    def hidden_neurons(self):
        return self._hidden_neurons

    def _compute_ntk(self, inputs):
        """Compute ntk in closed-form. """

        ntk = torch.pow(self.hidden_neurons.norm(), 2) * torch.eye(
            self._output_size)
        pos_idx = (self.encodes > 0).nonzero().squeeze()
        Dweight = self._decoder.weight.data[:, pos_idx]
        if Dweight.ndim == 1:
            Dweight = Dweight.unsqueeze(-1)
        inputs_norm2 = torch.pow(inputs.norm(), 2)
        ntk = ntk + inputs_norm2 * Dweight @ Dweight.t()

        return ntk

    def ntk_vec_prod(self, inputs, vec):
        """ 
        Compute ntk vector product in closed-form. 
        No need to explicitly compute the large ntk matrix first. 
        """
        vec = vec.squeeze()
        assert vec.ndim == 1 and len(vec) == self._input_size, \
            ("vec should has shape {}!".format(self._input_size))

        bottleneck_norm2 = torch.pow(self.hidden_neurons.norm(), 2)
        pos_idx = (self.encodes > 0).nonzero().squeeze()
        Dweight = self._decoder.weight.data[:, pos_idx]
        if Dweight.ndim == 1:
            Dweight = Dweight.unsqueeze(-1)
        inputs_norm2 = torch.pow(inputs.norm(), 2)

        ntk_vec = inputs_norm2 * Dweight @ (Dweight.t() @ vec)

        return ntk_vec + bottleneck_norm2 * vec
