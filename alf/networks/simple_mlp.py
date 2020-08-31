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
                 initializer=None,
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
            kernel_initializer=initializer,
            use_bias=False,
            use_bn=False)
        self._decoder = FC(
            input_size=hidden_layer_size,
            output_size=self._output_size,
            activation=identity,
            kernel_initializer=initializer,
            use_bias=False,
            use_bn=False)
        self._hidden_activation = activation

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
            requires_ntk (bool): whether compute ntk
        """
        self._encodes = self._encoder(inputs)
        self._hidden_neurons = self._hidden_activation(self._encodes)
        outputs = self._decoder(self._hidden_neurons)
        return outputs, state

    @property
    def hidden_neurons(self):
        return self._hidden_neurons

    def compute_ntk(self, inputs1, inputs2, hidden_neurons1, hidden_neurons2):
        """Compute ntk in closed-form. """

        inputs1 = inputs1.squeeze()
        inputs2 = inputs2.squeeze()
        hidden_neurons1 = hidden_neurons1.squeeze()
        hidden_neurons2 = hidden_neurons2.squeeze()

        assert inputs1.ndim == 1 and len(inputs1) == self._input_size, \
            ("inputs1 should has shape {}!".format(self._input_size))

        assert inputs2.ndim == 1 and len(inputs2) == self._input_size, \
            ("inputs2 should has shape {}!".format(self._input_size))

        assert (hidden_neurons1.ndim == 1) and (
            len(hidden_neurons1) == self._hidden_layer_size), \
            ("hidden_neurons1 should has shape {}!".format(self._hidden_layer_size))

        assert (hidden_neurons2.ndim == 1) and (
            len(hidden_neurons2) == self._hidden_layer_size), \
            ("hidden_neurons2 should has shape {}!".format(self._hidden_layer_size))

        ntk = torch.dot(hidden_neurons1, hidden_neurons2) * torch.eye(
            self._output_size)
        mask1 = hidden_neurons1 > 0
        mask2 = hidden_neurons2 > 0
        D1 = self._decoder.weight.data * mask1
        D2 = self._decoder.weight.data * mask2
        inputs_norm2 = torch.dot(inputs1, inputs2)
        ntk = ntk + inputs_norm2 * torch.matmul(D1, D2.t())

        return ntk

    def ntk_svgd(self, inputs, hidden_neurons, loss_func):
        """Compute the ntk svgd in closed-form. 

        """
        assert inputs.ndim == 2 and inputs.shape[-1] == self._input_size, \
            ("inputs should has shape (batch, {})!".format(self._input_size))

        num_particles = inputs.shape[0] // 2
        inputs_i, inputs_j = torch.split(inputs, num_particles, dim=0)
        hidden_i, hidden_j = torch.split(hidden_neurons, num_particles, dim=0)
        mask_i = (hidden_i > 0).float()  # [bi, d]
        mask_j = (hidden_j > 0).float()  # [bj, d]
        D = self._decoder.weight.data
        E = self._encoder.weight.data

        # compute the first term of ntk_logp
        loss_inputs = inputs_j
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), inputs_j)[0]  # [bj, n]
        ntk_logp_1 = torch.matmul(hidden_j.t(), loss_grad)  # [d, n]
        ntk_logp_1 = torch.matmul(hidden_i, ntk_logp_1)  # [bi, n]

        # compute the second term of ntk_logp
        M = torch.matmul(inputs_i, inputs_j.t())  # [bi, bj]
        M = torch.einsum('il,ij,jl->lij', mask_i, M, mask_j)  # [d, bi, bj]
        Dvp = torch.matmul(loss_grad, D).t().unsqueeze(-1)  # [d, bj, 1]
        ntk_logp_2 = torch.bmm(M, Dvp).squeeze(-1)  # [d, bi]
        ntk_logp_2 = torch.matmul(D, ntk_logp_2).t()  # [bi, n]

        ntk_logp = (ntk_logp_1 + ntk_logp_2) / num_particles

        # compute the first term of ntk_grad
        mask_j = mask_j.mean(0)  # [d]
        M1 = mask_j.unsqueeze(0).expand_as(hidden_i)  # [bi, d]
        ntk_grad_1 = hidden_i * M1  # [bi, d]
        ntk_grad_1 = torch.matmul(ntk_grad_1, E)  # [bi, n]

        # compute the second term of ntk_grad
        M2 = mask_j.unsqueeze(0).expand_as(D)  # [n, d]
        DM = D * M2  # [n, d]
        DX = torch.matmul(inputs_i, D)  # [bi, d]
        DXM = DX * mask_i  # [bi, d]
        ntk_grad_2 = torch.matmul(DXM, DM.t())  # [bi, n]

        return ntk_logp, ntk_grad_1 + ntk_grad_2, loss
