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
"""Networks using BatchEnsemble layers. """

import functools
import gin
import torch
import torch.nn as nn

import alf
from alf.initializers import variance_scaling_init
from alf.layers import FCBatchEnsemble, Conv2DBatchEnsemble
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common


@gin.configurable
class BatchEnsembleImageNetwork(Network):
    """
    A general template class for creating convolutional encoding networks.
    """

    def __init__(self,
                 input_channels,
                 input_size,
                 ensemble_size,
                 conv_layer_params,
                 pooling_layer_params=None,
                 same_padding=False,
                 activation=torch.relu_,
                 use_bias=None,
                 kernel_initializer=None,
                 flatten_output=False,
                 name="BatchEnsembleImageNetwork"):
        """
        Initialize the BatchEnsemble layers for encoding an image into a latent 
        vector. Currently there seems no need for this class to handle nested 
        inputs; If necessary, extend the argument list to support it in the future.

        How to calculate the output size:
        `<https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`_::

            H = (H1 - HF + 2P) // strides + 1

        where H = output size, H1 = input size, HF = size of kernel, P = padding.

        Regarding padding: in the previous TF version, we have two padding modes:
        ``valid`` and ``same``. For the former, we always have no padding (P=0); for
        the latter, it's also called "half padding" (P=(HF-1)//2 when strides=1
        and HF is an odd number the output has the same size with the input.
        Currently, PyTorch don't support different left and right paddings and
        P is always (HF-1)//2. So if HF is an even number, the output size will
        decrease by 1 when strides=1).

        Args:
            input_channels (int): number of channels in the input image
            input_size (int or tuple): the input image size (height, width)
            ensemble_size (int): ensemble size
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            pooling_layer_params (tuple): a tuple (str, int), where str denotes
                the type of pooling layers, options are [``max``, ``avg``],
                int denotes the pooling kernel.
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in `conv_layer_params` will be
                replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though)
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            flatten_output (bool): If False, the output will be an image
                structure of shape ``BxCxHxW``; otherwise the output will be
                flattened into a feature of shape ``BxN``.
        """
        input_size = common.tuplify2d(input_size)
        super().__init__(
            input_tensor_spec=TensorSpec((input_channels, ) + input_size),
            name=name)

        assert isinstance(conv_layer_params, tuple)
        assert len(conv_layer_params) > 0

        self._flatten_output = flatten_output
        self._conv_layer_params = conv_layer_params
        self._conv_layers = nn.ModuleList()
        for paras in conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            pooling_kernel = paras[4] if len(paras) > 4 else None
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            self._conv_layers.append(
                Conv2DBatchEnsemble(
                    input_channels,
                    filters,
                    kernel_size,
                    ensemble_size=ensemble_size,
                    output_ensemble_ids=True,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    pooling_kernel=pooling_kernel,
                    padding=padding))
            input_channels = filters

        if pooling_layer_params:
            assert isinstance(pooling_layer_params, tuple), \
                "The input params {} should be tuple".format(pooling_layer_params)
            pooling_type, pooling_kernel = pooling_layer_params
            assert pooling_type in ['max', 'avg'], \
                "Only 'max' and 'avg' pooling types are supported"
            if pooling_type is 'max':
                self._pooling = nn.MaxPool2d(kernel_size=pooling_kernel)
            else:
                self._pooling = nn.AvgPool2d(kernel_size=pooling_kernel)
        else:
            self._pooling = None

    def forward(self, inputs, state=()):
        """The empty state just keeps the interface same with other networks."""
        z = inputs
        for conv_l in self._conv_layers:
            z = conv_l(z)
        ensemble_ids = None
        if type(z) == tuple:
            z, ensemble_ids = z
        if self._pooling is not None:
            z = self._pooling(z)
        if self._flatten_output:
            z = z.view(z.size(0), -1)
        if ensemble_ids is not None:
            z = (z, ensemble_ids)
        return z, state


@alf.configurable
class BatchEnsembleNetwork(Network):
    def __init__(self,
                 input_tensor_spec,
                 ensemble_size,
                 conv_layer_params=None,
                 pooling_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 last_layer_size=None,
                 last_activation=None,
                 last_kernel_initializer=None,
                 last_use_fc_bn=False,
                 name="BatchEnsembleNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            ensemble_size (int): ensemble size
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            pooling_layer_params (tuple): a tuple (str, int), where str denotes
                the type of pooling layers, options are [``max``, ``avg``],
                int denotes the pooling kernel.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FCBatchEnsemble layer sizes.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If None, a variance_scaling_initializer will be
                used.
            use_fc_bn (bool): whether use Batch Normalization for fc layers.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_fc_bn (bool): whether use Batch Normalization for the last
                fc layer.
            last_kernel_initializer (Callable): initializer for the the
                additional layer specified by ``last_layer_size``.
                If None, it will be the same with ``kernel_initializer``. If
                ``last_layer_size`` is None, ``last_kernel_initializer`` will
                not be used.
            name (str):
        """

        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._ensemble_size = ensemble_size
        self._img_net = None
        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert input_tensor_spec.ndim == 3, \
                "The input shape {} should be like (C,H,W)!".format(
                    input_tensor_spec.shape)
            input_channels, height, width = input_tensor_spec.shape
            self._img_net = BatchEnsembleImageNetwork(
                input_channels, (height, width),
                ensemble_size=ensemble_size,
                conv_layer_params=conv_layer_params,
                pooling_layer_params=pooling_layer_params,
                activation=activation,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                flatten_output=True)
            output_spec = self._img_net.output_spec
            if isinstance(output_spec, tuple):
                output_spec = output_spec[0]
            input_size = output_spec.shape[0]
        else:
            assert input_tensor_spec.ndim == 1, \
                "The input shape {} should be like (N,)!".format(
                    input_tensor_spec.shape)
            input_size = input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        for size in fc_layer_params:
            self._fc_layers.append(
                FCBatchEnsemble(
                    input_size,
                    size,
                    ensemble_size=ensemble_size,
                    use_bias=True,
                    output_ensemble_ids=True,
                    activation=activation,
                    use_bn=use_fc_bn,
                    kernel_initializer=kernel_initializer))
            input_size = size

        if last_layer_size is not None or last_activation is not None:
            assert last_layer_size is not None and last_activation is not None, \
            "Both last_layer_size and last_activation need to be specified!"

            if last_kernel_initializer is None:
                common.warning_once(
                    "last_kernel_initializer is not specified "
                    "for the last layer of size {}.".format(last_layer_size))
                last_kernel_initializer = kernel_initializer

            self._fc_layers.append(
                FCBatchEnsemble(
                    input_size,
                    last_layer_size,
                    ensemble_size=ensemble_size,
                    use_bias=True,
                    activation=last_activation,
                    use_bn=last_use_fc_bn,
                    kernel_initializer=last_kernel_initializer))
            input_size = last_layer_size

        self._output_spec = TensorSpec((input_size, ),
                                       dtype=self._input_tensor_spec.dtype)

    def forward(self, inputs, state=(), full_ensemble=False):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.

        Returns:
            outputs (Tensor|tuple): if a Tensor, its shape should be ``[B, output_dim]``
                or ``[B, N, output_dim]``
        """
        outer_rank = alf.nest.utils.get_outer_rank(inputs,
                                                   self._input_tensor_spec)
        assert 1 <= outer_rank <= 2, ("inputs should have shape [B, %d, ...] "
                                      "or [B, ...]" % self._ensemble_size)
        if outer_rank == 2:
            assert inputs.shape[1] == self._ensemble_size, (
                "the inputs have shape "
                "[B, n, ...] but n does not match ensemble_size %d" %
                self._ensemble_size)

        if full_ensemble:
            batch_size = inputs.shape[0]
            inputs = torch.repeat_interleave(
                inputs, self._ensemble_size, dim=0)
            ensemble_ids = torch.arange(self._ensemble_size).repeat(batch_size)
            x = (inputs, ensemble_ids)  # ([B*N, d_in], [B*N])
        else:
            x = inputs

        if self._img_net is not None:
            x, state = self._img_net(x, state)
        if alf.summary.should_summarize_output():
            name = ('summarize_output/' + self.name + '.fc.0.' + 'input_norm.'
                    + common.exe_mode_name())
            alf.summary.scalar(
                name=name, data=torch.mean(x.norm(dim=list(range(1, x.ndim)))))
        i = 0
        for fc in self._fc_layers:
            x = fc(x)
            if alf.summary.should_summarize_output():
                name = ('summarize_output/' + self.name + '.fc.' + str(i) +
                        '.output_norm.' + common.exe_mode_name())
                alf.summary.scalar(
                    name=name,
                    data=torch.mean(x.norm(dim=list(range(1, x.ndim)))))
            i += 1

        if isinstance(x, tuple):
            x = x[0]
        if full_ensemble:
            x = x.reshape(batch_size, self._ensemble_size, -1)  # [B, N, d_out]
        return x, state
