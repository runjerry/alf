# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl.testing import parameterized
import numpy as np
import torch

import alf
from alf.networks import ParamConvNet, ParamNetwork, CriticDistributionParamNetwork
from alf.networks.encoding_networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.dist_utils import DistributionSpec


class ParamNetworksTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((1, True, True, False), (3, False, True, True))
    def test_param_convnet(self,
                           batch_size=1,
                           same_padding=False,
                           use_bias=True,
                           flatten_output=False):
        input_spec = TensorSpec((3, 32, 32), torch.float32)
        network = ParamConvNet(
            input_channels=input_spec.shape[0],
            input_size=input_spec.shape[1:],
            conv_layer_params=((16, (2, 2), 1, (1, 0)), (15, 2, (1, 2), 1, 2)),
            same_padding=same_padding,
            activation=torch.tanh,
            flatten_output=flatten_output)
        self.assertLen(network._conv_layers, 2)

        # test non-parallel forward
        image = input_spec.zeros(outer_dims=(batch_size, ))
        output, _ = network(image)
        if same_padding:
            output_shape = (batch_size, 15, 15, 7)
        else:
            output_shape = (batch_size, 15, 17, 8)
        if flatten_output:
            output_shape = (batch_size, np.prod(output_shape[1:]))
        self.assertEqual(output_shape[1:], network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))

        # test parallel forward
        replica = 2
        image = input_spec.zeros(outer_dims=(batch_size, ))
        replica_image = input_spec.zeros(outer_dims=(batch_size, replica))
        params = torch.randn(replica, network.param_length)
        network.set_parameters(params)
        output, _ = network(image)
        replica_output, _ = network(replica_image)
        self.assertEqual(output.shape, replica_output.shape)

        if same_padding:
            output_shape = (batch_size, replica, 15, 15, 7)
        else:
            output_shape = (batch_size, replica, 15, 17, 8)
        if flatten_output:
            output_shape = (*output_shape[0:2], np.prod(output_shape[2:]))
        self.assertEqual(output_shape[1:], network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))

    @parameterized.parameters(1, 3)
    def test_param_network(self, batch_size=1):
        input_spec = TensorSpec((3, 32, 32), torch.float32)
        conv_layer_params = ((16, (2, 2), 1, (1, 0)), (15, 2, (1, 2), 1))
        fc_layer_params = ((128, True), )
        last_layer_size = 10
        last_activation = math_ops.identity
        network = ParamNetwork(
            input_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            last_layer_param=(last_layer_size, True),
            last_activation=last_activation)
        self.assertLen(network._fc_layers, 2)
        ref_net = EncodingNetwork(
            input_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=(128, ),
            last_layer_size=10,
            last_activation=last_activation)

        # test non-parallel forward
        if ref_net._img_encoding_net is not None:
            for conv_l, pconv_l in zip(ref_net._img_encoding_net._conv_layers,
                                       network._conv_net._conv_layers):
                conv_l.weight.data.copy_(pconv_l.weight)
        for fc_l, pfc_l in zip(ref_net._fc_layers, network._fc_layers):
            fc_l.weight.data.copy_(pfc_l.weight.squeeze(0))
            if fc_l._bias is not None:
                fc_l.bias.data.copy_(pfc_l.bias.squeeze(0))
        image = input_spec.randn(outer_dims=(batch_size, ))
        output, _ = network(image)
        ref_output, _ = ref_net(image)
        output_shape = (batch_size, last_layer_size)
        self.assertEqual(output_shape[1:], network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))
        self.assertTensorEqual(output, ref_output, 1e-6)

        # test parallel forward
        replica = 2
        image = input_spec.randn(outer_dims=(batch_size, ))
        replica_image = input_spec.randn(outer_dims=(batch_size, replica))
        replica_image = torch.repeat_interleave(
            image.unsqueeze(1), replica, dim=1)
        params = torch.randn(replica, network.param_length)
        network.set_parameters(params)
        ref_pnet = ref_net.make_parallel(replica)
        if ref_pnet._img_encoding_net is not None:
            for conv_l, pconv_l in zip(ref_pnet._img_encoding_net._conv_layers,
                                       network._conv_net._conv_layers):
                conv_l.weight.data.copy_(
                    pconv_l.weight.reshape(replica, -1,
                                           *pconv_l.weight.shape[1:]))
        for fc_l, pfc_l in zip(ref_pnet._fc_layers, network._fc_layers):
            fc_l.weight.data.copy_(pfc_l.weight.squeeze(0))
            if fc_l._bias is not None:
                fc_l.bias.data.copy_(pfc_l.bias.squeeze(0))
        output, _ = network(image)
        ref_output, _ = ref_pnet(image)
        replica_output, _ = network(replica_image)
        self.assertEqual(output.shape, replica_output.shape)
        self.assertTensorEqual(output, replica_output, 1e-6)
        self.assertTensorEqual(output, ref_output, 1e-6)

        output_shape = (batch_size, replica, last_layer_size)
        self.assertEqual(output_shape[1:],
                         (replica, ) + network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))

    @parameterized.parameters(1, 3, (3, True))
    def test_critic_distribution_param_network(self,
                                               batch_size=3,
                                               deterministic=False):
        observation_spec = TensorSpec((3, 20, 20), torch.float32)
        action_spec = TensorSpec((5, ), torch.float32)
        input_spec = (observation_spec, action_spec)
        replica = 2

        observation_conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        action_fc_layer_params = ((10, True), (8, True))
        joint_fc_layer_params = ((6, True), (4, True))

        image = observation_spec.zeros(outer_dims=(batch_size, ))
        action = action_spec.randn(outer_dims=(batch_size, ))

        network_input = (image, action)

        critic_net = CriticDistributionParamNetwork(
            input_spec,
            observation_conv_layer_params=observation_conv_layer_params,
            action_fc_layer_params=action_fc_layer_params,
            joint_fc_layer_params=joint_fc_layer_params,
            deterministic=deterministic)

        params = torch.randn(replica, critic_net.param_length)
        critic_net.set_parameters(params)

        dist, state = critic_net(network_input)
        if deterministic:
            self.assertTrue(dist.shape, (batch_size, replica, 1))
            self.assertTrue(isinstance(critic_net.output_spec, TensorSpec))
        else:
            out = dist.sample((10, ))
            self.assertTrue(
                isinstance(critic_net.output_spec, DistributionSpec))
            self.assertTrue(dist.batch_shape, (batch_size, replica))
            self.assertTrue(dist.base_dist.batch_shape,
                            (batch_size, replica, 1))
            self.assertTrue(out.std() > 0)


if __name__ == "__main__":
    alf.test.main()
