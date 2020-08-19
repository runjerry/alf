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

import torch

import alf
from alf.algorithms.ntk import SimpleMLP
from alf.tensor_specs import TensorSpec


def jacobian(y, x, create_graph=False):
    """It is from Adam Paszke's implementation:
    https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.

    return torch.stack(jac).reshape(y.shape + x.shape)


class NtkTest(alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    def test_ntk(self, input_size=5, hidden_layer_size=2):
        spec = TensorSpec((input_size, ))
        mlp = SimpleMLP(spec, hidden_layer_size=hidden_layer_size)
        x = torch.randn(input_size)
        outputs, _ = mlp(x, requires_ntk=True)
        y, ntk = outputs

        # compute ntk using autograd
        Jd = jacobian(y, mlp._decoder.weight)
        Je = jacobian(y, mlp._encoder.weight)
        Jd = Jd.reshape(input_size, mlp._decoder.weight.data.nelement())
        Je = Je.reshape(input_size, mlp._encoder.weight.data.nelement())

        jac = torch.cat((Jd, Je), dim=-1)
        ntk2 = jac @ jac.t()

        self.assertArrayEqual(ntk, ntk2, 1e-6)


if __name__ == "__main__":
    alf.test.main()
