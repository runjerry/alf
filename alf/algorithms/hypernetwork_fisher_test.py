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

from absl.testing import parameterized
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.hypernetwork_fisher_algorithm import HyperNetworkFisher
from alf.algorithms.hypernetwork_networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

class HyperNetworkTest(parameterized.TestCase, alf.test.TestCase):
    def cov(self, data, rowvar=False):
        """Estimate a covariance matrix given data.

        Args:
            data (tensor): A 1-D or 2-D tensor containing multiple observations 
                of multiple dimentions. Each row of ``mat`` represents a
                dimension of the observation, and each column a single
                observation.
            rowvar (bool): If True, then each row represents a dimension, with
                observations in the columns. Othewise, each column represents
                a dimension while the rows contains observations.

        Returns:
            The covariance matrix
        """
        x = data.detach().clone()
        if x.dim() > 2:
            raise ValueError('data has more than 2 dimensions')
        if x.dim() < 2:
            x = x.view(1, -1)
        if not rowvar and x.size(0) != 1:
            x = x.t()
        fact = 1.0 / (x.size(1) - 1)
        x -= torch.mean(x, dim=1, keepdim=True)
        return fact * x.matmul(x.t()).squeeze()

    def assertArrayGreater(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertGreater(float(torch.min(x - y)), eps)

    def plot_samples(self, inputs, targets, analytic_preds, sampled_preds):
        print (inputs.shape)
        print (targets.shape)
        print (analytic_preds.shape)
        print (sampled_preds.shape)
        fig, ax = plt.subplots(1)
        fig.suptitle("GFSF fitting Linear Regression")
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        sampled_preds = sampled_preds.cpu().detach().numpy()
        analytic_preds = analytic_preds.cpu().detach().numpy()
        ax.scatter(targets, np.zeros_like(targets), color='r', label='targets')
        ax.scatter(analytic_preds, np.zeros_like(targets), color='g', label='analytic')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('pred_err.png')
        #plt.show()
        plt.close('all')

    def plot_cov(self, true_cov, analytic_cov, learned_cov):
        fig, ax = plt.subplots(3)
        fig.suptitle("GFSF Cov")
        true_cov = true_cov.cpu().numpy()
        analytic_cov = analytic_cov.cpu().numpy()
        learned_cov = learned_cov.cpu().detach().numpy()
        ax[0].set_title('True Covariance')
        sns.heatmap(true_cov, ax=ax[0])
        ax[1].set_title('Hypernet Analytic Covariance')
        sns.heatmap(analytic_cov, ax=ax[1])
        ax[2].set_title('Hypernet Learned Covariance')
        sns.heatmap(learned_cov, ax=ax[2])
        plt.tight_layout()
        plt.savefig('cov_err_{}.png'.format(self._step))
        #plt.show()
        plt.close('all')

    # @parameterized.parameters('svgd', 'gfsf')
    def test_bayesian_linear_regression(self, particles=None):
        """
        The hypernetwork is trained to generate the parameter vector for a linear
        regressor. The target linear regressor is :math:`y = X\beta + e`, where 
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix, 
        and :math:`y` is target ouputs. The posterior of :math:`\beta` has a 
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        For a linear generator with weight W and bias b, and takes standard Gaussian 
        noise as input, the output follows a Gaussian :math:`N(b, WW^T)`, which should 
        match the posterior :math:`p(\beta|X,y)` for both svgd and gfsf.
        
        """

        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_dim = 1
        batch_size = 100
        
        inputs = torch.randn(batch_size, input_size)
        beta = torch.rand(input_size, output_dim) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_dim)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(
            inputs.t() @ inputs)  # + torch.eye(input_size))
        true_mean = true_cov @ inputs.t() @ targets
        noise_dim = 3
        particles = 10
        train_batch_size = 100
        # gen_input = torch.randn(particles, noise_dim)
        d_iters = 5
        g_iters = 1
        algorithm = HyperNetworkFisher(
            input_tensor_spec=input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            # hidden_layers=(16, ),
            hidden_layers=None,
            loss_type='regression',
            d_iters=d_iters,
            g_iters=g_iters,
            parameterization='layer',
            optimizer=alf.optimizers.Adam(lr=1e-4),
            regenerate_for_each_batch=False)
        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov norm: {}".format(true_cov.norm()))
        print("ground truth cov: {}".format(true_cov))
        self._step = 0
        self._params = None
        self._gen_input = None

        def _train():
            train_inputs = inputs
            train_targets = targets
            # re-enable grad after stopping during critic training
            for p in algorithm._net.parameters():
                p.requires_grad = True

            if self._step % (d_iters+1):
                model = 'critic'
                self._params = algorithm.sample_parameters(noise=self._gen_input)
            else:
                model = 'generator'
                #print ('generator')
                self._gen_input = torch.randn(particles, noise_dim)
                self._params = algorithm.sample_parameters(noise=self._gen_input)
            
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                params=self._params,
                model=model,
                particles=particles)
            algorithm.update_with_gradient(alg_step.info)
            self._step += 1
        
        # def _train(): # linear predictor
        def _test():

            params = algorithm.sample_parameters(particles=100)
            analytic_mean = params.mean(0)
            analytic_cov = self.cov(params)

            print("-" * 68)
            try: # parameterization: layer
                weight = algorithm._net.layer_encoders[0]._fc_layers[0].weight
                learned_mean = algorithm._net.layer_encoders[0]._fc_layers[0].bias
            except AttributeError: # parameterization: network
                weight = algorithm._net._fc_layers[0].weight
                learned_mean = algorithm._net._fc_layers[0].bias

            print("norm of generator weight: {}".format(weight.norm()))
            learned_cov = weight @ weight.t()

            sampled_preds = algorithm.predict(inputs, params=params)
            sampled_preds = sampled_preds.squeeze()  # [batch, particles]

            analytic_preds = inputs @ analytic_mean
            predicts = inputs @ learned_mean

            spred_err = torch.norm((sampled_preds - targets).mean(1))
            cpred_err = torch.norm(analytic_preds - targets.squeeze())
            pred_err = torch.norm(predicts - targets.squeeze())
            
            mean_err = torch.norm(learned_mean - true_mean.squeeze())
            mean_err = mean_err / torch.norm(true_mean)

            smean_err = torch.norm(analytic_mean - true_mean.squeeze())
            smean_err = smean_err / torch.norm(true_mean)

            scov_err = torch.norm(analytic_cov - true_cov)
            scov_err = scov_err / torch.norm(true_cov)

            cov_err = torch.norm(learned_cov - true_cov)
            cov_err = cov_err / torch.norm(true_cov)
            
            print("Train Iter: {}".format(i))
            print("\tPred err {}".format(pred_err))
            print("\tSampled pred err {}".format(spred_err))
            print("\tMean err {}".format(mean_err))
            print("\tSampled mean err {}".format(smean_err))
            print("\tCov err {}".format(cov_err))
            print("\tSampled cov err {}".format(scov_err))
            print("learned_cov norm: {}".format(learned_cov.norm()))
            
            self.plot_samples(inputs, targets, analytic_preds, sampled_preds)
            self.plot_cov(true_cov, analytic_cov, learned_cov)

        for i in range(1000000):
            _train()
            if i % 10000 == 0:
                _test()

        # self.assertArrayGreater(init_err, final_err, 10.)

    def test_hypernetwork_classification(self):
        # TODO: out of distribution tests
        # If simply use a linear classifier with random weights,
        # the cross_entropy loss does not seem to capture the distribution.
        pass


if __name__ == "__main__":
    alf.test.main()
