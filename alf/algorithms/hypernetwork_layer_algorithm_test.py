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
from alf.algorithms.config import TrainerConfig
from alf.data_structures import LossInfo
from alf.algorithms.hypernetwork_layer_algorithm import HyperNetwork
from alf.algorithms.hypernetwork_networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops, datagen

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score


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

    def plot_predictions(self, inputs, targets, computed_preds, step):
        fig, ax = plt.subplots(1)
        fig.suptitle("Bayes Linear Regression Predictions")
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        computed_preds = computed_preds.cpu().detach().numpy()
        ax.scatter(targets, np.zeros_like(targets), color='r', label='targets')
        ax.scatter(computed_preds, np.zeros_like(targets), color='g', label='computed')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('predictions_step_{}.png'.format(step))
        # plt.show()
        plt.close('all')

    def plot_cov_heatmap(self, true_cov, computed_cov, learned_cov, step):
        fig, ax = plt.subplots(3)
        fig.suptitle("Bayes Linear Regression Covariance Heatmap")
        true_cov = true_cov.cpu().numpy()
        computed_cov = computed_cov.cpu().numpy()
        learned_cov = learned_cov.cpu().detach().numpy()
        ax[0].set_title('True Covariance')
        sns.heatmap(true_cov, ax=ax[0])
        ax[1].set_title('Hypernet Analytic Covariance')
        sns.heatmap(computed_cov, ax=ax[1])
        ax[2].set_title('Hypernet Learned Covariance')
        sns.heatmap(learned_cov, ax=ax[2])
        plt.tight_layout()
        plt.savefig('cov_heatmap_step_{}'.format(step))
        # plt.show()
        plt.close('all')

    @parameterized.parameters(
                              #('svgd3', False, True), ('svgd3', True, True),
                              #('gfsf', False, False), ('gfsf', True, False),
                              #('gfsf', False, True), ('gfsf', True, True ),
                              #('svgd3', False, True, False),
                              #('svgd3', False, True, True),
                              #('minmax', False, False, False),
                              ('minmax', False, True, True),
    )
    def test_bayesian_linear_regression(self,
                                        par_vi='svgd3',
                                        function_vi=False,
                                        amortize=True,
                                        functional_gradient=False,
                                        train_batch_size=10,
                                        num_particles=512):
        """
        The hypernetwork is trained to generate the parameter vector for a linear
        regressor. The target linear regressor is :math:`y = X\beta + e`, where 
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix, 
        and :math:`y` is target ouputs. The posterior of :math:`\beta` has a 
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        For a linear generator with weight W and bias b, and takes standard Gaussian 
        noise as input, the output follows a Gaussian :math:`N(b, WW^T)`, which should 
        match the posterior :math:`p(\beta|X,y)` for both ``svgd``, ``gfsf``, and
        ``minmax``.
        """

        print ("Testing {} with {} particles".format(par_vi, num_particles))
        print ("Amortize: {}, \nFunction values: {}, \nFunctional gradient: {}".format(
            amortize, function_vi, functional_gradient))
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_dim = 1
        batch_size = 100
        hidden_size = output_dim * batch_size
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_dim) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_dim)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(inputs.t() @ inputs) 
        true_mean = true_cov @ inputs.t() @ targets
        noise_dim = input_size
        if amortize is True:
            lr = 1e-3
        else:
            lr = 1e-2
        if functional_gradient:
            hidden_layers = ()
            parameterization = 'network'
        else:
            hidden_layers = None
            parameterization = 'layer'

        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=hidden_layers,
            loss_type='regression',
            par_vi=par_vi,
            num_particles=num_particles,
            amortize_vi=amortize,
            functional_gradient=functional_gradient,
            particle_optimizer=alf.optimizers.Adam(lr=lr),
            function_vi=function_vi,
            function_bs=train_batch_size,
            function_space_samples=1,
            use_pinverse=True,
            pinverse_type='network',
            pinverse_resolve=False,
            pinverse_solve_iters=3,
            parameterization=parameterization,
            critic_hidden_layers=(hidden_size, hidden_size),
            critic_l2_weight=5.0,
            optimizer=alf.optimizers.Adam(lr=lr),
            critic_optimizer=alf.optimizers.Adam(lr=lr))
        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov norm: {}".format(true_cov.norm()))
        print("ground truth cov: {}".format(true_cov))

        def _train(i, train_batch=None, entropy_regularization=None):
            if train_batch is None:
                perm = torch.randperm(batch_size)
                idx = perm[:train_batch_size]
                train_inputs = inputs[idx]
                train_targets = targets[idx]
            else:
                train_inputs, train_targets = train_batch
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size

            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles)
            if functional_gradient:
                pinverse_loss = alg_step.info.extra.pinverse
                if i % 500 == 0: print(pinverse_loss)
            
            if amortize or function_vi:
                loss_info, params = algorithm.update_with_gradient(alg_step.info)
            else:
                update_direction = alg_step.info.loss
                algorithm._particle_optimizer.zero_grad()
                algorithm._params.grad = update_direction
                algorithm._particle_optimizer.step()
        
        def _test(i, sampled_predictive=True):

            print("-" * 68)
            if amortize:
                if parameterization == 'layer':
                    weight = algorithm._generator._net.layer_encoders[0]._fc_layers[0].weight
                    learned_mean = algorithm._generator._net.layer_encoders[0]._fc_layers[0].bias
                else:
                    weight = algorithm._generator._net._fc_layers[0].weight
                    learned_mean = algorithm._generator._net._fc_layers[0].bias

                learned_cov = weight @ weight.t()
                print("norm of generator weight: {}".format(weight.norm()))
                print("norm of learned cov: {}".format(learned_cov.norm()))
               
                predicts = inputs @ learned_mean  # [batch]
                pred_err = torch.norm(predicts - targets.squeeze())

                mean_err = torch.norm(learned_mean - true_mean.squeeze())
                mean_err = mean_err / torch.norm(true_mean)

                cov_err = torch.norm(learned_cov - true_cov)
                cov_err = cov_err / torch.norm(true_cov)
                
                print("Train Iter: {}".format(i))
                print("\tpred err {}".format(pred_err))
                print("\tmean err {}".format(mean_err))
                print("\tcov err {}".format(cov_err))

            if sampled_predictive:
                params = algorithm.sample_parameters(num_particles=200)
                pred_step = algorithm.predict_step(inputs, params=params)
                if functional_gradient:
                    params = params[0]
                sampled_preds = pred_step.output.squeeze(
                )  # [batch, n_particles]
                spred_err = torch.norm((sampled_preds - targets).mean(1))
                print("train_iter {}: sampled pred err {}".format(
                    i, spred_err))

                computed_mean = params.mean(0)
                smean_err = torch.norm(computed_mean - true_mean.squeeze())
                smean_err = smean_err / torch.norm(true_mean)
                print("train_iter {}: sampled mean err {}".format(
                    i, smean_err))

                computed_cov = self.cov(params)
                scov_err = torch.norm(computed_cov - true_cov)
                scov_err = scov_err / torch.norm(true_cov)
                print("train_iter {}: sampled cov err {}".format(i, scov_err))

            self.plot_predictions(inputs, targets, predicts, i)
            self.plot_cov_heatmap(true_cov, computed_cov, learned_cov, i)
        
        train_iter = 10000
        for i in range(train_iter):
            _train(i)
            if i % 1000 == 0:
                _test(i)

        if amortize:
            if parameterization == 'layer':
                weight = algorithm._generator._net.layer_encoders[0]._fc_layers[0].weight
                learned_mean = algorithm._generator._net.layer_encoders[0]._fc_layers[0].bias
            else:
                weight = algorithm._generator._net._fc_layers[0].weight
                learned_mean = algorithm._generator._net._fc_layers[0].bias
            mean_err = torch.norm(learned_mean - true_mean.squeeze())
            mean_err = mean_err / torch.norm(true_mean)
            learned_cov = weight @ weight.t()
            cov_err = torch.norm(learned_cov - true_cov)
            cov_err = cov_err / torch.norm(true_cov)
            print("-" * 68)
            print("train_iter {}: mean err {}".format(train_iter, mean_err))
            print("train_iter {}: cov err {}".format(train_iter, cov_err))
            self.assertLess(mean_err, 0.5)
            self.assertLess(cov_err, 0.5)
        else:
            params = algorithm.sample_parameters()
            smean_err = torch.norm(params.mean(0) - true_mean.squeeze())
            smean_err = smean_err / torch.norm(true_mean)
            scov_err = torch.norm(self.cov(params) - true_cov)
            scov_err = scov_err / torch.norm(true_cov)
            print("train_iter {}: sampled mean err {}".format(i, smean_err))
            print("train_iter {}: sampled cov err {}".format(i, scov_err))
            self.assertLess(smean_err, 0.5)
            self.assertLess(scov_err, 0.5)


if __name__ == "__main__":
    alf.test.main()
