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
from alf.algorithms.hypernetwork_layer_algorithm import HyperNetwork
from alf.algorithms.hypernetwork_networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class HyperNetworkSampleTest(parameterized.TestCase, alf.test.TestCase):
    """ 
    HyperNetwork Sample Test
        A series of three tests given in order of increasing difficulty. 
        1. A 3-dimensional multivariate Gaussian distribution 
        2. A pair of standard normal distributions N(+/-5, .5) with vanishing
            overlapping mass
        3. A funnel distribution as given in (Neal, 2004). A heirarchical model
            that is difficult for HMC to fit. 
        
        We do not directly compute the posterior for any of these distributions
        We instead determine closeness by sampling predictors from our 
        hypernetwork, and comparing the resulting prediction statistics to 
        samples from the true model. 
    """
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

    def neglogprob_3dGaussian(self, data):
        """Function to get the corresponding log-prob of some input data
            under a 1d standard normal.
        Args:
            data (tensor): A 1-D tensor containing samples we want to evaluate 
        Returns:
            log probability of the data under the model
        """
        if data.dim() > 1:
            raise ValueError('data has more than 1 dimension')
        means = torch.tensor([0., 0., 0.])
        stdev = torch.tensor([0.5, 1., 2.0])
        dist = torch.distributions.MultivariateNormal(means, torch.diag(stdev**2))
        logp_x = dist.log_prob(data)
        return -logp_x

    def plot_hypernetwork_3d(self, inputs):
        mean = torch.tensor([0., 0., 0.]).cpu()
        stdev = torch.tensor([0.5, 1.0, 2.0]).cpu()
        x = torch.distributions.MultivariateNormal(
            mean, torch.diag(stdev**2)).sample([inputs.shape[0]])
        h_mean = inputs.mean(0)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.scatter(x[:, 1], x[:, 2], s=5, alpha=.4, color='b', label='True Samples')
        ax.scatter(inputs[:, 1], inputs[:, 2], s=5, alpha=.4, color='m', label='Net')
        ax.scatter(h_mean[1], h_mean[2], marker='*', color='g', s=100, label='Net Mean')
        ax.scatter(mean[1], mean[2], marker='*', color='C3', s=100, label='True Mean')
        
        ax.legend(fontsize=17)
        ax.grid()
        ax.set_ylim([-5, 5])
        ax.set_xlim([-5, 5])
        plt.savefig('hypernet_fit_3d.png')
        # plt.show()
        plt.close('all')
    
    #@parameterized.parameters(('svgd3'), ('gfsf'), ('minmax'))
    def test_3dGaussian_hypernetwork(self, par_vi='svgd', particles=32):
        """
        """
        print ('HyperNetwork: Fitting 3d Gaussian')
        input_size = 2
        output_dim = 1
        batch_size = 100
        noise_dim = 32
        particles = 32
        input_spec = TensorSpec((input_size, ), torch.float32)
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        train_batch_size = 50
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(32, 32),
            loss_type='regression',
            par_vi=par_vi,
            parameterization='layer',
            optimizer=alf.optimizers.Adam(lr=1e-4))
        
        loss_type = 'regression'
        ll_loss_func = self.neglogprob_3dGaussian
        target_mean = torch.zeros(3)
        target_std = torch.tensor([.5, 1.0, 2.0]).pow(.5).diag()
        dist = torch.distributions.MultivariateNormal(target_mean, target_std)
        inputs = dist.sample([10000]) # [B, D]
        targets = inputs
        
        def _train(train_batch=None, entropy_regularization=None):
            if train_batch is None:
                perm = torch.randperm(batch_size)
                idx = perm[:train_batch_size]
                train_inputs = inputs[idx]
                train_targets = targets[idx]
            else:
                train_inputs, train_targets = train_batch
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
            
            if loss_type == 'logprob':
                params = algorithm.sample_parameters(particles=particles)
                alg_step = algorithm._generator.train_step(
                    inputs=None,
                    loss_func=functools.partial(loss_func, inputs,
                        algorithm._param_net, 'regression'),
                    entropy_regularization=entropy_regularization,
                    outputs=params,
                    state=())
            elif loss_type == 'regression':
                train_inputs_ = train_inputs[:, :2]
                train_targets = train_inputs[:, 2:]
                alg_step = algorithm.train_step(
                    inputs=(train_inputs_, train_targets),
                    particles=particles,
                    entropy_regularization=entropy_regularization,
                    state=())

            algorithm.update_with_gradient(alg_step.info)
        
        def _test(i):
            test_p = 200
            test_samples = dist.sample([100])
            test_inputs = test_samples[:, :2] 
            test_targets = test_samples[:, 2:] 

            params = algorithm.sample_parameters(particles=test_p)
            sample_outputs = algorithm.predict_step(test_inputs, params).output
            sample_outputs = sample_outputs.transpose(0, 1)
            computed_mean = sample_outputs.mean(0)
            
            computed_cov = self.cov(sample_outputs.reshape(test_p, -1))
            computed_cov_data = self.cov(
                test_targets.unsqueeze(0).repeat(test_p, 1, 1).reshape(
                    test_p, -1))

            print("-" * 68)
            mean_err = torch.norm(computed_mean - test_targets.mean())
            pred_err = torch.norm(sample_outputs - test_targets)
            cov_err = torch.norm(computed_cov - computed_cov_data)
            
            print ('iter ', i)
            print ('mean error', mean_err)
            print ('pred err', pred_err)
            print ('cov_err', cov_err)
            test_inputs = test_inputs.unsqueeze(0).repeat(test_p, 1, 1)
            test_inputs = torch.cat((test_inputs, sample_outputs), -1)
            return test_inputs.reshape(-1, 3).detach()

        train_iter = 10000
        for i in range(train_iter):
            _train()
            if i % 1000 == 0:
                samples = _test(i)

        #self.assertLess(mean_err, .3)
        #self.assertLess(pred_err, .3)
        #self.assertLess(cov_err, .3)
        self.plot_hypernetwork_3d(samples.cpu().numpy())
    
    def log_prob_mog2(self, data):
        if data.dim() > 1:
            raise ValueError('data has more than 1 dimension')
        loc = torch.tensor([[-2.5, 0.0], [2.5, 0.0]])
        cov = torch.tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1)
        mix = torch.distributions.Categorical(torch.ones(2,))
        comp = torch.distributions.MultivariateNormal(loc, cov)
        mog_dist = torch.distributions.MixtureSameFamily(mix, comp)
        logp = mog_dist.log_prob(data)
        return logp

    def plot_hmc_mog2(self, inputs):
        loc = torch.tensor([[-2.5, 0.0], [2.5, 0.0]])
        cov = torch.tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1)
        mix = torch.distributions.Categorical(torch.ones(2,))
        comp = torch.distributions.MultivariateNormal(loc, cov)
        mog = torch.distributions.MixtureSameFamily(mix, comp)

        samples = mog.sample([len(inputs)]).cpu().numpy()
        loc1, loc2 = mog.component_distribution.mean.cpu().numpy()
        
        pmean = inputs.mean(0)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(inputs[:, 0], inputs[:, 1], color='m', marker='o', s=5, alpha=.4, label='HMC')
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=.4, color='b', label='True Samples')
        
        ax.scatter(pmean[0], pmean[1], marker='*',color='g',s=100,label='HMC Mean')
        ax.scatter(loc1[0], loc1[1], marker='*',color='C3',s=100,label='True Mean')
        ax.scatter(loc2[0], loc2[1], marker='*',color='C3',s=100,label='True Mean')
        ax.legend(fontsize=17)
        ax.grid()
        ax.set_ylim([-9, 9])
        ax.set_xlim([-5, 5])
        plt.savefig('hmc_fit_mog.png')
        # plt.show()
        plt.close('all')

    def test_MixtureGaussian_hmc(self):
        """
        Using a the log probability function defined above. We want to use this
        to train an HMC chain that evolves towards fitting the target mixture 
        distribution. 
        A Gaussian mixture model with N compnents, has a posterior distribution
        that is also a mixture model, distributed according to:
        p(\theta|x) = \sum^N_{i=1} \phi_i(x) N(\mu_i, \sigma_i), where \phi is 
        a categorical "indexing" distribution. 
        """
        print ('HMC: Fitting Mixture of 2 Gaussians')
        input_size = 2
        output_dim = 1
        batch_size = 100
        noise_dim = 32
        particles = 32
        input_spec = TensorSpec((input_size, ), torch.float32)
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        train_batch_size = 50
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(32, 32),
            loss_type='regression',
            par_vi=par_vi,
            parameterization='layer',
            optimizer=alf.optimizers.Adam(lr=1e-4))
        
        loss_type = 'regression'
        ll_loss_func = self.neglogprob_3dGaussian
        target_mean = torch.zeros(3)
        target_std = torch.tensor([.5, 1.0, 2.0]).pow(.5).diag()
        dist = torch.distributions.MultivariateNormal(target_mean, target_std)
        inputs = dist.sample([10000]) # [B, D]
        targets = inputs
        

        def _train(train_batch=None, entropy_regularization=None):
            if train_batch is None:
                perm = torch.randperm(batch_size)
                idx = perm[:train_batch_size]
                train_inputs = inputs[idx]
                train_targets = targets[idx]
            else:
                train_inputs, train_targets = train_batch
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
            
            if loss_type == 'logprob':
                params = algorithm.sample_parameters(particles=particles)
                alg_step = algorithm._generator.train_step(
                    inputs=None,
                    loss_func=functools.partial(loss_func, inputs,
                        algorithm._param_net, 'regression'),
                    entropy_regularization=entropy_regularization,
                    outputs=params,
                    state=())
            elif loss_type == 'regression':
                train_inputs_ = train_inputs[:, :2]
                train_targets = train_inputs[:, 2:]
                alg_step = algorithm.train_step(
                    inputs=(train_inputs_, train_targets),
                    particles=particles,
                    entropy_regularization=entropy_regularization,
                    state=())

            algorithm.update_with_gradient(alg_step.info)
        
        def _test(i):
            test_p = 200
            test_samples = dist.sample([100])
            test_inputs = test_samples[:, :2] 
            test_targets = test_samples[:, 2:] 

            params = algorithm.sample_parameters(particles=test_p)
            sample_outputs = algorithm.predict_step(test_inputs, params).output
            sample_outputs = sample_outputs.transpose(0, 1)
            computed_mean = sample_outputs.mean(0)
            
            computed_cov = self.cov(sample_outputs.reshape(test_p, -1))
            computed_cov_data = self.cov(
                test_targets.unsqueeze(0).repeat(test_p, 1, 1).reshape(
                    test_p, -1))

            print("-" * 68)
            mean_err = torch.norm(computed_mean - test_targets.mean())
            pred_err = torch.norm(sample_outputs - test_targets)
            cov_err = torch.norm(computed_cov - computed_cov_data)
            
            print ('iter ', i)
            print ('mean error', mean_err)
            print ('pred err', pred_err)
            print ('cov_err', cov_err)
            test_inputs = test_inputs.unsqueeze(0).repeat(test_p, 1, 1)
            test_inputs = torch.cat((test_inputs, sample_outputs), -1)
            return test_inputs.reshape(-1, 3).detach()

        train_iter = 10000
        for i in range(train_iter):
            _train()
            if i % 1000 == 0:
                samples = _test(i)

        #self.assertTensorClose(lside_mean, true_lmean, .1)
        #self.assertTensorClose(rside_mean, true_rmean, .1)
        #self.assertTensorClose(lside_std, true_lstd, .3)
        #self.assertTensorClose(rside_std, true_rstd, .3)
 

    def log_prob_funnel(self, data):
        v_dist = torch.distributions.Normal(0, 3)
        logp = v_dist.log_prob(data[0])
        x_dist = torch.distributions.Normal(0, torch.exp(-data[0])**.5)
        logp += x_dist.log_prob(data[1:]).sum()
        return logp

    def plot_hmc_funnel(self, inputs):
        y_dist = torch.distributions.Normal(0, 3)
        y_samples = y_dist.sample([1000]).cpu()
        x_dist = torch.distributions.Normal(0, torch.exp(-y_samples)**.5)
        x_samples = x_dist.sample().cpu()
        hmc_mean = inputs.mean(0)
        
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(inputs[:, 1], inputs[:, 0], s=5, color='m', alpha=.4, label='HMC')
        ax.scatter(x_samples.cpu(), y_samples.cpu(), s=5, alpha=.4, color='b', label='True Samples')
        ax.scatter(hmc_mean[1], hmc_mean[0], marker='o', color='g', s=200, label='HMC Mean')
        ax.scatter(x_samples.mean(), y_samples.mean(), marker='*', color='C3', s=200, label='True Mean')
        ax.legend(fontsize=17)
        ax.grid()
        ax.set_ylim([0, 7])
        ax.set_xlim([-4, 4])
        plt.savefig('hmc_fit_funnel.png')
        # plt.show()
        plt.close('all')

    def test_funnel_hmc(self):
        """
        Fit a Funnel distribution (Neal, 2003): a hierarchical model that's
        difficult to fit for even modern MCMC techniques without reparameterization:
        p(y|x) = N(y|0, 3) * \prod_i N(x_i|0, \sqrt{\exp{-y}})
        We can't directly compute this posterior, but the marginal p(y) w.r.t
        any component i is Gaussian with mean=0 and std=3. We can then compare
        samples from p(y) to check that we have fit the funnel. 
        """
        num_samples = 5000
        step_size = 0.2
        num_steps_per_sample = 25
        params = torch.ones(11)
        params[0] = 0.
        burn_in_steps = 0
        inv_mass = None
        algorithm = HMC(
            self.log_prob_funnel,
            params,
            num_samples=num_samples,
            steps_per_sample=num_steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            inv_mass=inv_mass,
            name='HMCTest')

        #def _test():
        #    return algorithm.sample(params, num_samples)
        #samples = _test()
        samples = torch.cat(samples).reshape(len(samples),-1)
        self.plot_hmc_funnel(samples.cpu().numpy())

        # compute the statistics of the ith marginal p(y) w.r.t i
        pv_mean = torch.tensor([0])
        pv_std = torch.tensor([3])
        for i in range(samples.shape[1]):
            sample_mean = samples[:, i].mean()
            sample_std = samples[:, i].std()
            self.assertTensorClose(sample_mean, pv_mean, .3)
            self.assertTensorClose(sample_std, pv_std, .8)
        
        
        
if __name__ == "__main__":
    alf.test.main()
