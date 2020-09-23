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

import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.hmc_algorithm import HMC
from alf.tensor_specs import TensorSpec

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class HMCTest(alf.test.TestCase):

    def log_prob_1dGaussian(self, data):
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
        return logp_x

    def plot_hmc_1d(self, inputs):
        mean = torch.tensor([0., 0., 0.]).cpu()
        stdev = torch.tensor([0.5, 1.0, 2.0]).cpu()
        x = torch.distributions.MultivariateNormal(
            mean, torch.diag(stdev**2)).sample([inputs.shape[0]])
        hmc_mean = inputs.mean(0)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(inputs[:, 0], inputs[:, 1], s=5, alpha=.4, color='m', label='HMC')
        ax.scatter(x[:, 0], x[:, 1], s=5, alpha=.4, color='b', label='True Samples')
        ax.scatter(hmc_mean[0], hmc_mean[1], marker='*', color='g', s=100, label='HMC Mean')
        ax.scatter(mean[0], mean[1], marker='*', color='C3', s=100, label='True Mean')
        ax.legend(fontsize=17)
        ax.grid()
        ax.set_ylim([-5, 5])
        ax.set_xlim([-5, 5])
        plt.savefig('hmc_fit_1d.png')
        # plt.show()
        plt.close('all')

    def test_1dGaussian_hmc(self):
        """
        Using a the log probability function defined above. We want to use this
        to train an HMC chain that evolves towards fitting the target normal 
        distribution. 
        """
        print ('HMC: Fitting 3d Gaussian')
        num_samples = 200
        step_size = 0.3
        num_steps_per_sample = 10
        params = torch.zeros(3)
        burn_in_steps = 0
        inv_mass = None
        algorithm = HMC(
            self.log_prob_1dGaussian,
            params,
            num_samples=num_samples,
            steps_per_sample=num_steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            inv_mass=inv_mass,
            name='HMCTest')

        def _test():
            samples = algorithm.sample(params, num_samples)
            return samples
        samples = _test()
        samples = torch.cat(samples).reshape(len(samples),-1)
        self.plot_hmc_1d(samples.cpu().numpy())

        sample_mean = samples.mean(0)
        sample_std = samples.std(0)
        true_mean = torch.tensor([0., 0., 0.])
        true_std = torch.tensor([0.5, 1., 2.0])
        
        # for speed. higher ``num_samples`` results in tighter fit
        self.assertTensorClose(sample_mean, true_mean, .1)
        self.assertTensorClose(sample_std, true_std, .3)
    
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
        num_samples = 1000
        step_size = 1.0
        num_steps_per_sample = 25
        params = torch.zeros(2)
        burn_in_steps = 0
        inv_mass = None
        algorithm = HMC(
            self.log_prob_mog2,
            params,
            num_samples=num_samples,
            steps_per_sample=num_steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            inv_mass=inv_mass,
            name='HMCTestMoG')
        
        def _test():
            samples = algorithm.sample(params, num_samples)
            return samples
        samples = _test()
        samples = torch.cat(samples).reshape(len(samples),-1)

        lside_samples = torch.stack([x for x in samples if x[0] < 0])
        rside_samples = torch.stack([x for x in samples if x[0] >= 0])

        lside_mean = lside_samples.mean(0)
        rside_mean = rside_samples.mean(0)
        lside_std = lside_samples.std(0)
        rside_std = rside_samples.std(0)
        true_lmean = torch.tensor([-2.5, 0.])
        true_lstd = torch.tensor([0.5, .5])
        true_rmean = torch.tensor([2.5, 0.])
        true_rstd = torch.tensor([0.5, .5])
        
        self.plot_hmc_mog2(samples.cpu().numpy())
        
        # for speed. higher ``num_samples`` results in tighter fit
        self.assertTensorClose(lside_mean, true_lmean, .1)
        self.assertTensorClose(rside_mean, true_rmean, .1)
        self.assertTensorClose(lside_std, true_lstd, .3)
        self.assertTensorClose(rside_std, true_rstd, .3)
 

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

        def _test():
            return algorithm.sample(params, num_samples)
        samples = _test()
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
