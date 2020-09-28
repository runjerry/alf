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
from scipy.stats import entropy
import matplotlib.cm as cm


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

        train_iter = 1000
        #for i in range(train_iter):
        #    _train()
        #    if i % 1000 == 0:
        #        samples = _test(i)

        #self.assertLess(mean_err, .3)
        #self.assertLess(pred_err, .3)
        #self.assertLess(cov_err, .3)
        self.plot_hypernetwork_3d(samples.cpu().numpy())
    
    
    def generate_data(self,
        n_samples=100,
        #means=[(1., 1.), (-1., 1.), (1., -1.), (-1., -1.)]):
        means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]):
        data = torch.zeros(n_samples, 2)
        labels = torch.zeros(n_samples)
        size = n_samples//len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size*i:size*(i+1)] = samples
            labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
        
        return data, labels.long()
    
    def plot_classification(self, i, algorithm, conf_style='mean', tag=''):
        #plt.style.use('classic')
        x = torch.linspace(-10, 10, 100)
        y = torch.linspace(-10, 10, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)

        outputs = []
        for _ in range(100):
            output = algorithm.predict_step(grid, particles=100).output.cpu()
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        outputs = F.softmax(outputs, -1).detach()  # [B, D]
        mean_outputs = outputs.mean(1).cpu()  # [B, D]
        std_outputs = outputs.std(1).cpu()

        if conf_style == 'mean':
            conf_outputs = mean_outputs.mean(-1)
        elif conf_style == 'max':
            conf_outputs = mean_outputs.max(-1)[0]
        elif conf_style == 'min':
            conf_outputs = mean_outputs.min(-1)[0]
        elif conf_style == 'entropy':
            print (mean_outputs.shape)
            conf_outputs = entropy(mean_outputs.T.numpy())
        conf_mean = mean_outputs.mean(-1)
        conf_std = std_outputs.max(-1)[0] * 1.94
        labels = mean_outputs.argmax(-1)
        data, _ = self.generate_data(n_samples=400) 
        print (conf_outputs.shape)
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_outputs, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("{} confidance".format(conf_style))
        plt.savefig('minmax_plots/conf_map{}-{}_{}.png'.format(i, conf_style, tag))
        plt.close('all')

        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig('minmax_plots/conf_map{}-std_{}.png'.format(tag, i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig('minmax_plots/conf_map{}-labels_{}.png'.format(tag, i))
        plt.close('all')

    #@parameterized.parameters(('svgd'), ('gfsf'))#, ('minmax'))
    def test_classification_hypernetwork(self, par_vi='minmax', particles=100):
        """
        Using a the log probability function defined above. We want to use this
        to train an HMC chain that evolves towards fitting the target mixture 
        distribution. 
        A Gaussian mixture model with N compnents, has a posterior distribution
        that is also a mixture model, distributed according to:
        p(\theta|x) = \sum^N_{i=1} \phi_i(x) N(\mu_i, \sigma_i), where \phi is 
        a categorical "indexing" distribution. 
        """
        print ('Hypernetwork: Fitting Mixture of 2 Gaussians')
        print ('params: {} - {} particles'.format(par_vi, particles))
        input_size = 2
        output_dim = 4
        batch_size = 100
        noise_dim = 16
        input_spec = TensorSpec((input_size, ), torch.float32)
        train_batch_size = 100
        
        train_nsamples = 100
        test_nsamples = 20
        inputs, targets = self.generate_data(train_nsamples)
        test_inputs, test_targets = self.generate_data(test_nsamples)
        
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((10, True), (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(32, 32),
            loss_type='classification',
            par_vi=par_vi,
            parameterization='layer',
            optimizer=alf.optimizers.Adam(lr=1e-4))
        
        def _train(entropy_regularization=None):
            perm = torch.randperm(train_nsamples)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                particles=particles,
                entropy_regularization=entropy_regularization,
                state=())

            algorithm.update_with_gradient(alg_step.info)
            return (alg_step.info.extra.generator.extra)
        
        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)
            probs = F.softmax(outputs, dim=-1)
            preds = probs.mean(1).cpu().argmax(-1)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)
            
            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, particles).reshape(-1, 1)
            sample_acc = sample_preds.eq(targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum()/len(targets_unrolled)

            print("-" * 68)
            print ('iter ', i)
            print ('MeanOfParticles Acc: ', mean_acc.item())
            print ('AllParticles Acc: ', sample_acc.item())
            with torch.no_grad():
                self.plot_classification(i, algorithm, 'entropy', par_vi)
            return sample_preds, targets_unrolled

        train_iter = 15000
        for i in range(train_iter):
            acc = _train()
            if i % 1000 == 0:
                preds, out_targets = _test(i)
                print ('train acc', acc)

        #self.assertTensorClose(lside_mean, true_lmean, .1)
        #self.assertTensorClose(rside_mean, true_rmean, .1)
        #self.assertTensorClose(lside_std, true_lstd, .3)
        #self.assertTensorClose(rside_std, true_rstd, .3)
 
if __name__ == "__main__":
    alf.test.main()
