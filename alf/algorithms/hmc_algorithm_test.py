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
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.algorithms.hmc_algorithm import HMC
from alf.tensor_specs import TensorSpec

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import matplotlib.cm as cm


class BNN(nn.Module):
    def __init__(self, n_hidden):
        super(BNN, self).__init__()
        self.n_hiden = n_hidden
        self.layers = []
        self.linear1 = nn.Linear(n_hidden[0], n_hidden[1], bias=True)
        self.linear2 = nn.Linear(n_hidden[1], n_hidden[2], bias=True)
        self.linear3 = nn.Linear(n_hidden[2], n_hidden[3], bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
 

class BNN2(nn.Module):
    def __init__(self, n_hidden):
        super(BNN2, self).__init__()
        self.n_hiden = n_hidden
        self.layers = []
        self.linear1 = nn.Linear(n_hidden[0], n_hidden[1], bias=False)

    def forward(self, x):
        return self.linear1(x)
              
class BNN3(nn.Module):
    def __init__(self, n_hidden):
        super(BNN3, self).__init__()
        self.n_hiden = n_hidden
        self.layers = []
        self.linear1 = nn.Linear(n_hidden[0], n_hidden[1], bias=True)
        self.linear2 = nn.Linear(n_hidden[1], n_hidden[2], bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
 
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
    
    def log_prob_funnel(self, data):
        v_dist = torch.distributions.Normal(0, 3)
        logp = v_dist.log_prob(data[0])
        x_dist = torch.distributions.Normal(0, torch.exp(-data[0])**.5)
        logp += x_dist.log_prob(data[1:]).sum()
        return logp

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
        #samples = _test()
        #samples = torch.cat(samples).reshape(len(samples),-1)
        #self.plot_hmc_funnel(samples.cpu().numpy())

        # compute the statistics of the ith marginal p(y) w.r.t i
        #pv_mean = torch.tensor([0])
        #pv_std = torch.tensor([3])
        #for i in range(samples.shape[1]):
        #    sample_mean = samples[:, i].mean()
        #    sample_std = samples[:, i].std()
        #    self.assertTensorClose(sample_mean, pv_mean, .3)
        #    self.assertTensorClose(sample_std, pv_std, .8)
        
        
    def generate_regression_data(self, n_train, n_test):
        x_train1 = torch.linspace(-6, -2, n_train//2).view(-1, 1)
        x_train2 = torch.linspace(2, 6, n_train//2).view(-1, 1)
        x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
        x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
        y_train = -(1 + x_train) * torch.sin(1.2*x_train) 
        y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

        x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
        y_test = -(1 + x_test) * torch.sin(1.2*x_test) 
        y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
        return (x_train, y_train), (x_test, y_test)
    
    def plot_bnn_regression(self, bnn_preds, data):
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1+gt_x) * torch.sin(1.2*gt_x) 
        gt_y += torch.ones_like(gt_x).normal_(0, 0.04).cpu()
        (x_train, y_train), (x_test, y_test) = data
        x_test = x_test.cpu().numpy()
        x_train = x_train.cpu().numpy()
        bnn_preds = bnn_preds.cpu()
        print (x_test.shape, bnn_preds.shape)
        plt.plot(x_test, bnn_preds[:].numpy().squeeze().T,
            'C0',alpha=0.01)
        plt.plot(x_test, bnn_preds.mean(0).squeeze().T, color='b',
            label='posterior mean', alpha=0.9)
        plt.plot(x_test,
            bnn_preds.mean(0).squeeze().T+2*bnn_preds.std(0).squeeze().T,
            'C1',alpha=0.8, linewidth=3)
        plt.plot(x_test,
            bnn_preds.mean(0).squeeze().T-2*bnn_preds.std(0).squeeze().T,
            'C1',alpha=0.8,linewidth=3)
        plt.scatter(x_train, y_train.cpu().numpy(),color='g', label='train pts',
            alpha=0.6)
        plt.plot(gt_x, gt_y, color='r', label='ground truth')
        plt.legend(fontsize=14, loc='best')
        plt.ylim([-6, 8])
        plt.savefig('plots/hmc_bnn.png')
        plt.close('all')

    def test_BayesianNNRegression(self):
        n_train = 80
        n_test = 200
        train_samples, test_samples = self.generate_regression_data(
            n_train, n_test)
        net = BNN3([1, 50, 1])
        params_init = torch.cat([
            p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = .1
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = 0.002
        num_samples = 10000
        burn_in_steps= 9800
        steps_per_sample = 25
        tau_out = .1
        print ('HMC: Fitting BNN to regression data')
        algorithm = HMC(
            params=params_init,
            num_samples=num_samples,
            steps_per_sample=steps_per_sample,
            step_size=step_size,
            model=net,
            burn_in_steps=burn_in_steps,
            model_loss='regression',
            tau_list=tau_list,
            tau_out=tau_out)

        def _train():
            train_data, train_labels = train_samples
            params_hmc = algorithm.sample_model(train_data, train_labels)
            return params_hmc

        def _test(hmc_params):
            test_data, test_labels = test_samples
            preds, log_probs = algorithm.predict_model(test_data, test_labels,
                samples=hmc_params)
            print ('Expected test log probability: {}'.format(torch.stack(
                log_probs).mean()))
            print ('Expected MSE: {}'.format(
                ((preds.mean(0) - test_labels)**2).mean()))
            return preds

        #hmc_params = _train()
        #bnn_preds = _test(hmc_params)
        #self.plot_bnn_regression(bnn_preds, (train_samples, test_samples))
    
    def generate_class_data(self,
        n_samples=100,
        means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]):
        #means=[(2., 2.), (-2., -2.)]):
        data = torch.zeros(n_samples, 2)
        labels = torch.zeros(n_samples)
        size = n_samples//len(means)
        for i, (x, y) in enumerate(means):
            dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
            samples = dist.sample([size])
            data[size*i:size*(i+1)] = samples
            labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
       
        plt.scatter(data[:, 0].cpu(), data[:, 1].cpu())
        plt.savefig('data_space.png')
        plt.close('all')
        return data, labels.long()
    
    def plot_bnn_classification(self, i, algorithm, samples, conf_style='mean',
        tag='restars'):
        x = torch.linspace(-10, 10, 100)
        y = torch.linspace(-10, 10, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
        outputs, _ = algorithm.predict_model(grid, y=None, samples=samples)
        print (outputs.shape)
        outputs = F.softmax(outputs, dim=-1)  # [B, D]
        mean_outputs = outputs.mean(0).cpu()
        std_outputs = outputs.std(0).cpu()

        if conf_style == 'mean':
            conf_outputs = mean_outputs.mean(-1)
        elif conf_style == 'max':
            conf_outputs = mean_outputs.max(-1)[0]
        elif conf_style == 'min':
            conf_outputs = mean_outputs.min(-1)[0]
        elif conf_style == 'entropy':
            conf_outputs = entropy(mean_outputs.T.numpy())
        
        conf_mean = mean_outputs.mean(-1)
        conf_std = std_outputs.max(-1)[0] * 1.96
        labels = mean_outputs.argmax(-1)
        data, _ = self.generate_class_data(n_samples=400) 
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_outputs, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("{} confidance".format(conf_style))
        plt.savefig('plots/conf_map{}-{}_{}.png'.format(i, conf_style, tag))
        plt.close('all')

        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig('plots/conf_map{}-std_{}.png'.format(tag, i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig('plots/conf_map{}-labels_{}.png'.format(tag, i))
        plt.close('all')

    def test_BayesianNNClassification(self):
        n_train = 100
        n_test = 20
        inputs, targets = self.generate_class_data(n_train)
        net = BNN([2, 10, 10, 4])
        params_init = torch.cat([p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = 1.
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = .005
        num_samples = 10000
        steps_per_sample = 25
        tau_out = 1.
        burn_in_steps= 9800
        print ('HMC: Fitting BNN to classification data')
        algorithm = HMC(
            params=params_init,
            num_samples=num_samples,
            steps_per_sample=steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            model=net,
            model_loss='classification',
            tau_list=tau_list,
            tau_out=tau_out)

        def _train():
            params_hmc = algorithm.sample_model(inputs, targets)
            return params_hmc

        def _test(hmc_params):
            test_data, test_labels = self.generate_class_data(n_test)
            preds, log_probs = algorithm.predict_model(test_data, test_labels,
                samples=hmc_params)
            print ('Expected test log probability: {}'.format(torch.stack(
                log_probs).mean()))
            print ('Expected XE loss: {}'.format(
                F.cross_entropy(preds.mean(0), test_labels).mean()))
            return preds
        
        for i in range(89, 90):
            #hmc_params = _train()
            #from sklearn.manifold import MDS
            #mds = MDS(n_components=2)
            import glob
            hmc_data = []
            paths = glob.glob('hmc_runs/*')
            for path in paths:
                arr = np.load(path)
                hmc_data.append(torch.from_numpy(arr))
            hmc_params = torch.stack(hmc_data)[:, ::100, :].reshape(-1, 184).cuda()

            #_params = torch.stack(hmc_params).detach().cpu().numpy()
            #_parmas = _params[::25]
            #print (_params.shape)
            #mds.fit(_params)
            #X = mds.fit_transform(_params)
            #plt.scatter(X[:, 0], X[:, 1], label='mds points')
            #plt.savefig('plots/mds_plot_hmc_small.png')
            #plt.close('all')
            #np.save('hmc_runs_2cls21/hmc_params_run_{}.npy'.format(i), _params)
            bnn_preds = _test(hmc_params)
            print ('plotting')
            with torch.no_grad():
                self.plot_bnn_classification(num_samples, algorithm, hmc_params,
                'entropy', 'hmc_2means_l50_cmap')
        
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

    def test_bayesian_linear_regression(self):
        """
        HMC is used to sample the parameter vector for a linear
        regressor. The target linear regressor is :math:`y = X\beta + e`, where 
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix, 
        and :math:`y` is target ouputs. The posterior of :math:`\beta` has a 
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        For a linear model with weight W and bias b, and standard Gaussian prior,
        the output follows a Gaussian :math:`N(b, WW^T)`, which should 
        match the posterior :math:`p(\beta|X,y)`.
        """
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_dim = 1
        batch_size = 100
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_dim) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_dim)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(
            inputs.t() @ inputs) 
        true_mean = true_cov @ inputs.t() @ targets
        net = BNN2([input_size, output_dim])
        params_init = torch.cat([p.flatten() for p in net.parameters()]).clone()
        tau_list = []
        tau = 1.
        for p in net.parameters():
            tau_list.append(tau)
        tau_list = torch.tensor(tau_list)
        step_size = .005
        num_samples = 6000
        steps_per_sample = 50
        tau_out = 1.
        burn_in_steps= 5800
        print ('HMC: Fitting BNN to classification data')
        algorithm = HMC(
            params=params_init,
            num_samples=num_samples,
            steps_per_sample=steps_per_sample,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            model=net,
            model_loss='regression',
            tau_list=tau_list,
            tau_out=tau_out)

        def _train():
            params_hmc = algorithm.sample_model(inputs, targets)
            return params_hmc

        def _test(params):
            print("-" * 68)
            preds, log_probs = algorithm.predict_model(inputs, targets,
                samples=params)
            params = torch.stack(params)
            computed_mean = params.mean(0)
            computed_cov = self.cov(params)
            print("-" * 68)
            spred_err = torch.norm((preds - targets).mean(1))
            print("sampled pred err: ", spred_err)

            smean_err = torch.norm(computed_mean - true_mean.squeeze())
            smean_err = smean_err / torch.norm(true_mean)
            print("sampled mean err: ", smean_err)

            computed_cov = self.cov(params)
            scov_err = torch.norm(computed_cov - true_cov)
            scov_err = scov_err / torch.norm(true_cov)
            print("sampled cov err: ", scov_err)
            
            self.assertLess(smean_err, .5)
            self.assertLess(scov_err, .5)

        #params_hmc = _train()
        #_test(params_hmc)

        #print("ground truth mean: {}".format(true_mean))
        #print("ground truth cov norm: {}".format(true_cov.norm()))

if __name__ == "__main__":
    alf.test.main()
