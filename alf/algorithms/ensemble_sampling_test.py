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
from alf.algorithms.sl_algorithm import SLAlgorithm
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

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

    def plot_ensemble_3d(self, inputs):
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
        plt.savefig('ensemble_fit_3d.png')
        # plt.show()
        plt.close('all')
    
    def test_3dGaussian_ensemble(self):
        """
        """
        print ('HyperNetwork: Fitting 3d Gaussian')
        input_size = 2
        output_dim = 1
        batch_size = 100
        input_spec = TensorSpec((input_size, ), torch.float32)
        train_batch_size = 50
        algorithm = SLAlgorithm(
            input_spec,
            fc_layer_params=((50, True),(50, True),),
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            predictor_size=10,
            loss_type='regression',
            optimizer=alf.optimizers.Adam(1e-4))

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
            
            train_inputs_ = train_inputs[:, :2]
            train_targets = train_inputs[:, 2:]
            alg_step = algorithm.train_step(
                inputs=(train_inputs_, train_targets),
                state=())

            algorithm.update_with_gradient(alg_step.info)
        
        def _test(i):
            test_samples = dist.sample([100])
            test_inputs = test_samples[:, :2] 
            test_targets = test_samples[:, 2:] 

            sample_outputs, _ = algorithm._net(test_inputs)
            sample_outputs = sample_outputs.transpose(0, 1)
            computed_mean = sample_outputs.mean(0)
            
            computed_cov = self.cov(sample_outputs.reshape(10, -1))
            computed_cov_data = self.cov(
                test_targets.unsqueeze(0).repeat(10, 1, 1).reshape(
                    10, -1))

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

        train_iter = 3000
        for i in range(train_iter):
            _train()
            if i % 1000 == 0:
                samples = _test(i)

        #self.assertLess(mean_err, .3)
        #self.assertLess(pred_err, .3)
        #self.assertLess(cov_err, .3)
        self.plot_ensemble_3d(samples.cpu().numpy())
    
    
    def generate_data(self,
        n_samples=100,
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
        os.makedirs('plots/{}'.format(tag), exist_ok=True)
        basedir = 'plots/{}'.format(tag)
        x = torch.linspace(-10, 10, 100)
        y = torch.linspace(-10, 10, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)

        outputs, _ = algorithm._net(grid)  # [B, N, D]
        mean_outputs = F.softmax(outputs, -1).mean(1).detach().cpu()  # [B, D]
        std_outputs = F.softmax(outputs, -1).std(1).detach().cpu()  # [B, D]
        if conf_style == 'mean':
            conf_outputs = mean_outputs.mean(-1)
        elif conf_style == 'max':
            conf_outputs = mean_outputs.max(-1)[0]
        elif conf_style == 'min':
            conf_outputs = mean_outputs.min(-1)[0]
        elif conf_style == 'entropy':
            conf_outputs = entropy(mean_outputs.T.numpy())
        conf_mean = mean_outputs.mean(-1)
        conf_std = std_outputs.max(-1)[0]
        labels = mean_outputs.argmax(-1)
        data, _ = self.generate_data(n_samples=400) 
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_outputs, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("{}".format(conf_style))
        plt.savefig(basedir+'/conf_map-{}_{}.png'.format(conf_style, i))
        plt.close('all')

        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig(basedir+'/conf_map-std_{}.png'.format(i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels, cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig(basedir+'/conf_map-labels_{}.png'.format(i))
        plt.close('all')

    def test_classification_ensemble(self):
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
        print ('params: {} member ensemble'.format(10))
        input_size = 2
        output_dim = 4
        batch_size = 100
        input_spec = TensorSpec((input_size, ), torch.float32)
        train_batch_size = 100
        n_members = 100
        
        inputs, targets = self.generate_data(n_samples=100)
        test_inputs, test_targets = self.generate_data(n_samples=20)
        algorithm = SLAlgorithm(
            input_spec,
            fc_layer_params=(10, 10,),
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            predictor_size=n_members,
            predictor_vote='soft',
            use_fc_bn=False,
            loss_type='classification',
            optimizer=alf.optimizers.Adam(1e-4))

        def _train():
            perm = torch.randperm(100)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                state=())

            algorithm.update_with_gradient(alg_step.info)
        
        def _test(i):
            outputs, _ = algorithm._net(test_inputs)
            probs = F.softmax(outputs, dim=-1)
            print (probs.shape)
            preds = probs.mean(1).cpu().argmax(-1)
            print (preds.shape, test_targets.shape)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)
            
            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, n_members).reshape(-1, 1)
            sample_acc = sample_preds.eq(targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum()/len(targets_unrolled)

            import umap
            _params_lst = list(algorithm._net.parameters())
            _params = []
            for entry in _params_lst:
                flat_entry = entry.view(n_members, -1)
                _params.append(flat_entry)
            _params = torch.cat(_params, dim=-1).detach().cpu().numpy()
            print (_params.shape)
            trans = umap.UMAP(n_neighbors=5, random_state=42).fit(_params)
            X = trans.embedding_
            plt.scatter(X[:, 0], X[:, 1], label='umap embedding')
            plt.savefig('plots/umap_ensemble_params')
            plt.close('all')
            print("-" * 68)
            print ('iter ', i)
            print ('Mean Acc: ', mean_acc.item())
            print ('Sample Acc: ', sample_acc.item())
            self.plot_classification(i, algorithm, 'entropy', 'ensemble')
            return sample_preds, targets_unrolled

        train_iter = 4000
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
