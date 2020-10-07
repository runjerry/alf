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

import os

class HyperNetworkSampleTest(parameterized.TestCase, alf.test.TestCase):
    """ 
    HyperNetwork Sample Test
        Two tests given in order of increasing difficulty. 
        1. A 3-dimensional multivariate Gaussian distribution 
        2. A 4 class classification problem, where the classes are distributed
            as 4 symmetric Normal distributions with non overlapping support. 
            The hypernetwork is trained to sample classification functions that
            fit the data, for the purpose of observing the predictive
            distributions of sampled funcitons on data outside the training
            distribution. 
        
        We do not directly compute the posterior for any of these distributions
        We instead determine closeness qualitatively by sampling predictors from
        our hypernetwork, and comparing the resulting prediction statistics to 
        samples drawn from an HMC-based neural network. 
    """
    
    def generate_class_data(self, n_samples=100,
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
    
    def plot_classification(self, i, algorithm, conf_style='entropy', tag=''):
        os.makedirs('plots/{}'.format(tag), exist_ok=True)
        basedir = 'plots/{}'.format(tag)
        x = torch.linspace(-10, 10, 100)
        y = torch.linspace(-10, 10, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)

        outputs = []
        for _ in range(100):
            output = algorithm.predict_step(grid, num_particles=100).output.cpu()
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        outputs = F.softmax(outputs, -1).detach()  # [B, D]
        mean_outputs = outputs.mean(1).cpu()  # [B, D]
        std_outputs = outputs.std(1).cpu()
        conf_outputs = entropy(mean_outputs.T.numpy())
        conf_mean = mean_outputs.mean(-1)
        conf_std = std_outputs.max(-1)[0] * 1.94
        labels = mean_outputs.argmax(-1)
        data, _ = self.generate_class_data(n_samples=400) 
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_outputs,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("{}".format(conf_style))
        plt.savefig(basedir+'/conf_map-{}_{}.png'.format(conf_style, i))
        plt.close('all')

        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig(basedir+'/conf_map-std_{}.png'.format(i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black')
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig(basedir+'/conf_map-labels_{}.png'.format(i))
        plt.close('all')

    @parameterized.parameters(#('svgd'), ('gfsf'), ('minmax'),
                              ('svgd', True), ('gfsf', True), ('minmax', True))
    def test_classification_hypernetwork(self,
                                         par_vi='minmax',
                                         function_vi=False,
                                         particles=100):
        """
        Symmetric 4-class classification problem. The training data are drawn
        from standard normal distributions, each class is given by one of
        these distributions. The hypernetwork is trained to generate parameters
        that achieves low loss / high accuracy on this data.
        """
        print ('Hypernetwork: Fitting 4 Classes')
        print ('params: {} - {} particles'.format(par_vi, particles))
        input_size = 2
        output_dim = 4
        batch_size = 100
        noise_dim = 32
        parameterization = 'layer'
        input_spec = TensorSpec((input_size, ), torch.float32)
        train_batch_size = 100
        
        train_nsamples = 100
        test_nsamples = 20
        inputs, targets = self.generate_class_data(train_nsamples)
        test_inputs, test_targets = self.generate_class_data(test_nsamples)
        
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((10, True), (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(16, 16),
            loss_type='classification',
            par_vi=par_vi,
            function_vi=function_vi,
            function_space_samples=20,
            function_bs=train_batch_size,
            num_particles=particles,
            parameterization=parameterization,
            optimizer=alf.optimizers.Adam(lr=1e-4),
            critic_optimizer=alf.optimizers.Adam(1e-4),
            critic_hidden_layers=(100, 100),
            critic_fc_bn=True,
            critic_train_iters=5)
        
        def _train(entropy_regularization=None):
            perm = torch.randperm(train_nsamples)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
            
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                state=())

            algorithm.update_with_gradient(alg_step.info)
            return (alg_step.info.extra.generator.extra)
        
        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)

            params = algorithm.sample_parameters(num_particles=200)
            _params = params.detach().cpu().numpy()
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(_params)
            X = pca.transform(_params)
            #trans = umap.UMAP(n_neighbors=5, random_state=42).fit(_params)
            #X = trans.embedding_
            plt.scatter(X[:, 0], X[:, 1], label='pca embeddeding')
            plt.savefig('plots/pca_svgd_fvi_hypernetwork_clf.png')
            plt.close('all')

            probs = F.softmax(outputs, dim=-1)
            preds = probs.mean(1).cpu().argmax(-1)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)
            
            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, particles).reshape(-1, 1)
            
            sample_acc = sample_preds.eq(targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum()/len(targets_unrolled)

            print ('-'*86)
            print ('iter ', i)
            print ('mean particle acc: ', mean_acc.item())
            print ('all particles acc: ', sample_acc.item())

            with torch.no_grad():
                self.plot_classification(i, algorithm, 'entropy', par_vi+'_fvi')
            return sample_preds, targets_unrolled

        train_iter = 8000
        for i in range(train_iter):
            acc = _train()
            if i % 1000 == 0:
                print ('train acc', acc.item()*100.)
                preds, out_targets = _test(i)

if __name__ == "__main__":
    alf.test.main()
