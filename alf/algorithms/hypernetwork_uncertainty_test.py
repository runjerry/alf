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
import seaborn as sns

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
    
    def generate_class_data(self, n_samples=200,
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
        
        return data, labels.long()
    
    def plot_classification(self, i, algorithm, conf_style='entropy', tag='',
        amortize='', fvi='', sub=''):
        basedir = 'plots/{}'.format(tag)
        if not amortize:
            basedir += '/ensemble'
        else:
            basedir += '/functional'
        if fvi:
            basedir += '/fvi'
        basedir += '_{}'.format(sub)
        os.makedirs(basedir, exist_ok=True)
        x = torch.linspace(-12, 12, 100)
        y = torch.linspace(-12, 12, 100)
        gridx, gridy = torch.meshgrid(x, y)
        grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
        
        if 'ensemble' in basedir:
            outputs = algorithm.predict_step(grid).output.cpu()
        else:
            noise_d = torch.distributions.Normal(torch.tensor([0]), torch.tensor([1.]))
            noise = noise_d.icdf(torch.linspace(1e-3, 1-1e-3, 512))
            params = algorithm.sample_parameters(noise=noise, num_particles=512)
            outputs = algorithm.predict_step(grid, params=params).output.cpu()
            #outputs = algorithm.predict_step(grid, params=paramsnum_particles=512).output.cpu()
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
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("{}".format(conf_style))
        plt.savefig(basedir+'/conf_map-{}_{}.png'.format(conf_style, i))
        plt.close('all')

        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("confidance (std)")
        plt.savefig(basedir+'/conf_map-std_{}.png'.format(i))
        plt.close('all')
        
        p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=labels,
            cmap='rainbow')
        p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
        cbar = plt.colorbar(p1)
        cbar.set_label("predicted labels")
        plt.savefig(basedir+'/conf_map-labels_{}.png'.format(i))
        print ('saved figure: ', basedir)
        plt.close('all')

    @parameterized.parameters(#('svgd3', False, False), ('svgd3', False, True),
                              ('minmax', True, True, False, 6, 6),
    )
    def test_classification_hypernetwork(self,
                                         par_vi='minmax',
                                         amortize=True,
                                         functional_gradient=False,
                                         function_vi=False,
                                         hidden_layers=10,
                                         noise_dim=16,
                                         num_particles=100):
        """
        Symmetric 4-class classification problem. The training data are drawn
        from standard normal distributions, each class is given by one of
        these distributions. The hypernetwork is trained to generate parameters
        that achieves low loss / high accuracy on this data.
        """
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        torch.set_default_dtype(torch.float64)


        print ('Hypernetwork: Fitting 4 Classes')
        print ('params: {} - {} particles'.format(par_vi, num_particles))
        print ("Amortize = {}, function_vi = {}".format(amortize, function_vi))
        input_size = 2
        output_dim = 4
        amortize = True
        functional_gradient = True
        parameterization = 'network'
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_batch_size = 100
        
        train_nsamples = 100
        test_nsamples = 200
        batch_size = train_nsamples
        inputs, targets = self.generate_class_data(train_nsamples)
        test_inputs, test_targets = self.generate_class_data(test_nsamples)
        noise_dim = 32
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((10, True), (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(32,),
            use_relu_mlp=False,
            loss_type='classification',
            par_vi=par_vi,
            amortize_vi=amortize,
            functional_gradient=functional_gradient,
            use_pinverse=True,
            pinverse_batch_size=num_particles,
            particle_optimizer=alf.optimizers.Adam(lr=1e-2),
            function_vi=function_vi,
            function_space_samples=95,
            function_bs=train_batch_size,
            num_particles=num_particles,
            parameterization=parameterization,
            optimizer=alf.optimizers.Adam(lr=1e-3),#, weight_decay=1e-4),
            critic_hidden_layers=(100, 100),
            critic_optimizer=alf.optimizers.Adam(lr=1e-3))
        
        def _train(i, entropy_regularization=None):
            perm = torch.randperm(train_nsamples)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
            
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles,
                state=())
            if functional_gradient:
                pinverse_loss = alg_step.info.extra.pinverse
                if i % 500 == 0: print ('pl', pinverse_loss)
            if amortize or function_vi:
                loss_info, params = algorithm.update_with_gradient(alg_step.info)
            else:
                update_direction = alg_step.info.loss
                algorithm._particle_optimizer.zero_grad()
                algorithm._params.grad = update_direction
                algorithm._particle_optimizer.step()

        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)

            params = algorithm.sample_parameters(num_particles=200)
            if functional_gradient:
                params = params[0]
            _params = params.detach().cpu().numpy()

            import os
            basedir = 'plots/{}'.format(par_vi)
            if not amortize:
                basedir += '/ensemble'
            if function_vi:
                basedir += '/fvi'
            if functional_gradient:
                basedir += '/functional'
            os.makedirs(basedir, exist_ok=True)
            np.save(basedir+'/params.npy', _params)

            probs = F.softmax(outputs, dim=-1)
            preds = probs.mean(1).cpu().argmax(-1)
            mean_acc = preds.eq(test_targets.cpu().view_as(preds)).float()
            mean_acc = mean_acc.sum() / len(test_targets)
            
            sample_preds = probs.cpu().argmax(-1).reshape(-1, 1)
            targets_unrolled = test_targets.unsqueeze(1).repeat(
                1, num_particles).reshape(-1, 1)
            
            sample_acc = sample_preds.eq(targets_unrolled.cpu().view_as(sample_preds)).float()
            sample_acc = sample_acc.sum()/len(targets_unrolled)

            print ('-'*86)
            print ('iter ', i)
            print ('mean particle acc: ', mean_acc.item())
            print ('all particles acc: ', sample_acc.item())

            with torch.no_grad():
                sub = '4cls_64z_2h64_net10zz2_25lr_ad1e3_1iter_.1addjac_3_512test'
                #sub = '4cls_64z_2h64_1inverse_ad1e2_l1'
                self.plot_classification(i, algorithm, 'entropy', par_vi,
                    amortize, function_vi, sub)
            return sample_preds, targets_unrolled
        
        
        train_iter = 500000
        for i in range(train_iter):
            _train(i)
            if i % 2000 == 0:
                preds, out_targets = _test(i)
        
    
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
    
    def plot_bnn_regression(self, i, algorithm, function_vi, data):

        sns.set_style('darkgrid')
        gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
        gt_y = -(1+gt_x) * torch.sin(1.2*gt_x) 
        #gt_y += torch.ones_like(gt_x).normal_(0, 0.04).cpu()
        (x_train, y_train), (x_test, y_test) = data
        outputs = algorithm.predict_step(x_test, num_particles=256).output.cpu()
        mean = outputs.mean(1).squeeze()
        std = outputs.std(1).squeeze()
        x_test = x_test.cpu().numpy()
        x_train = x_train.cpu().numpy()
        print (x_test.shape, outputs.shape, mean.shape, std.shape)

        plt.fill_between(x_test.squeeze(), mean.T+2*std.T, mean.T-2*std.T, alpha=0.5)
        plt.plot(gt_x, gt_y, color='red', label='ground truth')
        plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
        plt.scatter(x_train, y_train.cpu().numpy(),color='r', marker='+',
            label='train pts', alpha=1.0, s=50)
        plt.legend(fontsize=14, loc='best')
        #plt.ylim([-6, 8])
        plt.savefig('plots/fsvgd_bnn_{}.png'.format(i))
        plt.close('all')
    
    def test_BayesianNNRegression(self):
        n_train = 80
        n_test = 200
        input_size = 1
        output_dim = 1
        noise_dim = 151
        num_particles = 2
        amortize = True
        function_vi = False
        functional_gradient =False
        parameterization = 'network'
        input_spec = TensorSpec((input_size, ), torch.float64)
        train_batch_size = n_train
        batch_size = n_train

        train_samples, test_samples = self.generate_regression_data(
            n_train, n_test)
        inputs, targets = train_samples
        test_inputs, test_targets = test_samples
        print ('Fitting BNN to regression data')
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=((50, True),),#, (10, True)),
            last_layer_param=(output_dim, True),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            hidden_layers=(151,),
            use_relu_mlp=False,
            loss_type='regression',
            par_vi='svgd3',
            amortize_vi=amortize,
            functional_gradient=functional_gradient,
            use_pinverse=True,
            pinverse_resolve=False,
            pinverse_solve_iters=1,
            use_jac_regularization=False,
            pinverse_batch_size=num_particles,
            particle_optimizer=alf.optimizers.Adam(lr=5e-4),
            function_vi=function_vi,
            function_space_samples=95,
            function_bs=train_batch_size,
            num_particles=num_particles,
            parameterization=parameterization,
            optimizer=alf.optimizers.Adam(lr=1e-3),
            critic_hidden_layers=(32,32),
            critic_optimizer=alf.optimizers.Adam(lr=1e-3))
        
        def _train(entropy_regularization=None):
            perm = torch.randperm(n_train)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            if entropy_regularization is None:
                entropy_regularization = train_batch_size / batch_size
            
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=entropy_regularization,
                num_particles=num_particles,
                state=())
            if amortize or function_vi:
                loss_info, params = algorithm.update_with_gradient(alg_step.info)
            else:
                update_direction = alg_step.info.loss
                algorithm._particle_optimizer.zero_grad()
                algorithm._params.grad = update_direction
                algorithm._particle_optimizer.step()

        def _test(i):
            outputs, _ = algorithm._param_net(test_inputs)
            mse_err = (outputs.mean(1) - test_targets).pow(2).mean()
            print ('Expected MSE: {}'.format(mse_err))
        """
        for i in range(20000):
            _train()
            if i % 1000 == 0:
                _test(i)
                with torch.no_grad():
                    data = (train_samples, test_samples)
                    self.plot_bnn_regression(i, algorithm, function_vi, data)
        with torch.no_grad():
            data = (train_samples, test_samples)
            self.plot_bnn_regression(i, algorithm, function_vi, data)
        """
        
if __name__ == "__main__":
    alf.test.main()
