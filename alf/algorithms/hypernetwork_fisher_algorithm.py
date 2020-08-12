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

from absl import logging
import gin
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable

import alf
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.algorithms.hypernetwork_layer_generator import ParamLayers
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.summary_utils import record_time

HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


@gin.configurable
class HyperNetworkFisher(Generator):
    """HyperNetwork 

    HyperrNetwork is a generator that generates a set of parameters for a predefined
    neural network from a random noise input. It is based on the following work:

    https://github.com/neale/HyperGAN

    Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, 
    Performant Neural Networks." International Conference on Machine Learning. 2019.

    Major differences versus the original paper are:

        * A single genrator that generates parameters for all network layers.

        * Remove the mixer and the distriminator.

        * Amortized particle-based variational inference (ParVI) to train the hypernetwork,
          in particular, two ParVI methods are implemented:

            1. amortized Stein Variational Gradient Descent (SVGD):

            Feng et al "Learning to Draw Samples with Amortized Stein Variational
            Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

            2. amortized Wasserstein ParVI with Smooth Functions (GFSF):

            Liu, Chang, et al. "Understanding and accelerating particle-based 
            variational inference." International Conference on Machine Learning. 2019.

    """

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 noise_dim=32,
                 hidden_layers=(64, 64),
                 use_fc_bn=False,
                 particles=32,
                 entropy_regularization=1.,
                 kernel_sharpness=1.,
                 d_iters=5,
                 g_iters=1,
                 loss_type="classification",
                 loss_func: Callable = None,
                 voting="soft",
                 optimizer=None,
                 regenerate_for_each_batch=True,
                 parameterization="layer",
                 print_network=True,
                 name="HyperNetwork"):
        """
        Args:
            Args for the generated parametric network
            ====================================================================
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            parameterization (str): choice of parameterization for the 
                hypernetwork. Choices are [``network``, ``layer``].
                a parameterization of ``network`` uses a single generator to 
                generate all the weights at once. A parameterization of
                ``layer`` uses one generator of each layer of output parameters

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization
            kernel_sharpness (float): Used only for entropy_regularization > 0.
                We calcualte the kernel in SVGD as:
                    :math:`\exp(-kernel_sharpness * reduce_mean(\frac{(x-y)^2}{width}))`
                where width is the elementwise moving average of :math:`(x-y)^2`

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            loss_func (Callable): loss_func(outputs, targets)   
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            regenerate_for_each_batch (bool): If True, particles will be regenerated 
                for every training batch.
            print_network (bool): whether print out the archetectures of networks.
            name (str):
        """
        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))
    
        self._parameterization = parameterization
        assert self._parameterization in ['network', 'layer'], "Hypernetwork " \
                "can only be parameterized by \"network\" or \"layer\" " \
                "generators"
        if self._parameterization == 'network':
            net = EncodingNetwork(
                noise_spec,
                fc_layer_params=hidden_layers,
                use_fc_bn=use_fc_bn,
                last_layer_size=gen_output_dim,
                last_activation=math_ops.identity,
                name="Generator")
        else:
            net = ParamLayers(
                noise_dim=noise_dim,
                particles=particles,
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                last_layer_param=last_layer_param,
                last_activation=math_ops.identity,
                hidden_layers=hidden_layers,
                activation=activation,
                use_fc_bn=use_fc_bn,
                optimizer=optimizer,
                name="Generator")
            optimizer=net.default_optimizer

        disc_fc_params = (256, 256)
        print (gen_output_dim)
        disc_net = EncodingNetwork(
                TensorSpec(shape=(gen_output_dim, )),
                conv_layer_params=None,
                fc_layer_params=disc_fc_params,
                activation=torch.nn.functional.relu,
                last_layer_size=gen_output_dim,
                last_activation=math_ops.identity,
                name="Critic")
        # disc_net.apply(self._spectral_norm)
        

        if print_network:
            print("Generated network")
            print("-" * 68)
            print(param_net)

            print("Generator network")
            print("-" * 68)
            if parameterization == 'network':
                print(net)
            else:
                net.print_hypernetwork_layers()
            print ("Critic network")
            print("-" * 68)
            print (disc_net)

        super().__init__(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            kernel_sharpness=kernel_sharpness,
            optimizer=optimizer,
            name=name)
        
        self._param_net = param_net
        self._disc_net = disc_net
        self._particles = particles
        self._train_loader = None
        self._test_loader = None
        self._regenerate_for_each_batch = regenerate_for_each_batch
        self._loss_func = loss_func
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._d_iters = d_iters
        self._g_iters = g_iters

        self._noise = None
        self._noise_dim = noise_dim
        
        assert (voting in ['soft', 'hard'
                           ]), ("voting only supports \"soft\" and \"hard\"")
        self._voting = voting
        if loss_type == 'classification':
            self._compute_loss = self._classification_loss
            self._vote = self._classification_vote
            if self._loss_func is None:
                self._loss_func = F.cross_entropy
        elif loss_type == 'regression':
            self._compute_loss = self._regression_loss
            self._vote = self._regression_vote
            if self._loss_func is None:
                self._loss_func = F.mse_loss
        else:
            assert ValueError(
                "loss_type only supports \"classification\" and \"regression\""
            )

    def set_data_loader(self, train_loader, test_loader=None):
        """Set data loadder for training and testing."""
        self._train_loader = train_loader
        self._test_loader = test_loader

    def set_particles(self, particles):
        """Set the number of particles to sample through one forward
        pass of the hypernetwork. """
        self._particles = particles

    @property
    def particles(self):
        return self._particles

    def sample_parameters(self, noise=None, particles=None, training=True):
        "Sample parameters for an ensemble of networks." ""
        if noise is None and particles is None:
            particles = self.particles
        params, _ = self._predict(
            inputs=None, noise=noise, batch_size=particles, training=training)
        return params

    def predict(self, inputs, params=None, particles=None):
        """Predict ensemble outputs for inputs using the hypernetwork model."""
        if params is None:
            params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return outputs

    def train_iter(self, particles=None, state=None):
        """Perform one epoch (iteration) of training."""

        assert self._train_loader is not None, "Must set data_loader first."

        with record_time("time/train"):
            loss = 0.
            if self._loss_type == 'classification':
                avg_acc = []
            params = None
            if not self._regenerate_for_each_batch:
                params = self.sample_parameters(particles=particles)

            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())

                for p in self._net.parameters():
                    p.requires_grad = True

                if batch_idx % (self._d_iters+1):
                    model = 'critic'
                    params = self.sample_parameters(self._noise, self._particles)
                else:
                    model = 'generator'
                    self._noise = torch.randn(self._particles, self._noise_dim)
                    params = self.sample_parameters(self._noise, particles)

                alg_step = self.train_step(
                        (data, target),
                        model=model,
                        params=params,
                        particles=particles,
                        state=state)
                self.update_with_gradient(alg_step.info)
                loss += alg_step.info.extra.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.extra)
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc)
            logging.info("Avg acc: {}".format(acc.mean() * 100))
        logging.info("Cum loss: {}".format(loss))

        return batch_idx + 1
    

    def train_step(self,
                   inputs,
                   model,
                   params=None,
                   loss_func=None,
                   particles=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            params (Tensor): sampled parameter for param_net, if None,
                will re-sample.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor with
                shape [batch_size] as a loss for optimizing the generator
            particles (int): number of sampled particles. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if loss_func is None:
            loss_func = self._loss_func

        assert model in ['generator', 'critic'], "argument ``model`` is " \
            "required to be either ``generator`` or ``critic``, got " \
            "{}".format(model)

        if model == 'critic':
            self._grad_func = self._fisher_loss_critic
        elif model == 'generator':
            self._grad_func = self._fisher_loss_generator
        
        if self._regenerate_for_each_batch:
            params = self.sample_parameters(particles=particles)
        else:
            assert params is not None, "Need sample params first."

        train_info, loss_propagated = self._grad_func(
                inputs,
                params,
                loss_func)

        return AlgStep(
            output=params,
            state=(),
            info=LossInfo(loss=loss_propagated, extra=train_info))
    
    def _approx_jacobian_trace(self, critic_out, params):
        """Hutchinson's trace Jacobian estimator O(1) call to autograd"""
        eps = torch.randn_like(critic_out)
        jvp = torch.autograd.grad(
                critic_out,
                params,
                grad_outputs=eps,
                retain_graph=True,
                create_graph=True)[0]
        tr_jvp = torch.einsum('bi,bi->b', jvp, eps)
        return tr_jvp

    def _exact_jacobian_trace(self, fx, x):
        vals = []
        for i in range(x.size(1)):
            fxi = fx[:, i]
            dfxi_dxi = torch.autograd.grad(
                    fxi.sum(),
                    x,
                    grad_outputs=None,
                    retain_graph=True,
                    create_graph=True)[0][:, i][:, None]
            vals.append(dfxi_dxi)
        vals = torch.cat(vals, dim=1)
        return vals.sum(dim=1)
    
    def _spectral_norm(self, module):
        if 'weight' in module._parameters:
            torch.nn.utils.spectral_norm(module)

    def _fisher_loss_critic(self, inputs, params, loss_func):
        """compute optim direction \phi^*(x) (fisher-ns)"""
        data, target = inputs
        particles = params.shape[0]
        self._param_net.set_parameters(params)
        output, _ = self._param_net(data)
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                *target.shape[1:])
        loss, extra = self._compute_loss(output, target, loss_func)
        log_p = torch.autograd.grad(loss.sum(), params)[0]

        critic_samples = self._disc_net(params)[0]  # [n x params]
        log_p_f = log_p * critic_samples # [n x params]
        tr_critic = self._approx_jacobian_trace(critic_samples, params) # [n]
        
        for p in self._net.parameters():
            p.requires_grad = False

        # Estimate S(p, q)
        stein_pq = (log_p_f + tr_critic.unsqueeze(1)).mean(1) # [n x 1]
        
        lamb = 10.
        l2_penalty = (critic_samples * critic_samples).mean(1) * lamb
        loss_pq = stein_pq - l2_penalty
        
        adv_loss = -1 * loss_pq
        
        train_info = HyperNetworkLossInfo(loss=loss.detach(), extra=extra)
        loss_propagated = adv_loss
        
        return train_info, loss_propagated
    
    def _fisher_loss_generator(self, inputs, params, loss_func):
        """backpropagate transformed particles back to generator"""    
        fx = self._disc_net(params)[0]
        train_info = HyperNetworkLossInfo(loss=torch.zeros(1), extra=torch.zeros(1))
        loss_propagated = torch.sum(fx.detach() * params, dim=1)
        return train_info, loss_propagated


    def _classification_loss(self, output, target, loss_func):
        pred = output.max(-1)[1]
        acc = pred.eq(target).float().mean(0)
        avg_acc = acc.mean()
        loss = loss_func(output.transpose(1, 2), target)
        return loss, avg_acc

    def _regression_loss(self, output, target, loss_func):
        out_shape = output.shape[-1]
        assert (target.shape[-1] == out_shape), (
            "feature dimension of output and target does not match.")
        loss = .5 * loss_func(
            output.reshape(-1, out_shape),
            target.reshape(-1, out_shape),
            reduction='sum')
        return loss, ()

    def _rbf_func(self, x, y, h_min=1e-3):
        r"""Compute the rbf kernel and its gradient w.r.t. first entry 
            :math:`K(x, y), \nabla_x K(x, y)`

        Args:
            x (Tensor): set of N particles, shape (Nx x W), where W is the 
                dimenseion of each particle
            y (Tensor): set of N particles, shape (Ny x W), where W is the 
                dimenseion of each particle
            h_min (float): minimum kernel bandwidth

        Returns:
            :math:`K(x, y)` (Tensor): the RBF kernel of shape (Nx x Ny)
            :math:`\nabla_x K(x, y)` (Tensor): the derivative of RBF kernel of shape (Nx x Ny x D)
            
        """
        Nx, Dx = x.shape
        Ny, Dy = y.shape
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, W]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        h = self._median_width(dist_sq)
        h = torch.max(h, torch.as_tensor([h_min]))

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                                  -2 * diff / h)  # [Nx, Ny, W]

        return kappa, kappa_grad

    def _median_width(self, mat_dist):
        """Compute the kernel width from median of the distance matrix."""

        values, _ = torch.topk(
            mat_dist.view(-1), k=mat_dist.nelement() // 2 + 1)
        median = values[-1]
        return median / np.log(mat_dist.shape[0])

    def _kernel_width(self):
        # TODO: implement the kernel bandwidth selection via Heat equation.
        return self._kernel_sharpness

    def evaluate(self, loss_func=None, particles=None):
        """Evaluate on a randomly drawn network. """

        assert self._test_loader is not None, "Must set test_loader first."
        if loss_func is None:
            loss_func = self._loss_func
        if self._use_fc_bn:
            self._net.eval()
        params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        if self._use_fc_bn:
            self._net.train()
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._param_net(data)  # [B, N, D]
                loss, extra = self._vote(output, target, loss_func)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            logging.info("Test acc: {}".format(test_acc * 100))
        logging.info("Test loss: {}".format(test_loss))

    def _classification_vote(self, output, target, loss_func):
        """ensmeble the ooutputs from sampled classifiers."""
        particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = pred.mode(1)[0]  # [B, 1]
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss = loss_func(output.transpose(1, 2), target)
        return loss, correct

    def _regression_vote(self, output, target, loss_func):
        """ensemble the outputs for sampled regressors."""
        particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = loss_func(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        total_loss = loss_func(output, target)
        return loss, total_loss

