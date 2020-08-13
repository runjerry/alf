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
<<<<<<< HEAD
=======
import functools
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
import gin
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable

import alf
<<<<<<< HEAD
=======
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
<<<<<<< HEAD
from alf.utils import common, math_ops
=======
from alf.utils import common, math_ops, summary_utils
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
from alf.utils.summary_utils import record_time

HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


<<<<<<< HEAD
@gin.configurable
class HyperNetwork(Generator):
    """HyperNetwork 

    HyperrNetwork is a generator that generates a set of parameters for a predefined
    neural network from a random noise input. It is based on the following work:
=======
def classification_loss(output, target):
    pred = output.max(-1)[1]
    acc = pred.eq(target).float().mean(0)
    avg_acc = acc.mean()
    loss = F.cross_entropy(output.transpose(1, 2), target)
    return HyperNetworkLossInfo(loss=loss, extra=avg_acc)


def regression_loss(output, target):
    out_shape = output.shape[-1]
    assert (target.shape[-1] == out_shape), (
        "feature dimension of output and target does not match.")
    loss = 0.5 * F.mse_loss(
        output.reshape(-1, out_shape),
        target.reshape(-1, out_shape),
        reduction='sum')
    return HyperNetworkLossInfo(loss=loss, extra=())


def neglogprob(inputs, param_net, loss_type, params):
    if loss_type == 'regression':
        loss_func = regression_loss
    elif loss_type == 'classification':
        loss_func = classification_loss
    else:
        raise ValueError("Unsupported loss_type: %s" % loss_type)

    param_net.set_parameters(params)
    particles = params.shape[0]
    data, target = inputs
    output, _ = param_net(data)  # [B, N, D]
    target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                        *target.shape[1:])
    return loss_func(output, target)


@gin.configurable
class HyperNetwork(Algorithm):
    """HyperNetwork 

    HyperrNetwork algorithm maintains a generator that generates a set of 
    parameters for a predefined neural network from a random noise input. 
    It is based on the following work:
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

    https://github.com/neale/HyperGAN

    Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, 
    Performant Neural Networks." International Conference on Machine Learning. 2019.

    Major differences versus the original paper are:

<<<<<<< HEAD
        * A single generator that generates parameters for all network layers.

        * Remove the mixer and the discriminator.

        * Amortized particle-based variational inference (ParVI) to train the hypernetwork,
          in particular, two ParVI methods are implemented:

            1. amortized Stein Variational Gradient Descent (SVGD):

            Feng et al "Learning to Draw Samples with Amortized Stein Variational
            Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

            2. amortized Wasserstein ParVI with Smooth Functions (GFSF):

            Liu, Chang, et al. "Understanding and accelerating particle-based 
            variational inference." International Conference on Machine Learning. 2019.
=======
    * A single genrator that generates parameters for all network layers.

    * Remove the mixer and the distriminator.

    * The generator is trained with Amortized particle-based variational 
      inference (ParVI) methods, please refer to generator.py for details.
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

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
<<<<<<< HEAD
                 particles=32,
                 entropy_regularization=1.,
                 kernel_sharpness=1.,
                 loss_type="classification",
                 loss_func: Callable = None,
                 voting="soft",
                 par_vi="gfsf",
                 optimizer=None,
                 regenerate_for_each_batch=True,
                 print_network=False,
=======
                 particles=10,
                 entropy_regularization=1.,
                 loss_type="classification",
                 voting="soft",
                 par_vi="svgd",
                 optimizer=None,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
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
<<<<<<< HEAD
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format 
                ``(size, use_bias)``, where ``use_bias`` is optional. It
                appends an additional layer at the very end. Note that if 
                ``last_activation`` is specified, ``last_layer_param`` has to
                be specified explicitly
=======
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization
<<<<<<< HEAD
            kernel_sharpness (float): Used only for entropy_regularization > 0.
                We calcualte the kernel in SVGD as:
                    :math:`\exp(-kernel_sharpness * reduce_mean(\frac{(x-y)^2}{width}))`
                where width is the elementwise moving average of :math:`(x-y)^2`
=======
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
<<<<<<< HEAD
            loss_func (Callable): loss_func(outputs, targets)   
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            regenerate_for_each_batch (bool): If True, particles will be regenerated 
                for every training batch.
            print_network (bool): whether print out the archetectures of networks.
            name (str):
        """
=======
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            par_vi (str): types of particle-based methods for variational inference,
                types are [``svgd``, ``svgd2``, ``svgd3``, ``gfsf``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            logging_training (bool): whether logging loss and acc during training.
            logging_evaluate (bool): whether logging loss and acc of evaluate.
            config (TrainerConfig): configuration for training
            name (str):
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))
        net = EncodingNetwork(
            noise_spec,
            fc_layer_params=hidden_layers,
            use_fc_bn=use_fc_bn,
            last_layer_size=gen_output_dim,
            last_activation=math_ops.identity,
            name="Generator")

<<<<<<< HEAD
        if print_network:
            print("Generated network")
            print("-" * 68)
            print(param_net)

            print("Generator network")
            print("-" * 68)
            print(net)

        super().__init__(
=======
        if logging_network:
            logging.info("Generated network")
            logging.info("-" * 68)
            logging.info(param_net)

            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)

        if par_vi == 'svgd':
            par_vi = 'svgd3'

        self._generator = Generator(
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
<<<<<<< HEAD
            kernel_sharpness=kernel_sharpness,
=======
            par_vi=par_vi,
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
            optimizer=optimizer,
            name=name)

        self._param_net = param_net
        self._particles = particles
<<<<<<< HEAD
        self._train_loader = None
        self._test_loader = None
        self._regenerate_for_each_batch = regenerate_for_each_batch
        self._loss_func = loss_func
        self._par_vi = par_vi
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
=======
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
        assert (voting in ['soft', 'hard'
                           ]), ("voting only supports \"soft\" and \"hard\"")
        self._voting = voting
        if loss_type == 'classification':
<<<<<<< HEAD
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
        if par_vi == 'gfsf':
            self._grad_func = self._stein_grad
        elif par_vi == 'svgd':
            self._grad_func = self._svgd_grad
        else:
            assert ValueError("par_vi only supports \"gfsf\" and \"svgd\"")
=======
            self._vote = self._classification_vote
        elif loss_type == 'regression':
            self._vote = self._regression_vote
        else:
            raise ValueError("Unsupported loss_type: %s" % loss_type)
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

    def set_data_loader(self, train_loader, test_loader=None):
        """Set data loadder for training and testing."""
        self._train_loader = train_loader
        self._test_loader = test_loader
<<<<<<< HEAD
=======
        self._entropy_regularization = 1 / len(train_loader)
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

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
<<<<<<< HEAD
        params, _ = self._predict(
            inputs=None, noise=noise, batch_size=particles, training=training)
        return params

    def predict(self, inputs, params=None, particles=None):
        """Predict ensemble outputs for inputs using the hypernetwork model."""
=======
        generator_step = self._generator.predict_step(
            noise=noise, batch_size=particles, training=training)
        return generator_step.output

    def predict_step(self, inputs, params=None, particles=None, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.
        
        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, will resample.
            particles (int): size of sampled ensemble.
            state: not used.

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
        if params is None:
            params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
<<<<<<< HEAD
        return outputs
=======
        return AlgStep(output=outputs, state=(), info=())
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

    def train_iter(self, particles=None, state=None):
        """Perform one epoch (iteration) of training."""

        assert self._train_loader is not None, "Must set data_loader first."
<<<<<<< HEAD
=======
        alf.summary.increment_global_counter()
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
        with record_time("time/train"):
            loss = 0.
            if self._loss_type == 'classification':
                avg_acc = []
<<<<<<< HEAD
            params = None
            if not self._regenerate_for_each_batch:
                params = self.sample_parameters(particles=particles)
=======
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target),
<<<<<<< HEAD
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
=======
                                           particles=particles,
                                           state=state)
                loss_info, params = self.update_with_gradient(alg_step.info)
                # loss += alg_step.info.extra.generator.loss
                loss += loss_info.extra.generator.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.generator.extra)
        acc = None
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._loss_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
        self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c

        return batch_idx + 1

    def train_step(self,
                   inputs,
<<<<<<< HEAD
                   params=None,
                   loss_func=None,
                   particles=None,
=======
                   particles=None,
                   entropy_regularization=None,
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
<<<<<<< HEAD
            params (Tensor): sampled parameter for param_net, if None,
                will re-sample.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor with
                shape [batch_size] as a loss for optimizing the generator
=======
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
            particles (int): number of sampled particles. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
<<<<<<< HEAD
        if loss_func is None:
            loss_func = self._loss_func
        if self._regenerate_for_each_batch:
            params = self.sample_parameters(particles=particles)
        else:
            assert params is not None, "Need sample params first."

        train_info, loss_propagated = self._grad_func(inputs, params,
                                                      loss_func)

        return AlgStep(
            output=params,
            state=(),
            info=LossInfo(loss=loss_propagated, extra=train_info))

    def _stein_grad(self, inputs, params, loss_func):
        """Compute particle gradients via gfsf (stein estimator). """
        data, target = inputs
        particles = params.shape[0]
        self._param_net.set_parameters(params)
        output, _ = self._param_net(data)  # [B, N, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss, extra = self._compute_loss(output, target, loss_func)

        loss_grad = torch.autograd.grad(loss.sum(), params)[0]
        logq_grad = self._score_func(params)
        grad = loss_grad - logq_grad

        train_info = HyperNetworkLossInfo(loss=loss, extra=extra)
        loss_propagated = torch.sum(grad.detach() * params, dim=-1)

        return train_info, loss_propagated

    def _svgd_grad(self, inputs, params, loss_func):
        """Compute particle gradients via svgd. """
        data, target = inputs
        particles = params.shape[0] // 2
        params_i, params_j = torch.split(params, particles, dim=0)
        self._param_net.set_parameters(params_j)
        output, _ = self._param_net(data)  # [B, N/2, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss, extra = self._compute_loss(output, target, loss_func)

        loss_grad = torch.autograd.grad(loss.sum(), params_j)[0]  # [Nj, W]
        q_i = params_i + torch.rand_like(params_i) * 1e-8
        q_j = params_j + torch.rand_like(params_j) * 1e-8
        kappa, kappa_grad = self._rbf_func(q_j, q_i)  # [Nj, Ni], [Nj, Ni, W]
        Nj = kappa.shape[0]
        kernel_logp = torch.einsum('ji, jw->iw', kappa, loss_grad) / Nj
        grad = (kernel_logp - kappa_grad.mean(0))  # [Ni, W]

        train_info = HyperNetworkLossInfo(loss=loss, extra=extra)
        loss_propagated = torch.sum(grad.detach() * params_i, dim=-1)

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
        loss = loss_func(
            output.reshape(-1, out_shape), target.reshape(-1, out_shape))
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

    def _score_func(self, x, alpha=1e-6, h_min=1e-3):
        r"""Compute the stein estimator of the score function 
            :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`

        Args:
            x (Tensor): set of N particles, shape (N x D), where D is the 
                dimenseion of each particle
            alpha (float): weight of regularization for inverse kernel

        Returns:
            :math:`\nabla\log q` (Tensor): the score function of shape (N x D)
            
        """
        N, D = x.shape
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        dist_sq = torch.sum(diff**2, -1)  # [N, N]

        # compute the kernel width
        h = self._median_width(dist_sq)
        h = torch.max(h, torch.as_tensor([h_min]))

        kappa = torch.exp(-dist_sq / h)  # [N, N]
        kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]

        return kappa_inv @ kappa_grad

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
=======
        params = self.sample_parameters(particles=particles)
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization

        return self._generator.train_step(
            inputs=None,
            loss_func=functools.partial(neglogprob, inputs, self._param_net,
                                        self._loss_type),
            outputs=params,
            entropy_regularization=entropy_regularization,
            state=())

    def evaluate(self, particles=None):
        """Evaluate on a randomly drawn network. """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        if self._use_fc_bn:
            self._generator.eval()
        params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        if self._use_fc_bn:
            self._generator.train()
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._param_net(data)  # [B, N, D]
<<<<<<< HEAD
                loss, extra = self._vote(output, target, loss_func)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            logging.info("Test acc: {}".format(test_acc * 100))
        logging.info("Test loss: {}".format(test_loss))

    def _classification_vote(self, output, target, loss_func):
=======
                loss, extra = self._vote(output, target)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            alf.summary.scalar(name='eval/test_acc', data=test_acc * 100)
        if self._logging_evaluate:
            if self._loss_type == 'classification':
                logging.info("Test acc: {}".format(test_acc * 100))
            logging.info("Test loss: {}".format(test_loss))
        alf.summary.scalar(name='eval/test_loss', data=test_loss)

    def _classification_vote(self, output, target):
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
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
<<<<<<< HEAD
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
=======
        loss = classification_loss(output.transpose(1, 2), target)
        return loss, correct

    def _regression_vote(self, output, target):
        """ensemble the outputs for sampled regressors."""
        particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = regression_loss(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        total_loss = regression_loss(output, target)
        return loss, total_loss

    def summarize_train(self, loss_info, params, cum_loss=None, avg_acc=None):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.

        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._config.debug_summaries:
            summary_utils.summarize_loss(loss_info)
        if cum_loss is not None:
            alf.summary.scalar(name='train_epoch/neglogprob', data=cum_loss)
        if avg_acc is not None:
            alf.summary.scalar(name='train_epoch/avg_acc', data=avg_acc)
>>>>>>> 48fe04a0e6d09d6894d409a5668394b65ad1329c
