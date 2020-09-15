# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""A generic generator."""

import gin
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.mi_estimator import MIEstimator
from alf.data_structures import AlgStep, LossInfo, namedtuple
import alf.nest as nest
from alf.networks import Network, EncodingNetwork
from alf.networks.relu_mlp import ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.averager import AdaptiveAverager

GeneratorLossInfo = namedtuple("GeneratorLossInfo",
                               ["generator", "mi_estimator"])


@gin.configurable
class CriticAlgorithm(Algorithm):
    """Wrap a critic network as an Algorithm for flexible gradient updates
    called by the Generator.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_dim=1,
                 hidden_layers=(3, 3),
                 net: Network = None,
                 spectral_norm=False,
                 optimizer=None,
                 name="CriticAlgorithm"):
        """Create a CriticAlgorithm.

        Args:
            input_tensor_spec (TensorSpec): spec of inputs. 
            output_dim (int): dimension of output, default value is 1.
            hidden_layers (tuple): size of hidden layers.
            net (Network): network for predicting outputs from inputs.
                If None, a default one with hidden_layers will be created
            spectral_norm (bool): whether or not apply spectral norm on net.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training.
            name (str): name of this CriticAlgorithm.
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        if net is None:
            if output_dim == 1:
                self._output_dim = output_dim
                net = EncodingNetwork(
                    input_tensor_spec=input_tensor_spec,
                    fc_layer_params=hidden_layers,
                    last_layer_size=output_dim,
                    last_activation=math_ops.identity,
                    name='Critic')
            else:
                self._output_dim = input_tensor_spec.shape[0]
                net = ReluMLP(
                    input_tensor_spec=input_tensor_spec,
                    hidden_layers=hidden_layers,
                    name="Critic")
        if spectral_norm:
            net.apply(self._spectral_norm)

        self._net = net

    def _spectral_norm(self, module):
        if 'weight' in module._parameters:
            torch.nn.utils.spectral_norm(module)

    def _reset_parameters(self, module):
        if hasattr(module, '_reset_parameters'):
            module._reset_parameters()

    def reset_net_parameters(self):
        self._net.apply(self._reset_parameters)

    def predict_step(self,
                     inputs,
                     state=None,
                     compute_jac=False,
                     training=False):
        """Predict for one step of inputs.

        Args:
            inputs (Tensor): inputs for prediction.
            state: not used.
            compute_jac (bool): whether to compute the diagonal of the 
                jacobian matrix.

        Returns:
            AlgStep:
            - output (Tensor): prediction only if compute_jac is False.
                (prediction, gradients) if compute_jac is True. 
            - state: not used.
        """
        outputs = self._net(inputs)[0]
        if compute_jac:
            if self._output_dim == 1:
                jac = torch.autograd.grad(
                    outputs.sum(), inputs, create_graph=training)[0]
            else:
                jac = self._net.compute_jac_diag(inputs)
            outputs = (outputs, jac)
        return AlgStep(output=outputs, state=(), info=())


@gin.configurable
class Generator(Algorithm):
    """Generator

    Generator generates outputs given `inputs` (can be None) by transforming
    a random noise and input using `net`:


    The generator is trained to minimize the following objective:

        :math:`E(loss\_func(net([noise, input]))) - entropy\_regulariztion \cdot H(P)`

    where P is the (conditional) distribution of outputs given the inputs
    implied by `net` and H(P) is the (conditional) entropy of P.

    If the loss is the (unnormalized) negative log probability of some
    distribution Q and the ``entropy_regularization`` is 1, this objective is
    equivalent to minimizing :math:`KL(P||Q)`.

    It uses two different ways to optimize `net` depending on
    ``entropy_regularization``:

    * ``entropy_regularization`` = 0: the minimization is achieved by simply
      minimizing loss_func(net([noise, inputs]))

    * entropy_regularization > 0: the minimization is achieved using amortized
      particle-based variational inference (ParVI), in particular, two ParVI
      methods are implemented:

        1. amortized Stein Variational Gradient Descent (SVGD):

        Feng et al "Learning to Draw Samples with Amortized Stein Variational
        Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

        2. amortized Wasserstein ParVI with Smooth Functions (GFSF):

        Liu, Chang, et al. "Understanding and accelerating particle-based 
        variational inference." International Conference on Machine Learning. 2019.

    It also supports an additional optional objective of maximizing the mutual
    information between [noise, inputs] and outputs by using mi_estimator to
    prevent mode collapse. This might be useful for ``entropy_regulariztion`` = 0
    as suggested in section 5.1 of the following paper:

    Hjelm et al `Learning Deep Representations by Mutual Information Estimation
    and Maximization <https://arxiv.org/pdf/1808.06670.pdf>`_
    """

    def __init__(
            self,
            output_dim,
            noise_dim=32,
            input_tensor_spec=None,
            hidden_layers=(256, ),
            net: Network = None,
            net_moving_average_rate=None,
            entropy_regularization=0.,
            mi_weight=None,
            mi_estimator_cls=MIEstimator,
            par_vi="gfsf",
            critic_l2_weight=1.,
            critic_spectral_norm=False,
            num_critic_iter=10,
            # num_critic_outer_iter=10,
            optimizer=None,
            critic_optimizer=None,
            # potential_optimizer=None,
            name="Generator"):
        r"""Create a Generator.

        Args:
            output_dim (int): dimension of output
            noise_dim (int): dimension of noise
            input_tensor_spec (nested TensorSpec): spec of inputs. If there is
                no inputs, this should be None.
            hidden_layers (tuple): size of hidden layers.
            net (Network): network for generating outputs from [noise, inputs]
                or noise (if inputs is None). If None, a default one with
                hidden_layers will be created
            net_moving_average_rate (float): If provided, use a moving average
                version of net to do prediction. This has been shown to be
                effective for GAN training (arXiv:1907.02544, arXiv:1812.04948).
            entropy_regularization (float): weight of entropy regularization
            mi_estimator_cls (type): the class of mutual information estimator
                for maximizing the mutual information between [noise, inputs]
                and [outputs, inputs].
            par_vi (string): ParVI methods, options are
                [``svgd``, ``svgd2``, ``svgd3``, ``gfsf``],
                * svgd: empirical expectation of SVGD is evaluated by a single 
                    resampled particle. The main benefit of this choice is it 
                    supports conditional case, while all other options do not.
                * svgd2: empirical expectation of SVGD is evaluated by splitting
                    half of the sampled batch. It is a trade-off between 
                    computational efficiency and convergence speed.
                * svgd3: empirical expectation of SVGD is evaluated by 
                    resampled particles of the same batch size. It has better
                    convergence but involves resampling, so less efficient
                    computaionally comparing with svgd2.
                * gfsf: wasserstein gradient flow with smoothed functions. It 
                    involves a kernel matrix inversion, so computationally most
                    expensive, but in some case the convergence seems faster 
                    than svgd approaches.
            critic_l2_weight (float): weight of l2 penalty for training critic.
            critic_spectral_norm (bool): whether or not apply spectral norm on 
                critic network.
            num_critic_inner_iter (int): number of innner optimization iterations
                for critic update.
            num_critic_outer_iter (int): number of outer optimization iterations
                for critic update.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training
            critic_optimizer (torch.optim.Optimizer): (optional) optimizer for 
                training the critic when using ``minmax_wgf``.
            name (str): name of this generator
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        self._output_dim = output_dim
        self._noise_dim = noise_dim
        self._entropy_regularization = entropy_regularization
        self._par_vi = par_vi
        if entropy_regularization == 0:
            self._grad_func = self._ml_grad
        else:
            if par_vi == 'gfsf':
                self._grad_func = self._gfsf_grad
            elif par_vi == 'svgd':
                self._grad_func = self._svgd_grad
            elif par_vi == 'svgd2':
                self._grad_func = self._svgd_grad2
            elif par_vi == 'svgd3':
                self._grad_func = self._svgd_grad3
            elif par_vi == 'minmax_wgf':
                self._grad_func = self._minmax_wgf_grad
                self._num_critic_iter = num_critic_iter
                self._critic_spectral_norm = critic_spectral_norm
                # self._num_critic_outer_iter = num_critic_outer_iter
                self._critic_l2_weight = critic_l2_weight
                # self._critic = CriticAlgorithm(TensorSpec(shape=(output_dim, )),
                #                                # output_dim=output_dim,
                #                                spectral_norm=critic_spectral_norm,
                #                                optimizer=critic_optimizer)
                # self._potential = CriticAlgorithm(TensorSpec(shape=(output_dim, )),
                #                                # output_dim=output_dim,
                #                                spectral_norm=critic_spectral_norm,
                #                                optimizer=potential_optimizer)
            else:
                raise ValueError("Unsupported par_vi method: %s" % par_vi)

            self._kernel_width_averager = AdaptiveAverager(
                tensor_spec=TensorSpec(shape=()))

        noise_spec = TensorSpec(shape=(noise_dim, ))

        if net is None:
            net_input_spec = noise_spec
            if input_tensor_spec is not None:
                net_input_spec = [net_input_spec, input_tensor_spec]
            net = EncodingNetwork(
                input_tensor_spec=net_input_spec,
                fc_layer_params=hidden_layers,
                last_layer_size=output_dim,
                last_activation=math_ops.identity,
                name="Generator")

        self._mi_estimator = None
        self._input_tensor_spec = input_tensor_spec
        if mi_weight is not None:
            x_spec = noise_spec
            y_spec = TensorSpec((output_dim, ))
            if input_tensor_spec is not None:
                x_spec = [x_spec, input_tensor_spec]
            self._mi_estimator = mi_estimator_cls(
                x_spec, y_spec, sampler='shift')
            self._mi_weight = mi_weight
        self._net = net
        self._predict_net = None
        self._net_moving_average_rate = net_moving_average_rate
        if net_moving_average_rate:
            self._predict_net = net.copy(name="Genrator_average")
            self._predict_net_updater = common.get_target_updater(
                self._net, self._predict_net, tau=net_moving_average_rate)

    def _trainable_attributes_to_ignore(self):
        return ["_predict_net"]

    @property
    def noise_dim(self):
        return self._noise_dim

    def _predict(self, inputs=None, noise=None, batch_size=None,
                 training=True):
        if inputs is None:
            assert self._input_tensor_spec is None
            if noise is None:
                assert batch_size is not None
                noise = torch.randn(batch_size, self._noise_dim)
            gen_inputs = noise
        else:
            nest.assert_same_structure(inputs, self._input_tensor_spec)
            batch_size = nest.get_nest_batch_size(inputs)
            if noise is None:
                noise = torch.randn(batch_size, self._noise_dim)
            else:
                assert noise.shape[0] == batch_size
                assert noise.shape[1] == self._noise_dim
            gen_inputs = [noise, inputs]
        if self._predict_net and not training:
            outputs = self._predict_net(gen_inputs)[0]
        else:
            outputs = self._net(gen_inputs)[0]
        return outputs, gen_inputs

    def predict_step(self,
                     inputs=None,
                     noise=None,
                     batch_size=None,
                     training=False,
                     state=None):
        """Generate outputs given inputs.

        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            noise (Tensor): input to the generator.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None
            training (bool): whether train the generator.
            state: not used

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        outputs, _ = self._predict(
            inputs=inputs,
            noise=noise,
            batch_size=batch_size,
            training=training)
        return AlgStep(output=outputs, state=(), info=())

    def train_step(self,
                   inputs,
                   loss_func,
                   outputs=None,
                   batch_size=None,
                   entropy_regularization=None,
                   state=None):
        """
        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            outputs (Tensor): generator's output (possibly from previous runs) used
                for this train_step.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor or namedtuple
                of tensors with field `loss`, which is a Tensor of
                shape [batch_size] a loss term for optimizing the generator.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None.
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if outputs is None:
            outputs, gen_inputs = self._predict(inputs, batch_size=batch_size)
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization

        loss, loss_propagated = self._grad_func(inputs, outputs, loss_func,
                                                entropy_regularization)
        mi_loss = ()
        if self._mi_estimator is not None:
            mi_step = self._mi_estimator.train_step([gen_inputs, outputs])
            mi_loss = mi_step.info.loss
            loss_propagated = loss_propagated + self._mi_weight * mi_loss

        return AlgStep(
            output=outputs,
            state=(),
            info=LossInfo(
                loss=loss_propagated,
                extra=GeneratorLossInfo(generator=loss, mi_estimator=mi_loss)))

    def _ml_grad(self, inputs, outputs, loss_func,
                 entropy_regularization=None):
        loss_inputs = outputs if inputs is None else [outputs, inputs]
        loss = loss_func(loss_inputs)

        grad = torch.autograd.grad(loss.sum(), outputs)[0]
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _kernel_width(self, dist):
        """Update kernel_width averager and get latest kernel_width. """
        if dist.ndim > 1:
            dist = torch.sum(dist, dim=-1)
            assert dist.ndim == 1, "dist must have dimension 1 or 2."
        width, _ = torch.median(dist, dim=0)
        width = width / np.log(len(dist))
        self._kernel_width_averager.update(width)

        return self._kernel_width_averager.get()

    def _rbf_func(self, x, y):
        """Compute RGF kernel, used by svgd_grad. """
        d = (x - y)**2
        d = torch.sum(d, -1)
        h = self._kernel_width(d)
        w = torch.exp(-d / h)

        return w

    def _rbf_func2(self, x, y):
        r"""
        Compute the rbf kernel and its gradient w.r.t. first entry 
        :math:`K(x, y), \nabla_x K(x, y)`, used by svgd_grad2.

        Args:
            x (Tensor): set of N particles, shape (Nx x W), where W is the 
                dimenseion of each particle
            y (Tensor): set of N particles, shape (Ny x W), where W is the 
                dimenseion of each particle

        Returns:
            :math:`K(x, y)` (Tensor): the RBF kernel of shape (Nx x Ny)
            :math:`\nabla_x K(x, y)` (Tensor): the derivative of RBF kernel of shape (Nx x Ny x D)
            
        """
        Nx, Dx = x.shape
        Ny, Dy = y.shape
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, W]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        h = self._kernel_width(dist_sq.view(-1))

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                                  -2 * diff / h)  # [Nx, Ny, W]
        return kappa, kappa_grad

    def _score_func(self, x, alpha=1e-5):
        r"""
        Compute the stein estimator of the score function 
        :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`,
        used by gfsf_grad. 

        Args:
            x (Tensor): set of N particles, shape (N x D), where D is the 
                dimenseion of each particle
            alpha (float): weight of regularization for inverse kernel
                this parameter turns out to be crucial for convergence.

        Returns:
            :math:`\nabla\log q` (Tensor): the score function of shape (N x D)
            
        """
        N, D = x.shape
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        dist_sq = torch.sum(diff**2, -1)  # [N, N]
        h, _ = torch.median(dist_sq.view(-1), dim=0)
        h = h / np.log(N)

        kappa = torch.exp(-dist_sq / h)  # [N, N]
        kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]

        return -kappa_inv @ kappa_grad

    def _svgd_grad(self, inputs, outputs, loss_func, entropy_regularization):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by a single resampled particle. 
        """
        outputs2, _ = self._predict(inputs, batch_size=outputs.shape[0])
        kernel_weight = self._rbf_func(outputs, outputs2)
        weight_sum = entropy_regularization * kernel_weight.sum()

        kernel_grad = torch.autograd.grad(weight_sum, outputs2)[0]

        loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        weighted_loss = kernel_weight.detach() * neglogp

        loss_grad = torch.autograd.grad(weighted_loss.sum(), outputs2)[0]
        grad = loss_grad - kernel_grad
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _svgd_grad2(self, inputs, outputs, loss_func, entropy_regularization):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by splitting half of the sampled batch. 
        """
        assert inputs is None, '"svgd2" does not support conditional generator'
        num_particles = outputs.shape[0] // 2
        outputs_i, outputs_j = torch.split(outputs, num_particles, dim=0)
        loss_inputs = outputs_j
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs_j)[0]  # [Nj, D]

        # [Nj, Ni], [Nj, Ni, D]
        kernel_weight, kernel_grad = self._rbf_func2(outputs_j, outputs_i)
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [Ni, D]
        grad = kernel_logp - entropy_regularization * kernel_grad.mean(0)
        loss_propagated = torch.sum(grad.detach() * outputs_i, dim=-1)
        return loss, loss_propagated

    def _svgd_grad3(self, inputs, outputs, loss_func, entropy_regularization):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by resampled particles of the same batch size. 
        """
        assert inputs is None, '"svgd3" does not support conditional generator'
        num_particles = outputs.shape[0]
        outputs2, _ = self._predict(inputs, batch_size=num_particles)
        loss_inputs = outputs2
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs2)[0]  # [N2, D]

        # [N2, N], [N2, N, D]
        kernel_weight, kernel_grad = self._rbf_func2(outputs2, outputs)
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [N, D]
        grad = kernel_logp - entropy_regularization * kernel_grad.mean(0)
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _gfsf_grad(self, inputs, outputs, loss_func, entropy_regularization):
        """Compute particle gradients via GFSF (Stein estimator). """
        assert inputs is None, '"gfsf" does not support conditional generator'
        loss_inputs = outputs
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs)[0]  # [N2, D]

        logq_grad = self._score_func(outputs)
        grad = loss_grad + entropy_regularization * logq_grad
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _critic_jac_diag_over_preds(self, inputs):
        assert inputs.shape[-1] == self._output_dim, (
            "inputs dim does not match the critic input_dim")
        critic_step = self._critic.predict_step(inputs, compute_jac=True)
        preds, jac_diag = critic_step.output
        outputs = jac_diag / (preds + 1e-6)
        # outputs = jac_diag / preds
        return outputs

    def _critic_train_step(self, inputs, vec):
        assert (inputs.shape[-1] == self._output_dim), (  # and (
            # vec.shape[-1] == self._output_dim), (
            "inputs dim does not match the critic input_dim")
        # vec_preds = self._critic_jac_diag_over_preds(inputs)
        # loss = F.mse_loss(vec_preds, vec, reduction='sum')
        critic_step = self._critic.predict_step(
            inputs, compute_jac=True, training=True)
        preds, jac = critic_step.output
        # vec_preds = preds * vec
        # loss = (vec_preds - jac_diag) * (vec_preds - jac_diag)
        # norm = (preds * preds).sum(-1) # + 1e-6
        # loss = loss.sum(-1) - norm * 10.
        # idx1 = torch.where(torch.abs(preds) < 1e-5)
        # idx2 = torch.where(torch.abs(preds) > 1e-5)
        # loss1 = jac[idx1] * jac[idx1]
        # loss2 = (vec[idx2] - jac[idx2]/preds[idx2]) * (vec[idx2] - jac[idx2]/preds[idx2])
        # loss = loss1.sum() + loss2.sum()
        loss = (vec * preds - jac) * (vec * preds - jac)
        norm = preds * preds
        loss = loss - norm * 20

        return -loss

    def _minmax_wgf_grad(self, inputs, outputs, loss_func,
                         entropy_regularization):
        """Compute particle gradient via minmax WGF. """
        assert inputs is None, '"gfsf" does not support conditional generator'
        loss = loss_func(outputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs)[0]  # [N2, D]

        # solve for optimal logq gradient vector
        # batch_size = 64
        # for outer_iter in range(self._num_critic_outer_iter):
        # vec = torch.rand(outputs.shape)
        num_particles = outputs.shape[0]
        outputs2, _ = self._predict(inputs, batch_size=num_particles)
        # self._critic.reset_net_parameters()
        self._critic = CriticAlgorithm(
            TensorSpec(shape=(self._output_dim, )),
            spectral_norm=self._critic_spectral_norm,
            optimizer=alf.optimizers.AdamTF(lr=1e-4))
        # vec = torch.rand(outputs.shape)
        perm = torch.randperm(self._output_dim)
        idx = perm[0]
        for outer_iter in range(4):
            inputs = outputs2.detach().clone()
            inputs.requires_grad = True
            # vec = self._critic_jac_diag_over_preds(inputs).detach()
            critic_step = self._critic.predict_step(inputs, compute_jac=True)
            preds, jac = critic_step.output
            vec = jac[:, [idx]] / (preds + 1e-6)
            for inner_iter in range(5):
                # for inner_iter in range(self._num_critic_iter):

                # perm = torch.randperm(outputs.shape[0])
                # idx = perm[:batch_size]
                # inputs = outputs[idx].detach().clone()
                # ivec = vec[idx]
                critic_inputs = outputs2.detach().clone()
                critic_inputs.requires_grad = True
                critic_loss = self._critic_train_step(critic_inputs,
                                                      vec.detach())
                self._critic.update_with_gradient(LossInfo(loss=critic_loss))

        neg_logq_grad = torch.zeros(outputs.shape)
        critic_step = self._critic.predict_step(outputs, compute_jac=True)
        preds, jac = critic_step.output
        vec = jac[:, [idx]] / (preds + 1e-6)
        neg_logq_grad[:, [idx]] = vec
        # neg_logq_grad = self._critic_jac_diag_over_preds(outputs)
        grad = loss_grad - entropy_regularization * neg_logq_grad
        # print("mean logq_grad norm: {}".format(logq_grad.mean(0).norm()))
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def after_update(self, training_info):
        if self._predict_net:
            self._predict_net_updater()
