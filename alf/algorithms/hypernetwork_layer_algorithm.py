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
import functools
import gin
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.networks.relu_mlp import ReluMLP
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.algorithms.hypernetwork_layer_generator import ParamLayers
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops, summary_utils
from alf.utils.summary_utils import record_time

from alf.algorithms.sl_algorithm import SLAlgorithm


HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


@gin.configurable
class HyperNetwork(SLAlgorithm):
    """HyperNetwork 

    HyperrNetwork algorithm maintains a generator that generates a set of 
    parameters for a predefined neural network from a random noise input. 
    It is based on the following work:

    https://github.com/neale/HyperGAN

    Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, 
    Performant Neural Networks." International Conference on Machine Learning. 2019.

    Major differences versus the original paper are:

    * A single genrator that generates parameters for all network layers.

    * Remove the mixer and the distriminator.

    * The generator is trained with Amortized particle-based variational 
      inference (ParVI) methods, please refer to generator.py for details.

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
                 num_particles=10,
                 entropy_regularization=1.,
                 parameterization='layer',
                 use_relu_mlp=False,
                 loss_type="classification",
                 voting="soft",
                 par_vi="svgd",
                 amortize_vi=True,
                 particle_optimizer=None,
                 function_vi=False,
                 functional_gradient=False,
                 use_pinverse=False,
                 pinverse_use_eps=True,
                 pinverse_type='network',
                 pinverse_resolve=False,
                 pinverse_solve_iters=1,
                 use_jac_regularization=False,
                 square_jac=True,
                 pinverse_batch_size=None,
                 optimizer=None,
                 critic_optimizer=None,
                 critic_hidden_layers=(100, 100),
                 critic_l2_weight=0.,
                 function_bs=None,
                 function_space_samples=0,
                 logging_network=False,
                 logging_training=False,
                 logging_evaluate=False,
                 config: TrainerConfig = None,
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
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            num_particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization
            parameterization (str): choice of parameterization for the
                hypernetwork. Choices are [``network``, ``layer``].
                A parameterization of ``network`` uses a single generator to
                generate all the weights at once. A parameterization of ``layer``
                uses one generator for each layer of output parameters.

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
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
        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))
        assert parameterization in ['network', 'layer'], "Hypernetwork " \
                "can only be parameterized by \"network\" or \"layer\" " \
                "generators"
        if parameterization == 'network':
            if functional_gradient:
                net = ReluMLP(
                    noise_spec,
                    hidden_layers=hidden_layers,
                    output_size=gen_output_dim,
                    bias=True,
                    name='Generator')
            else:
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
                particles=num_particles,
                input_tensor_spec=input_tensor_spec,
                conv_layer_params=conv_layer_params,
                fc_layer_params=fc_layer_params,
                last_layer_param=last_layer_param,
                last_activation=math_ops.identity,
                hidden_layers=hidden_layers,
                activation=activation,
                use_fc_bn=use_fc_bn,
                use_bias=True,
                optimizer=optimizer,
                name="Generator")

        if function_vi:
            assert function_bs is not None, (
                "Need to specify batch size of function outputs.")
            critic_input_dim = function_bs * last_layer_param[0]
        else:
            critic_input_dim = gen_output_dim

        if logging_network:
            logging.info("Generated network")
            logging.info("-" * 68)
            logging.info(param_net)

            logging.info("Generator network")
            logging.info("-" * 68)
            logging.info(net)

        if par_vi == 'svgd':
            par_vi = 'svgd3'

        super().__init__(input_tensor_spec=(),
            loss_type=loss_type,
            net=param_net,
            predictor_vote=voting,
            optimizer=optimizer,
            name=name) 

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            par_vi=par_vi,
            functional_gradient=functional_gradient,
            use_pinverse=use_pinverse,
            pinverse_type=pinverse_type,
            pinverse_use_eps=pinverse_use_eps,
            pinverse_resolve=pinverse_resolve,
            pinverse_solve_iters=pinverse_solve_iters,
            pinverse_batch_size=pinverse_batch_size,
            use_jac_regularization=use_jac_regularization,
            square_jac = square_jac,
            amortize_vi=amortize_vi,
            optimizer=None,
            critic_input_dim=critic_input_dim,
            critic_relu_mlp=functional_gradient,
            critic_hidden_layers=critic_hidden_layers,
            critic_l2_weight=critic_l2_weight,
            critic_optimizer=critic_optimizer,
            name=name)
        
        self._param_net = param_net
        self._parameterization = parameterization
        self._amortize_vi = amortize_vi
        self._functional_gradient = functional_gradient
        self._function_vi = function_vi
        self._function_space_samples = function_space_samples
        self._num_particles = num_particles
        self._entropy_regularization = entropy_regularization
        self._train_loader = None
        self._test_loader = None
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        self._logging_training = logging_training
        self._logging_evaluate = logging_evaluate
        self._config = config

        if not self._amortize_vi:
            particle_params = self._generator.predict_step(
                batch_size=num_particles, training=True).output
            self._params = torch.nn.Parameter(particle_params.clone(),
                requires_grad=True)
            self._particle_optimizer = particle_optimizer
            self._particle_optimizer.add_param_group({"params": self._params})
            self._param_net.set_parameters(self._params.data)
   
    def set_num_particles(self, num_particles):
        """Set the number of particles to sample through one forward
        pass of the hypernetwork. """
        self._num_particles = num_particles
    
    @property
    def num_particles(self):
        return self._num_particles

    def sample_parameters(self, noise=None, num_particles=None, training=True):
        "Sample parameters for an ensemble of networks." ""
        if noise is None and num_particles is None:
            num_particles = self._num_particles
        if self._amortize_vi:
            if self._functional_gradient:
                output, _ = self._generator._predict(batch_size=num_particles)
            else:
                output = self._generator.predict_step(
                noise=noise, batch_size=num_particles, training=training).output
            
        else:
            output = self._params
            self._param_net.set_parameters(output)
        return output

    def predict_step(self, inputs, params=None, num_particles=None, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.
        
        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, will resample.
            num_particles (int): size of sampled ensemble.
            state: not used.

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        if params is None:
            params = self.sample_parameters(num_particles=num_particles)
        if self._functional_gradient:
            params = params[0]
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return AlgStep(output=outputs, state=(), info=())
     
    def train_iter(self, state=None):
        """ Perform a single (iteration) epoch of training"""
        assert self._train_loader is not None, "Must set data_loader first"
        alf.summary.increment_global_counter()
        with record_time("time/train"):
            loss = 0.
            pinverse_loss = 0
            if self._loss_type == 'classification':
                avg_acc = []
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target), state=state)
                if self._amortize_vi or self._function_vi:
                    loss_info, params = self.update_with_gradient(alg_step.info)
                else:
                    update_direction = alg_step.info.loss
                    self._particle_optimizer.zero_grad()
                    self._params.grad = update_direction
                    self._particle_optimizer.step()
                    loss_info = alg_step.info
                    params =  [('ensemble_params', 0)]

                if hasattr(loss_info.extra, 'generator'):
                    loss += loss_info.extra.generator.loss
                    pinverse_loss += loss_info.extra.pinverse
                else:
                    loss += loss_info.loss
                if self._loss_type == 'classification':
                    if hasattr(alg_step.info.extra, 'generator'):
                        avg_acc.append(alg_step.info.extra.generator.extra)
                    else:
                        avg_acc.append(alg_step.info.extra)
        acc = None
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc).mean() * 100
        if self._logging_training:
            if self._loss_type == 'classification':
                logging.info("Avg acc: {}".format(acc))
            logging.info("Cum loss: {}".format(loss))
            if pinverse_loss is not None:
                pinverse_loss /= batch_idx
                logging.info("Avg pinverse loss: {}".format(pinverse_loss))
        #self.summarize_train(loss_info, params, cum_loss=loss, avg_acc=acc)
        return batch_idx + 1

    def train_step(self,
                   inputs,
                   num_particles=None,
                   entropy_regularization=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            num_particles (int): number of sampled particles. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        if num_particles is None:
            num_particles = self._num_particles
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization
        
        data, target = inputs
        if self._function_vi:
            loss_func = functools.partial(self._function_neglogprob,
                                        target.view(-1))
            transform_func = functools.partial(self._function_transform,
                                               data)
        else:
            loss_func = functools.partial(self._neglogprob, inputs)
            transform_func = None

        return self._generator.train_step(
            inputs=None,
            loss_func=loss_func,
            batch_size=num_particles,
            entropy_regularization=entropy_regularization,
            transform_func=transform_func,
            state=())

    def evaluate(self, num_particles=None):
        """Evaluate on a randomly drawn network. """

        assert self._test_loader is not None, "Must set test_loader first."
        logging.info("==> Begin testing")
        if self._use_fc_bn:
            self._generator.eval()
        params = self.sample_parameters(num_particles=num_particles)
        if self._functional_gradient:
            params, _ = params
        self._param_net.set_parameters(params)
        if self._use_fc_bn:
            self._generator.train()
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._param_net(data.double())  # [B, N, D]
                #output, _ = self._param_net(data)  # [B, N, D]
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
    
    def _function_transform(self, data, params):
        """
        Transform the generator outputs to its corresponding function values
        evaluated on the training batch. Used when function_vi is True.

        Args:
            data (torch.Tensor): training batch input.
            params (torch.Tensor): sampled outputs from the generator.

        Returns:
            outputs (torch.Tensor): outputs of param_net under params
                evaluated on data.
            desity_outputs (torch.Tensor): outputs of param_net under params
                evaluated on perturbed samples from the training batch
        """
        num_particles = params.shape[0]
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(data)  # [B, P, D]
        outputs = outputs.transpose(0, 1)
        outputs = outputs.view(num_particles, -1)  # [P, B * D]

        samples = data[-self._function_space_samples:]
        noise = torch.zeros_like(samples).uniform_(-3, 4)
        perturbed_samples = noise#samples * noise
        density_outputs, _ = self._param_net(perturbed_samples)
        density_outputs = density_outputs.transpose(0, 1)
        density_outputs = density_outputs.view(num_particles, -1)

        return outputs, density_outputs

    def _function_neglogprob(self, targets, outputs):
        """
        Function computing negative log_prob loss for function outputs.
        Used when function_vi is True.

        Args:
            targets (torch.Tensor): target values of the training batch.
            outputs (torch.Tensor): function outputs to evaluate the loss.

        Returns:
            negative log_prob for outputs evaluated on current training batch.
        """
        num_particles = outputs.shape[0]
        targets = targets.unsqueeze(0).expand(num_particles, *targets.shape)

        return self._loss_func(outputs, targets)

    def _neglogprob(self, inputs, params):
        """
        Function computing negative log_prob loss for generator outputs.
        Used when function_vi is False.

        Args:
            inputs (torch.Tensor): (data, target) of training batch.
            params (torch.Tensor): generator outputs to evaluate the loss.

        Returns:
            negative log_prob for params evaluated on current training batch.
        """
        self._param_net.set_parameters(params)
        num_particles = params.shape[0]
        data, target = inputs
        output, _ = self._param_net(data.double())  # [B, P, D]
        #output, _ = self._param_net(data)  # [B, P, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], num_particles,
                                            *target.shape[1:])
        return self._loss_func(output, target)


    def eval_uncertainty(self, num_particles=None):
        # Soft voting for now
        if num_particles is None:
            num_particles = 100
        params = self.sample_parameters(num_particles=num_particles)
        if self._functional_gradient:
            params, _ = params
        if self._generator._par_vi == 'minmax':
            params = params[0]
        self._param_net.set_parameters(params)
        with torch.no_grad():
            outputs = self._predict_dataset(
                self._test_loader,
                num_particles)
        probs = F.softmax(outputs, -1).mean(0)
        entropy = entropy_fn(probs.T.cpu().detach().numpy())
        with torch.no_grad():
            outputs_outlier = self._predict_dataset(
                self._outlier_test,
                num_particles)
        probs_outlier = F.softmax(outputs_outlier, -1).mean(0)
        entropy_outlier = entropy_fn(probs_outlier.T.cpu().detach().numpy())
        auroc_entropy = self._auc_score(entropy, entropy_outlier)
        logging.info("AUROC score: {}".format(auroc_entropy))

