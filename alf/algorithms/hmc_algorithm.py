"""A generic generator."""

from absl import logging
import gin
import numpy as np
import torch
import torch.nn as nn

from alf.algorithms.algorithm import Algorithm
from alf.algorithms.mi_estimator import MIEstimator
from alf.data_structures import AlgStep, LossInfo, namedtuple
import alf.nest as nest
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
import types
from collections import OrderedDict


HMCLossInfo = namedtuple("HMCLossInfo", ["sampler"])


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


@gin.configurable
class HMC(Algorithm):
    """Hamiltonian Monte Carlo

    MCMC method for obtaining a sequence of random variables that
    converge to a target distribution. 
    HMC corresponds to the MH algorithm, that evolves according to 
    hamiltonian dynamics. Where the MH algorithm uses a proposal distribution
    given by a Gaussian random walk, HMC proposes samples with high acceptance
    probability due to the energy conserving properties of hamiltonian dynamics.

    The algorithm is as follows:
    
    1. Sample momentum by doing Gibbs sampling
    2. perform N leapfrog steps to move to new state
    3. accept/reject sample from new state

    
    """
    def __init__(self,
             log_prob_func=None,
             params=None,
             num_samples=10,
             steps_per_sample=10,
             step_size=0.1,
             burn_in_steps=0,
             inv_mass=None,
             model=None,
             model_loss='regression',
             tau_list=None,
             tau_out=.1,
             name="HMC"):
        r"""Instantiate an HMC Sampler.

        Args for training an HMC sampler
            log_prob_func (Callable):
            params (torch.tensor):
            num_samples (int):
            steps_per_sample (int):
            step_size (int):
            burn_in_steps (int):
            inv_mass (float):
        ===================================================================
        Args for training a Bayesian neural network with HMC
            model (nn.module):
            tau_list (list):
            tau_out (float):
            name (str): name of this HMC sampler
        """
        super().__init__(train_state_spec=(), name=name)
        self._log_prob_func = log_prob_func
        self._params_init = params
        self._num_samples = num_samples
        self._step_size = step_size
        self._steps_per_sample = steps_per_sample
        self._burn_in_steps = burn_in_steps
        self._inv_mass = inv_mass
        self._model = model
        self._model_loss = model_loss
        self._tau_list = tau_list
        self._tau_out = tau_out
 
    def set_log_prob_func(self, fn):
        self._log_prob_func = fn

    def _collect_gradients(self, log_prob, params):
        if isinstance(log_prob, tuple):
            log_prob[0].backward()
            params_list = list(log_prob[1])
            params = torch.cat([p.flatten() for p in params_list])
            params.grad = torch.cat([p.grad.flatten() for p in params_list])
        else:
            params.grad = torch.autograd.grad(log_prob,params)[0]
        return params

    def _sample_momentum_gibbs(self, params, mass=None):
        """Gibbs sampling for momentum, given the mass.
            Sample from a normal distribution centered at 0 with stdev given 
            by the input mass"""
        if mass is None:
            dist = torch.distributions.Normal(
                torch.zeros_like(params),
                torch.ones_like(params))
        else:
            if mass.dim() == 2:
                dist = torch.distributions.MultivariateNormal(
                    torch.zeros_like(params),
                    mass)
            elif mass.dim() == 1:
                dist = torch.distributions.Normal(
                    torch.zeros_like(params),
                    mass)
        return dist.sample()

    def _leapfrog(self, momentum):

        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = self._log_prob_func(p)
            p = self._collect_gradients(log_prob, p)
            return p.grad

        ret_params = []
        ret_momenta = []
        momentum += 0.5 * self._step_size * params_grad(self._current_params)
        for n in range(self._steps_per_sample):
            if self._inv_mass is None:
                self._current_params = self._current_params + self._step_size * momentum
            else:
                #Assume G is diag here so 1/Mass = G inverse
                if len(self._inv_mass.shape) == 2:
                    pv = torch.matmul(self._inv_mass, momentum.view(-1,1)).view(-1)
                    self._current_params = self._current_params + self._step_size * pv
                else:
                    self._current_params = self.current_params + step_size * inv_mass * momentum
            p_grad = params_grad(self._current_params)
            momentum += self._step_size * p_grad
            ret_params.append(self._current_params.clone())
            ret_momenta.append(momentum.clone())
        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * self._step_size * p_grad.clone()
        return ret_params, ret_momenta


    def _acceptance(self, h_old, h_new):
        return float(-h_new + h_old)

    def _has_nan_or_inf(self, value):
        if torch.is_tensor(value):
            value = torch.sum(value)
            isnan = int(torch.isnan(value)) > 0
            isinf = int(torch.isinf(value)) > 0
            return isnan or isinf
        else:
            value = float(value)
            return (value == float('inf')) or (value == float('-inf')) or (value == float('NaN'))
 
    def _hamiltonian(self, params, momentum):
        log_prob = self._log_prob_func(params)
        if self._has_nan_or_inf(log_prob):
            logging.info('Invalid log_prob: {}, params: {}'.format(log_prob, params))
            return None

        potential = -log_prob
        if self._inv_mass is None:
            kinetic = 0.5 * torch.dot(momentum, momentum)
        else:
            if len(self._inv_mass.shape) == 2:
                kinetic = 0.5 * torch.matmul(
                    momentum.view(1,-1),
                    torch.matmul(inv_mass, momentum.view(-1,1))).view(-1)
            else:
                kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
        hamiltonian = potential + kinetic
        return hamiltonian


    def sample(self, params_init, num_samples=0):

        assert params_init.dim() == 1, "``params_init`` must be a 1d tensor."
        assert self._burn_in_steps <= num_samples, "``burn_in_steps`` must be less than "\
            "num_samples."

        # Invert mass matrix once (As mass is used in Gibbs resampling step)
        mass = None
        if self._inv_mass is not None:
            if len(self._inv_mass.shape) == 2:
                mass = torch.inverse(self._inv_mass)
            elif len(self._inv_mass.shape) == 1:
                mass = 1/self._inv_mass
        
        self._current_params = params_init.clone().requires_grad_()
        if num_samples <= 0:
            num_samples = self._num_samples
        
        ret_params = [self._current_params.clone()]
        self._num_rejected = 0.
        logging.info('Sampling (HMC; Implicit Integrator) {} Samples'.format(num_samples))

        def reject():
            self._num_rejected += 1
            self._current_params = ret_params[-1]
            return ret_params[-self._steps_per_sample:]

        for n in range(num_samples):
            momentum = self._sample_momentum_gibbs(self._current_params, mass=mass)
            H = self._hamiltonian(self._current_params, momentum)
            if H is None:  # NaN err
                l_params = reject()
                if n > self._burn_in_steps:
                    ret_params.extend(l_params)
                continue

            leapfrog_params, leapfrog_momenta = self._leapfrog(momentum)
            self._current_params = leapfrog_params[-1].detach().requires_grad_()

            momentum = leapfrog_momenta[-1]
            new_H = self._hamiltonian(self._current_params, momentum)
            if new_H is None:  # NaN err
                l_params = reject()
                if n > self._burn_in_steps:
                    ret_params.extend(l_params)
                continue

            rho = min(0., self._acceptance(H, new_H))

            if rho >= torch.log(torch.rand(1)):
                if n > self._burn_in_steps:
                    ret_params.extend(leapfrog_params)
            else:
                l_params = reject()
                if n > self._burn_in_steps:
                    ret_params.extend(l_params)

        logging.info('Acceptance Rate {:.2f}'.format(
            1 - self._num_rejected/num_samples))
        samples = list(map(lambda t: t.detach(), ret_params))
        return samples

    def _unflatten_to_model(self, params):
        if params.dim() != 1:
            raise ValueError('Expecting a 1d flattened_params')
        params_list = []
        i = 0
        for val in list(self._model.parameters()):
            length = val.nelement()
            param = params[i:i+length].view_as(val)
            params_list.append(param)
            i += length
        return params_list

    def _define_model_log_prob(self, x, y, param_list, param_shapes,
        predict=False):

        fmodel = self._make_functional(self._model)
        dist_list = []
        for tau in self._tau_list:
            dist_list.append(torch.distributions.Normal(
                torch.zeros_like(self._tau_list[0]), tau**-0.5))

        def log_prob_func(params):
            params_unflattened = self._unflatten_to_model(params)
            i_prev = 0
            l_prior = torch.zeros_like(params[0], requires_grad=True)
            for weights, index, shape, dist in zip(
                self._model.parameters(), param_list, param_shapes, dist_list):
                w = params[i_prev:index+i_prev]
                l_prior = dist.log_prob(w).sum() + l_prior
                i_prev += index

            # Sample prior if no data
            if x is None:
                return l_prior

            output = fmodel(x, params=params_unflattened)

            if self._model_loss is 'binary_class':
                crit = nn.BCEWithLogitsLoss(reduction='sum')
                ll = - self._tau_out *(crit(output, y))
            elif self._model_loss is 'classification':
                crit = nn.CrossEntropyLoss(reduction='sum')
                ll = - self._tau_out *(crit(output, y.long().view(-1)))
            elif self._model_loss is 'regression':
                ll = - 0.5 * self._tau_out * ((output - y) ** 2).sum(0)
            
            if predict:
                return ll + l_prior, output
            else:
                return ll + l_prior
        self.set_log_prob_func(log_prob_func)

    def sample_model(self, x, y):       
        param_shapes = []
        param_list = []
        build_tau = False
        if self._tau_list is None:
            self._tau_list = []
            build_tau = True
        for weights in self._model.parameters():
            param_shapes.append(weights.shape)
            param_list.append(weights.nelement())
            if build_tau:
                self._tau_list.append(1.)

        self._define_model_log_prob(x, y, param_list, param_shapes)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.sample(self._params_init)

    def predict_model(self, x, y, samples):      
        param_shapes = []
        param_list = []
        build_tau = False
        if self._tau_list is None:
            self._tau_list = []
            build_tau = True
        for weights in self._model.parameters():
            param_shapes.append(weights.shape)
            param_list.append(weights.nelement())
            if build_tau:
                self._tau_list.append(1.)

        self._define_model_log_prob(x, y, param_list, param_shapes,
            predict=True)

        pred_log_prob_list = []
        pred_list = []
        for sample in samples:
            logprob, pred = self._log_prob_func(sample)
            pred_log_prob_list.append(logprob.detach()) 
            pred_list.append(pred.detach())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return torch.stack(pred_list), pred_log_prob_list

    def _get_torch_functional(self, module, params_box, params_offset):
        self_ = Scope()
        num_params = len(module._parameters)
        param_names = list(module._parameters.keys())
        # Set dummy variable to bias_None to rename as flag if no bias
        if 'bias' in param_names and module._parameters['bias'] is None:
            param_names[-1] = 'bias_None' # Remove last name (hopefully bias) from list
        forward = type(module).forward
        _internal_attrs = {'_backend', '_parameters', '_buffers',
            '_backward_hooks', '_forward_hooks', '_forward_pre_hooks',
            '_modules'}

        for name, attr in module.__dict__.items():
            if name in _internal_attrs:
                continue   #If internal attributes skip
            setattr(self_, name, attr)

        child_params_offset = params_offset + num_params
        for name, child in module.named_children():
            child_params_offset, fchild = self._get_torch_functional(
                child, params_box, child_params_offset)
            self_._modules[name] = fchild  # fchild is functional child
            setattr(self_, name, fchild)
        def fmodule(*args, **kwargs):
            if 'bias_None' in param_names:
                params_box[0].insert(params_offset + 1, None)
            for name, param in zip(param_names,
                params_box[0][params_offset:params_offset + num_params]):
                if name == 'bias_None':
                    setattr(self_, 'bias', None)
                else:
                    setattr(self_, name, param)
            return forward(self_, *args) #, **kwargs)
        return child_params_offset, fmodule

    def _make_functional(self, module):
        params_box = [None]
        _, fmodule_func = self._get_torch_functional(module, params_box, 0)

        def fmodule(*args, **kwargs):
            params_box[0] = kwargs.pop('params')
            return fmodule_func(*args, **kwargs)
        return fmodule

