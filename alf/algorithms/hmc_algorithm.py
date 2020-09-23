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

HMCLossInfo = namedtuple("HMCLossInfo", ["sampler"])


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
             log_prob_func,
             params,
             num_samples=10,
             steps_per_sample=10,
             step_size=0.1,
             burn_in_steps=0,
             inv_mass=None,
             name="HMC"):
        r"""Instantiate an HMC Sampler.

        Args:
            log_prob_func (Callable):
            params (torch.tensor):
            num_samples (int):
            steps_per_sample (int):
            step_size (int):
            burn_in_steps (int):
            inv_mass (float):
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


    def sample(self, params_init, num_samples=10):

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

