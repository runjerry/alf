# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Loss for TRPO algorithm."""

from collections import namedtuple

import math
import numpy as np
import torch
import torch.nn as nn

import alf

from alf.data_structures import LossInfo
from alf.algorithms.actor_critic_loss import _normalize_advantages
import alf.nest as nest
from alf.utils.losses import element_wise_squared_loss
from alf.utils import common, dist_utils, tensor_utils, value_ops

TRPOLossInfo = namedtuple("TRPOLossInfo", ["td_loss", "neg_entropy"])


@alf.configurable
class TRPOLoss(nn.Module):
    """TRPO loss."""

    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=False,
                 td_lambda=0.95,
                 use_td_lambda_return=True,
                 normalize_advantages=True,
                 advantage_clip=None,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 importance_ratio_clipping=0.2,
                 log_prob_clipping=0.0,
                 delta_kl=0.01,
                 damping=0.01,
                 check_numerics=False,
                 debug_summaries=False,
                 name="TRPOLoss"):
        """
        Implement the simplified surrogate loss in equation (9) of `Proximal
        Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_.

        The total loss equals to

        .. code-block:: python

            (policy_gradient_loss      # (L^{CLIP} in equation (9))
            + td_loss_weight * td_loss # (L^{VF} in equation (9))
            - entropy_regularization * entropy)

        This loss works with ``PPOAlgorithm``. The advantages and returns are
        pre-computed by ``PPOAlgorithm.preprocess()``. One known difference with
        `baselines.ppo2` is that value estimation is not clipped here, while
        `baselines.ppo2` also clipped value if it deviates from returns too
        much.

        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_advantages (bool): If True, normalize advantage to zero
                mean and unit variance within batch for caculating policy
                gradient.
            advantage_clip (float): If set, clip advantages to :math:`[-x, x]`
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
            importance_ratio_clipping (float):  Epsilon in clipped, surrogate
                PPO objective. See the cited paper for more detail.
            log_prob_clipping (float): If >0, clipping log probs to the range
                ``(-log_prob_clipping, log_prob_clipping)`` to prevent ``inf/NaN``
                values.
            delta_kl (float): KL-divergence bound between behavior policy and 
                current policy.
            check_numerics (bool):  If true, checking for ``NaN/Inf`` values. For
                debugging only.
        """
        super().__init__()

        self._td_loss_weight = td_loss_weight
        self._name = name
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._use_gae = use_gae
        self._lambda = td_lambda
        self._use_td_lambda_return = use_td_lambda_return
        self._normalize_advantages = normalize_advantages
        assert advantage_clip is None or advantage_clip > 0, (
            "Clipping value should be positive!")
        self._advantage_clip = advantage_clip
        self._entropy_regularization = entropy_regularization
        self._debug_summaries = debug_summaries

        self._importance_ratio_clipping = importance_ratio_clipping
        self._log_prob_clipping = log_prob_clipping
        self._delta_kl = delta_kl
        self._damping = damping
        self._check_numerics = check_numerics

    def forward(self, actor_network, experience, train_info):
        """Cacluate actor critic loss. The first dimension of all the tensors is
        time dimension and the second dimesion is the batch dimension.

        Args:
            experience (nest): experience used for training. All tensors are
                time-major.
            train_info (nest): information collected for training. It is batched
                from each ``AlgStep.info`` returned by ``rollout_step()``
                (on-policy training) or ``train_step()`` (off-policy training).
                All tensors in ``train_info`` are time-major.
        Returns:
            LossInfo: with ``extra`` being ``TRPOLossInfo``.
        """

        value = train_info.value
        returns, advantages = self._calc_returns_and_advantages(
            experience, value)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("values", value.mean())
                alf.summary.scalar("returns", returns.mean())
                alf.summary.scalar("advantages/mean", advantages.mean())
                alf.summary.histogram("advantages/value", advantages)
                alf.summary.scalar(
                    "explained_variance_of_return_by_value",
                    tensor_utils.explained_variance(value, returns))

        if self._normalize_advantages:
            advantages = _normalize_advantages(advantages)

        if self._advantage_clip:
            advantages = torch.clamp(advantages, -self._advantage_clip,
                                     self._advantage_clip)

        self._policy_update(actor_network, experience, train_info, advantages)

        td_loss = self._td_error_loss_fn(returns.detach(), value)

        loss = self._td_loss_weight * td_loss

        entropy_loss = ()
        if self._entropy_regularization is not None:
            entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
                train_info.action_distribution, return_sum=False)
            entropy_loss = alf.nest.map_structure(lambda x: -x, entropy)
            loss -= self._entropy_regularization * sum(
                alf.nest.flatten(entropy_for_gradient))

        return LossInfo(
            loss=loss,
            extra=TRPOLossInfo(td_loss=td_loss, neg_entropy=entropy_loss))

    def _policy_update(self, actor_network, experience, train_info,
                       advantages):
        scope = alf.summary.scope(self.__class__.__name__)
        policy_distribution = train_info.action_distribution
        collect_policy_distribution = experience.rollout_info.action_distribution
        action = experience.action
        sample_action_log_prob = dist_utils.compute_log_probability(
            collect_policy_distribution, action).detach()
        action_log_prob = dist_utils.compute_log_probability(
            policy_distribution, action)
        importance_ratio = (action_log_prob - sample_action_log_prob).exp()
        pg_loss = -importance_ratio * advantages
        observation = experience.observation
        observation = observation.reshape(-1, observation.shape[-1])

        def _Fvp(v):
            action_dist, _ = actor_network(observation,
                                           train_info.prev_actor_state)
            normal_dist = action_dist.base_dist
            mean = normal_dist.mean
            std = normal_dist.stddev
            log_std = std.log()
            fixed_mean = mean.detach()
            fixed_std = std.detach()
            fixed_log_std = log_std.detach()
            kl = log_std - fixed_log_std + (fixed_std.pow(2) + (fixed_mean - mean).pow(2)) \
                / (2.0 * std.pow(2)) - 0.5
            kl = kl.sum(1, keepdim=True)
            kl_grads = torch.autograd.grad(
                kl.mean(), actor_network.parameters(), create_graph=True)
            kl_grads_flat = torch.cat([grad.view(-1) for grad in kl_grads])
            kl_v = kl_grads_flat * v
            kl_hess_v = torch.autograd.grad(kl_v.sum(),
                                            actor_network.parameters())
            kl_hess_v_flat = torch.cat(
                [grad.contiguous().view(-1) for grad in kl_hess_v]).detach()
            return kl_hess_v_flat + v * self._damping

        def _pg_loss(enable_grad=True):
            with torch.set_grad_enabled(enable_grad):
                action_distribution, _ = actor_network(
                    observation, train_info.prev_actor_state)
            action_log_prob = dist_utils.compute_log_probability(
                policy_distribution, action)
            importance_ratio = (action_log_prob - sample_action_log_prob).exp()
            pg_loss = -importance_ratio * advantages
            return pg_loss.mean()

        grads = torch.autograd.grad(
            pg_loss.mean(), actor_network.parameters(), retain_graph=True)
        grads_flat = torch.cat([grad.view(-1) for grad in grads]).detach()
        cg_grads_flat = common.conjugate_gradient(
            _Fvp, -grads_flat, max_steps=10)
        gHg = cg_grads_flat.dot(_Fvp(cg_grads_flat))

        pg_step = math.sqrt(2 * self._delta_kl / gHg) * cg_grads_flat
        expected_improve = -grads_flat.dot(pg_step)

        cur_params_flat = self._get_actor_params_flat(actor_network)
        success, new_params_flat = self._line_search(actor_network, _pg_loss,
                                                     cur_params_flat, pg_step,
                                                     expected_improve)
        if success:
            self._set_actor_flat_params(actor_network, new_params_flat)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with scope:
                alf.summary.histogram('pg_loss', pg_loss)

    def _line_search(self,
                     actor_network,
                     loss_fn,
                     init_x,
                     fullstep,
                     expected_improve_rate,
                     max_backtracks=10,
                     accept_ratio=.1):
        loss = loss_fn(enable_grad=False)
        x = init_x
        for step_frac in [.5**t for t in range(max_backtracks)]:
            x_new = x + step_frac * fullstep
            self._set_actor_flat_params(actor_network, x_new)
            loss_new = loss_fn(enable_grad=False)
            improve = loss - loss_new
            expected_improve = expected_improve_rate * step_frac
            ratio = improve / expected_improve
            if ratio > accept_ratio and improve > 0:
                return True, x_new

        return False, x

    def _get_actor_params_flat(self, actor_network):
        params = []
        for param in actor_network.parameters():
            params.append(param.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def _set_actor_flat_params(self, actor_network, flat_params):
        prev_ind = 0
        for param in actor_network.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(
                param.size()))
            prev_ind += flat_size

    def _calc_returns_and_advantages(self, experience, value):
        advantages = experience.rollout_info.advantages
        returns = experience.rollout_info.returns
        return returns, advantages
