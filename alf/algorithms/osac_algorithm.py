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
"""Optimistic Soft Actor Critic algorithm."""

from absl import logging
import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops

OsacActionState = namedtuple(
    "OsacActionState", ["actor_network", "explorer_network"], default_value=())

OsacCriticState = namedtuple("OsacCriticState", ["critics", "target_critics"])

OsacState = namedtuple(
    "OsacState", ["action", "actor", "explorer", "critic"], default_value=())

OsacCriticInfo = namedtuple("OsacCriticInfo", ["critics", "target_critic"])

OsacActorInfo = namedtuple(
    "OsacActorInfo", ["actor_loss", "neg_entropy"], default_value=())

OsacExplorerInfo = namedtuple(
    "OsacExplorerInfo", ["explorer_loss", "neg_entropy"], default_value=())

OsacInfo = namedtuple(
    "OsacInfo", [
        "action_distribution", "actor", "explorer", "critic", "alpha",
        "explore_alpha", "log_pi", "log_pi_explore"
    ],
    default_value=())

OsacLossInfo = namedtuple(
    'OsacLossInfo', ('actor', 'explorer', 'critic', 'alpha', 'explore_alpha'))


def _set_target_entropy(name, target_entropy, flat_action_spec):
    """A helper function for computing the target entropy under different
    scenarios of ``target_entropy``.

    Args:
        name (str): the name of the algorithm that calls this function.
        target_entropy (float|Callable|None): If a floating value, it will return
            as it is. If a callable function, then it will be called on the action
            spec to calculate a target entropy. If ``None``, a default entropy will
            be calculated.
        flat_action_spec (list[TensorSpec]): a flattened list of action specs.
    """
    if target_entropy is None or callable(target_entropy):
        if target_entropy is None:
            target_entropy = dist_utils.calc_default_target_entropy
        target_entropy = np.sum(list(map(target_entropy, flat_action_spec)))
        logging.info("Target entropy is calculated for {}: {}.".format(
            name, target_entropy))
    else:
        logging.info("User-supplied target entropy for {}: {}".format(
            name, target_entropy))
    return target_entropy


@alf.configurable
class OsacAlgorithm(OffPolicyAlgorithm):
    r"""Optimistic Soft Actor Critic algorithm:

    Target policy is trained exactly the same as SAC, while another policy
    is trained with :math:'max(Q_1, Q_2)' for exploration only.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explorer_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 reward_weights=None,
                 use_entropy_reward=True,
                 use_parallel_network=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 beta_ub=4.6,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 explorer_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 explore_alpha_optimizer=None,
                 debug_summaries=False,
                 name="OsacAlgorithm"):
        """
        Refer to SacAlgorithm for Args besides the following.

        Args:
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`    
            explorer_optimizer (torch.optim.optimizer): The optimizer for explorer.
            explore_alpha_optimizer (torch.optim.optimizer): The optimizer for 
                explorer alpha.
        """
        assert action_spec.is_continuous, "Only continuous action is supported"

        self._num_critic_replicas = num_critic_replicas
        self._use_parallel_network = use_parallel_network

        critic_networks, actor_network, explorer_network = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            explorer_network_cls, critic_network_cls)

        self._use_entropy_reward = use_entropy_reward

        def _init_log_alpha():
            return nn.Parameter(torch.tensor(float(initial_log_alpha)))

        log_alpha = _init_log_alpha()
        log_explore_alpha = _init_log_alpha()

        action_state_spec = OsacActionState(
            actor_network=actor_network.state_spec,
            explorer_network=explorer_network.state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=OsacState(
                action=action_state_spec,
                actor=critic_networks.state_spec,
                explorer=critic_networks.state_spec,
                critic=OsacCriticState(
                    critics=critic_networks.state_spec,
                    target_critics=critic_networks.state_spec)),
            predict_state_spec=OsacState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if explorer_optimizer is not None:
            self.add_optimizer(explorer_optimizer, [explorer_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, nest.flatten(log_alpha))
        if explore_alpha_optimizer is not None:
            self.add_optimizer(explore_alpha_optimizer,
                               nest.flatten(log_explore_alpha))

        self._log_alpha = log_alpha
        self._log_explore_alpha = log_explore_alpha
        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(float(max_log_alpha))
        else:
            self._max_log_alpha = None

        self._actor_network = actor_network
        self._explorer_network = explorer_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._prior_actor = None
        if prior_actor_ctor is not None:
            self._prior_actor = prior_actor_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            total_action_dims = sum(
                [spec.numel for spec in alf.nest.flatten(action_spec)])
            self._target_entropy = -target_kld_per_dim * total_action_dims
        else:
            self._target_entropy = _set_target_entropy(
                self.name, target_entropy, nest.flatten(self._action_spec))

        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

        self._beta_ub = beta_ub

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, explorer_network_cls,
                       critic_network_cls):
        def _make_parallel(net):
            if self._use_parallel_network:
                nets = net.make_parallel(self._num_critic_replicas)
            else:
                nets = alf.networks.NaiveParallelNetwork(
                    net, self._num_critic_replicas)
            return nets

        assert actor_network_cls is not None, (
            "ActorNetwork must be provided!")
        actor_network = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        assert explorer_network_cls is not None, (
            "ExplorerNetwork must be provided!")
        explorer_network = explorer_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        assert critic_network_cls is not None, (
            "CriticNetwork must be provided!")
        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            output_tensor_spec=reward_spec)
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network, explorer_network

    def _predict_action(self,
                        observation,
                        state: OsacActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False):
        """
        Differences between SacAlgorithm._predict_action:

        1. Only continuous actions are supported.

        2. Add a switch for explore mode where an explore policy is used. 
        """
        new_state = OsacActionState()
        if explore:
            action_dist, explorer_network_state = self._explorer_network(
                observation, state=state.explorer_network)
            new_state = new_state._replace(
                explorer_network=explorer_network_state)
        else:
            action_dist, actor_network_state = self._actor_network(
                observation, state=state.actor_network)
            new_state = new_state._replace(actor_network=actor_network_state)

        if eps_greedy_sampling:
            action = dist_utils.epsilon_greedy_sample(action_dist,
                                                      epsilon_greedy)
        else:
            action = dist_utils.rsample_action_distribution(action_dist)
        return action_dist, action, new_state

    def predict_step(self,
                     time_step: TimeStep,
                     state: OsacState,
                     epsilon_greedy=1.0):
        action_dist, action, action_state = self._predict_action(
            time_step.observation,
            state=state.action,
            epsilon_greedy=epsilon_greedy,
            eps_greedy_sampling=True)
        return AlgStep(
            output=action,
            state=OsacState(action=action_state),
            info=OsacInfo(action_distribution=action_dist))

    def rollout_step(self, time_step: TimeStep, state: OsacState):
        """Same as SacAlgorithm.rollout_step except that `explore` is set to be
        `self._explore` when calling `_predict_action`.
        """
        action_dist, action, action_state = self._predict_action(
            time_step.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            explore=True)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, time_step.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, time_step.observation, action,
                state.critic.target_critics)
            critic_state = OsacCriticState(
                critics=critics_state, target_critics=target_critics_state)
            actor_state = critics_state
            explorer_state = critics_state
        else:
            actor_state = state.actor
            explorer_state = state.explorer
            critic_state = state.critic

        new_state = OsacState(
            action=action_state,
            actor=actor_state,
            explorer=explorer_state,
            critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=OsacInfo(action_distribution=action_dist))

    def _compute_critics(self, critic_net, observation, action, critics_state):
        observation = (observation, action)
        # critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)
        return critics, critics_state

    def _actor_train_step(self, exp: Experience, state, action, log_pi):
        neg_entropy = sum(nest.flatten(log_pi))

        critics, critics_state = self._compute_critics(
            self._critic_networks, exp.observation, action, state)

        if self.has_multidim_reward():
            # Multidimensional reward: [B, replicas, reward_dim]
            critics = critics * self.reward_weights
        # min over replicas
        q_value = critics.min(dim=1)[0]

        cont_alpha = torch.exp(self._log_alpha).detach()

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_info = LossInfo(
            loss=actor_loss + cont_alpha * log_pi,
            extra=OsacActorInfo(
                actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _explorer_train_step(self, exp: Experience, state, action,
                             log_pi_explore):
        neg_entropy = sum(nest.flatten(log_pi_explore))

        critics, critics_state = self._compute_critics(
            self._critic_networks, exp.observation, action, state)

        if self.has_multidim_reward():
            # Multidimensional reward: [B, replicas, reward_dim]
            critics = critics * self.reward_weights
        # min over replicas
        q_value = critics.max(dim=1)[0]
        cont_alpha = torch.exp(self._log_explore_alpha).detach()

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def explorer_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        explorer_loss = nest.map_structure(explorer_loss_fn, dqda, action)
        explorer_loss = math_ops.add_n(nest.flatten(explorer_loss))
        explorer_info = LossInfo(
            loss=explorer_loss + cont_alpha * log_pi_explore,
            extra=OsacExplorerInfo(
                explorer_loss=explorer_loss, neg_entropy=neg_entropy))
        return critics_state, explorer_info

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape ``[batch_size, replicas, num_actions]``.
        Returns:
            Tensor: selected Q values with shape ``[batch_size, replicas]``.
        """
        # action shape: [batch_size] -> [batch_size, n, 1]
        action = action.view(q_values.shape[0], 1, -1).expand(
            -1, q_values.shape[1], -1).long()
        return q_values.gather(-1, action).squeeze(-1)

    def _critic_train_step(self, exp: Experience, state: OsacCriticState,
                           action):
        critics, critics_state = self._compute_critics(
            self._critic_networks, exp.observation, exp.action, state.critics)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks, exp.observation, action,
            state.target_critics)

        if self.has_multidim_reward():
            sign = self.reward_weights.sign()
            target_critics = (target_critics * sign).min(dim=1)[0] * sign
        else:
            target_critics = target_critics.min(dim=1)[0]

        target_critic = target_critics.reshape(exp.reward.shape)

        target_critic = target_critic.detach()

        state = OsacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = OsacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi):
        alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(), self._log_alpha, log_pi,
            self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    def _explore_alpha_train_step(self, log_pi_explore):
        explore_alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(), self._log_explore_alpha,
            log_pi_explore, self._target_entropy)
        return sum(nest.flatten(explore_alpha_loss))

    def train_step(self, exp: Experience, state: OsacState):
        (action_distribution, action, action_state) = self._predict_action(
            exp.observation, state=state.action)

        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_distribution, action)

        (explore_action_distribution, explore_action,
         explore_action_state) = self._predict_action(
             exp.observation, state=state.action, explore=True)
        action_state._replace(
            explorer_network=explore_action_state.explorer_network)

        log_pi_explore = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                            explore_action_distribution,
                                            explore_action)

        log_pi = sum(nest.flatten(log_pi))
        log_pi_explore = sum(nest.flatten(log_pi_explore))

        if self._prior_actor is not None:
            prior_step = self._prior_actor.train_step(exp, ())
            log_prior = dist_utils.compute_log_probability(
                prior_step.output, action)
            log_pi = log_pi - log_prior
            log_pi_explore = log_pi_explore - log_prior

        actor_state, actor_loss = self._actor_train_step(
            exp, state.actor, action, log_pi)
        explorer_state, explorer_loss = self._explorer_train_step(
            exp, state.explorer, explore_action, log_pi_explore)
        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action)
        alpha_loss = self._alpha_train_step(log_pi)
        explore_alpha_loss = self._explore_alpha_train_step(log_pi_explore)

        state = OsacState(
            action=action_state,
            actor=actor_state,
            explorer=explorer_state,
            critic=critic_state)
        info = OsacInfo(
            action_distribution=action_distribution,
            actor=actor_loss,
            explorer=explorer_loss,
            critic=critic_info,
            alpha=alpha_loss,
            explore_alpha=explore_alpha_loss,
            log_pi=log_pi,
            log_pi_explore=log_pi_explore)
        return AlgStep(action, state, info)

    def after_update(self, experience, train_info: OsacInfo):
        self._update_target()
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_explore_alpha)

    def calc_loss(self, experience, train_info: OsacInfo):
        critic_loss = self._calc_critic_loss(experience, train_info)
        alpha_loss = train_info.alpha
        explore_alpha_loss = train_info.explore_alpha
        actor_loss = train_info.actor
        explorer_loss = train_info.explorer

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())
                alf.summary.scalar("explore_alpha",
                                   self._log_explore_alpha.exp())

        return LossInfo(
            loss=math_ops.add_ignore_empty(
                actor_loss.loss + explorer_loss.loss,
                critic_loss.loss + alpha_loss + explore_alpha_loss),
            priority=critic_loss.priority,
            extra=OsacLossInfo(
                actor=actor_loss.extra,
                explorer=explorer_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss,
                explore_alpha=explore_alpha_loss))

    def _calc_critic_loss(self, experience, train_info: OsacInfo):
        # We need to put entropy reward in ``experience.reward`` instead of
        # ``target_critics`` because in the case of multi-step TD learning,
        # the entropy should also appear in intermediate steps!
        # This doesn't affect one-step TD loss, however.
        if self._use_entropy_reward:
            with torch.no_grad():
                entropy_reward = nest.map_structure(
                    lambda la, lp: -torch.exp(la) * lp, self._log_alpha,
                    train_info.log_pi)
                entropy_reward = sum(nest.flatten(entropy_reward))
                gamma = self._critic_losses[0].gamma
                experience = experience._replace(
                    reward=experience.reward + entropy_reward * gamma)

        critic_info = train_info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(experience=experience,
                  value=critic_info.critics[:, :, i, ...],
                  target_value=critic_info.target_critic).loss)

        critic_loss = math_ops.add_n(critic_losses)

        if (experience.batch_info != ()
                and experience.batch_info.importance_weights is not ()):
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
