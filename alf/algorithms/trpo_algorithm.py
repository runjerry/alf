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
"""TRPO algorithm."""

from collections import namedtuple
import gin
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.actor_critic_algorithm import ActorCriticState, ActorCriticInfo
from alf.algorithms.trpo_loss import TRPOLoss
from alf.data_structures import TimeStep, AlgStep, namedtuple, Experience
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, value_ops, dist_utils
from .config import TrainerConfig

TRPOInfo = namedtuple(
    "TRPOInfo", [
        "action_distribution", "value", "returns", "advantages",
        "prev_actor_state"
    ],
    default_value=())


@alf.configurable
class TRPOAlgorithm(ActorCriticAlgorithm):
    """TRPO Algorithm.
    Implement the simplified surrogate loss in equation (9) of "Trust Region
    Policy Optimization" https://arxiv.org/abs/1502.05477

    It works with ``trpo_loss.TRPOLoss``. 
    """

    def is_on_policy(self):
        return False

    def preprocess_experience(self, exp: Experience):
        """Compute advantages and put it into exp.rollout_info."""
        advantages = value_ops.generalized_advantage_estimation(
            rewards=exp.reward,
            values=exp.rollout_info.value,
            step_types=exp.step_type,
            discounts=exp.discount * self._loss._gamma,
            td_lambda=self._loss._lambda,
            time_major=False)
        advantages = torch.cat([
            advantages,
            torch.zeros(*advantages.shape[:-1], 1, dtype=advantages.dtype)
        ],
                               dim=-1)
        returns = exp.rollout_info.value + advantages
        return exp._replace(
            rollout_info=TRPOInfo(
                action_distribution=exp.rollout_info.action_distribution,
                returns=returns,
                advantages=advantages))

    def rollout_step(self, time_step: TimeStep, state: ActorCriticState):
        """Rollout for one step."""
        value, value_state = self._value_network(
            time_step.observation, state=state.value)

        action_distribution, actor_state = self._actor_network(
            time_step.observation, state=state.actor)

        action = dist_utils.sample_action_distribution(action_distribution)
        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state, value=value_state),
            info=TRPOInfo(
                action_distribution=action_distribution,
                value=value,
                prev_actor_state=state.actor))

    def calc_loss(self, experience, train_info: TRPOInfo):
        """Calculate loss."""
        return self._loss(self._actor_network, experience, train_info)

    def _trainable_attributes_to_ignore(self):
        return ['_actor_network']
