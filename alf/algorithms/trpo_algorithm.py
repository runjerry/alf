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
from alf.utils import common, value_ops, dist_utils, tensor_utils
from .config import TrainerConfig

TRPOInfo = namedtuple(
    "TRPOInfo", [
        "step_type", "discount", "reward", "action",
        "rollout_action_distribution", "returns", "advantages",
        "action_distribution", "value", "reward_weights", "observation",
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

    @property
    def on_policy(self):
        return False

    def train_step(self, inputs: TimeStep, state, rollout_info):
        alg_step = self._rollout_step(inputs, state)
        return alg_step._replace(
            info=rollout_info._replace(
                step_type=alg_step.info.step_type,
                reward=alg_step.info.reward,
                discount=alg_step.info.discount,
                action_distribution=alg_step.info.action_distribution,
                value=alg_step.info.value,
                reward_weights=alg_step.info.reward_weights))

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        if rollout_info.reward.ndim == 3:
            # [B, T, D] or [B, T, 1]
            discounts = rollout_info.discount.unsqueeze(-1) * self._loss.gamma
        else:
            # [B, T]
            discounts = rollout_info.discount * self._loss.gamma

        advantages = value_ops.generalized_advantage_estimation(
            rewards=rollout_info.reward,
            values=rollout_info.value,
            step_types=rollout_info.step_type,
            discounts=discounts,
            td_lambda=self._loss._lambda,
            time_major=False)
        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)

        returns = rollout_info.value + advantages
        return root_inputs, TRPOInfo(
            rollout_action_distribution=rollout_info.action_distribution,
            returns=returns,
            action=rollout_info.action,
            observation=rollout_info.observation,
            advantages=advantages)

    def rollout_step(self, inputs: TimeStep, state: ActorCriticState):
        """Rollout for one step."""
        alg_step = super().rollout_step(inputs, state)

        return alg_step._replace(
            info=TRPOInfo(
                action=alg_step.info.action,
                value=alg_step.info.value,
                step_type=alg_step.info.step_type,
                reward=alg_step.info.reward,
                discount=alg_step.info.discount,
                action_distribution=alg_step.info.action_distribution,
                reward_weights=alg_step.info.reward_weights,
                observation=inputs.observation,
                prev_actor_state=state.actor))

    def calc_loss(self, info: TRPOInfo):
        """Calculate loss."""
        return self._loss(self._actor_network, info)

    def _trainable_attributes_to_ignore(self):
        return ['_actor_network']
