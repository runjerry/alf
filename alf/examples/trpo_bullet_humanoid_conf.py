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

from functools import partial
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.environments.suite_gym import wrap_env
from alf.networks import NormalProjectionNetwork, ActorDistributionNetwork, ValueNetwork
from alf.optimizers import Adam, AdamTF
from alf.utils.losses import element_wise_huber_loss

import pybullet_envs
from alf.nest.utils import NestConcat
from alf.optimizers import Adam, AdamTF
from alf.utils.losses import element_wise_squared_loss

from alf.examples import trpo_conf

env_name = "HumanoidBulletEnv-v0"

# environment config
alf.config(
    'create_environment', env_name=env_name, num_parallel_environments=96)

alf.config('wrap_env', clip_action=False)

# algorithm config
fc_layer_params = (32, 32, 32)

actor_network_ctor = partial(
    ActorDistributionNetwork,
    fc_layer_params=fc_layer_params,
    activation=torch.tanh,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        projection_output_init_gain=1e-5,
        std_bias_initializer_value=0.0,
        std_transform=torch.exp))

value_network_ctor = partial(
    ValueNetwork, fc_layer_params=fc_layer_params, activation=torch.tanh)

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=actor_network_ctor,
    value_network_ctor=value_network_ctor)

alf.config(
    'TRPOLoss',
    # entropy_regularization=1e-2,
    gamma=0.99,
    td_lambda=0.95,
    td_error_loss_fn=element_wise_squared_loss,
    normalize_advantages=True)

# training config
alf.config(
    'Agent',
    optimizer=AdamTF(lr=3e-4, gradient_clipping=0.5, clip_by_global_norm=True))

alf.config(
    'TrainerConfig',
    mini_batch_length=1,
    unroll_length=512,
    mini_batch_size=4096,
    num_updates_per_train_iter=20,
    num_iterations=1000,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=100,
    # num_eval_episodes=5,
    debug_summaries=True,
    random_seed=0,
    summarize_grads_and_vars=True,
    summary_interval=10)
