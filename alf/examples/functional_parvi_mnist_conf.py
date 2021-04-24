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

import alf
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.utils.datagen import load_mnist

# dataset config
data_creator = partial(
    load_mnist, label_idx=[0, 1, 2, 3, 4, 5], train_bs=50, test_bs=100)
data_creator_outlier = partial(
    load_mnist, label_idx=[6, 7, 8, 9], train_bs=100, test_bs=100)

# network architecture
CONV_LAYER_PARAMS = ((6, 5, 1, 2, 2), (16, 5, 1, 0, 2), (120, 5, 1))
FC_LAYER_PARAMS = ((84, True), )

# algorithm config
alf.config(
    'FuncParVIAlgorithm',
    data_creator=data_creator,
    data_creator_outlier=data_creator_outlier,
    par_vi='svgd',
    output_dim=6,
    num_particles=10,
    conv_layer_params=CONV_LAYER_PARAMS,
    fc_layer_params=FC_LAYER_PARAMS,
    optimizer=alf.optimizers.Adam(lr=1e-3, weight_decay=1e-4),
    critic_optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4),
    critic_hidden_layers=(512, 512),
    critic_iter_num=5,
    critic_l2_weight=10.0,
    num_train_classes=6,
    logging_training=True,
    logging_evaluate=True,
    entropy_regularization=1.0,
    loss_type='classification')

# training config
alf.config(
    'TrainerConfig',
    ml_type='sl',
    algorithm_ctor=FuncParVIAlgorithm,
    num_iterations=100,
    num_checkpoints=1,
    evaluate=True,
    eval_uncertainty=True,
    eval_interval=1,
    summary_interval=1,
    debug_summaries=True,
    summarize_grads_and_vars=True)
