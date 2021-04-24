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
from alf.algorithms.batchensemble_sl_algorithm import BatchEnsembleSLAlgorithm
from alf.utils.datagen import load_cifar10

# dataset config
# train_loader, test_loader = load_cifar10(train_bs=50, test_bs=100)
train_loader, test_loader = load_cifar10(
    label_idx=[0, 1, 2, 3, 4, 5], train_bs=50, test_bs=100)
outlier_loaders = load_cifar10(
    label_idx=[6, 7, 8, 9], train_bs=100, test_bs=100)

# network architecture
CONV_LAYER_PARAMS = ((32, 3, 1, 0, 2), (64, 3, 1, 0, 2), (64, 3, 1, 0, 2))
FC_LAYER_PARAMS = (128, )

# optimizer config
parvi = 'svgd'
batch_size = train_loader.batch_size
# entropy_regularization = batch_size / len(train_loader.dataset)
# entropy_regularization = entropy_regularization / 10
entropy_regularization = 1.
random_seed = 3
full_ensemble_train = True

optimizer = alf.optimizers.Adam(
    lr=1e-3,
    weight_decay=1e-4,
    parvi=parvi,
    repulsive_weight=entropy_regularization)

# algorithm config
alf.config(
    'BatchEnsembleSLAlgorithm',
    train_loader=train_loader,
    test_loader=test_loader,
    outlier_loaders=outlier_loaders,
    output_dim=6,
    ensemble_size=10,
    conv_layer_params=CONV_LAYER_PARAMS,
    fc_layer_params=FC_LAYER_PARAMS,
    optimizer=optimizer,
    full_ensemble_train=full_ensemble_train,
    sl_type='classification',
    logging_training=True,
    logging_evaluate=True)

# training config
alf.config(
    'TrainerConfig',
    ml_type='sl',
    algorithm_ctor=BatchEnsembleSLAlgorithm,
    num_iterations=400,
    num_checkpoints=2,
    evaluate=True,
    eval_uncertainty=True,
    eval_interval=1,
    summary_interval=1,
    random_seed=random_seed,
    debug_summaries=True,
    summarize_grads_and_vars=True)
