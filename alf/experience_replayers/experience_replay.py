# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""
NOTE: The APIs in this file are subject to changes when we implement generic replay
buffers for off-policy drivers in the future.
"""

import six
import abc
import tensorflow as tf
import gin.tf

from tf_agents.utils.nest_utils import get_outer_rank

from alf.utils.common import flatten_once
from alf.experience_replayers.replay_buffer import ReplayBuffer

from alf.utils import common, nest_utils


@six.add_metaclass(abc.ABCMeta)
class ExperienceReplayer(object):
    """
    Base class for implementing experience storing and replay. A subclass should
    implement the abstract functions. This class object will be used by OffPolicyDrivers
    after training data are accumulated for a certain amount of time.
    """

    @abc.abstractmethod
    def observe(self, exp):
        """
        Observe a batch of `exp`, potentially storing it to replay buffers.

        Args:
            exp (Experience): each item has the shape of (`num_envs`, `env_batch_size`,
                `unroll_length`, ...), where `num_envs` is the number of tf_agents
                *batched* environments, each of which contains `env_batch_size`
                independent & parallel single environments.
        """

    @abc.abstractmethod
    def replay(self, sample_batch_size, mini_batch_length):
        """Replay experiences from buffers

        Args:
            sample_batch_size (int): A batch size to specify the number of items to
                return. A batch of `sample_batch_size` items is returned, where each
                tensor in items will have its first dimension equal to sample_batch_size
                and the rest of the dimensions match the corresponding data_spec.
            mini_batch_length (int): the temporal length of each sample

        Output:
            exp (Experience): each item has the shape (`sample_batch_size`,
                `mini_batch_length`, ...)
        """

    @abc.abstractmethod
    def replay_all(self):
        """Replay all experiences

        Output:
            exp (Experience): each item has the shape (`full_batch_size`,
                `buffer_length`, ...)
        """

    @abc.abstractmethod
    def clear(self):
        """Clear all buffers."""

    @abc.abstractmethod
    def batch_size(self):
        """
        Return the buffer's batch_size, assuming all buffers having the same
        batch_size
        """


@gin.configurable
class OnetimeExperienceReplayer(ExperienceReplayer):
    """
    A simple one-time experience replayer. For each incoming `exp`,
    it stores it with a temporary variable which is used for training
    only once.

    Example algorithms: IMPALA, PPO2

    NOTE: this replayer can only be run in the eager mode, because
    self._experience is updated by python assignment
    """

    def __init__(self):
        self._experience = None
        self._batch_size = None

    def observe(self, exp):
        # The shape is [learn_queue_cap, unroll_length, env_batch_size, ...]
        exp = tf.nest.map_structure(lambda e: common.transpose2(e, 1, 2), exp)
        # flatten the shape (num_envs, env_batch_size)
        self._experience = tf.nest.map_structure(flatten_once, exp)
        if self._batch_size is None:
            self._batch_size = self._experience.step_type.shape[0]

    def replay(self, sample_batch_size, mini_batch_length):
        """Get a random batch.

        Args:
            sample_batch_size (int): number of sequences
            mini_batch_length (int): the length of each sequence
        Returns:
            Experience: experience batch in batch major (B, T, ...)
            tf_uniform_replay_buffer.BufferInfo: information about the batch
        """
        raise NotImplementedError()  # Only supports replaying all!

    def replay_all(self):
        return self._experience

    def clear(self):
        self._experience = None

    @property
    def batch_size(self):
        assert self._batch_size, "No experience is observed yet!"
        return self._batch_size


@gin.configurable
class SyncUniformExperienceReplayer(ExperienceReplayer):
    """
    For synchronous off-policy training.

    Example algorithms: DDPG, SAC
    """

    def __init__(self, experience_spec, batch_size):
        self._experience_spec = experience_spec
        self._buffer = ReplayBuffer(experience_spec, batch_size)
        self._data_iter = None

    @tf.function
    def observe(self, exp):
        """
        For the sync driver, `exp` has the shape (`env_batch_size`, ...)
        with `num_envs`==1 and `unroll_length`==1.
        """
        outer_rank = get_outer_rank(exp, self._experience_spec)

        if outer_rank == 1:
            self._buffer.add_batch(exp, exp.env_id)
        elif outer_rank == 3:
            # The shape is [learn_queue_cap, unroll_length, env_batch_size, ...]
            for q in tf.range(tf.shape(exp.step_type)[0]):
                for t in tf.range(tf.shape(exp.step_type)[1]):
                    bat = tf.nest.map_structure(lambda x: x[q, t, ...], exp)
                    self._buffer.add_batch(bat, bat.env_id)
        else:
            raise ValueError("Unsupported outer rank %s of `exp`" % outer_rank)

    def replay(self, sample_batch_size, mini_batch_length):
        """Get a random batch.

        Args:
            sample_batch_size (int): number of sequences
            mini_batch_length (int): the length of each sequence
        Returns:
            Experience: experience batch in batch major (B, T, ...)
        """
        return self._buffer.get_batch(sample_batch_size, mini_batch_length)

    def replay_all(self):
        return self._buffer.gather_all()

    def clear(self):
        self._buffer.clear()

    @property
    def batch_size(self):
        return self._buffer.num_environments
