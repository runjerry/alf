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
"""Classes for storing data for sampling."""

import tensorflow as tf

from tf_agents.utils import common as tfa_common
from alf.utils.nest_utils import get_nest_batch_size


class DataBuffer(tf.Module):
    """A simple circular buffer supporting random sampling.
    """

    def __init__(self, data_spec: tf.TensorSpec, capacity, name="DataBuffer"):
        """Create a DataBuffer.

        Args:
            data_spec (nested TensorSpec): spec for the data item (without batch
                dimension) to be stored.
            capacity (int): capacity of the buffer.
            name (str): name of the buffer
        """
        super().__init__()
        self._capacity = capacity

        def _create_buffer(tensor_spec):
            shape = [capacity] + tensor_spec.shape.as_list()
            return tfa_common.create_variable(
                name=name + "/buffer",
                initializer=tf.zeros(shape, dtype=tensor_spec.dtype),
                dtype=tensor_spec.dtype,
                shape=shape,
                trainable=False)

        self._buffer = tf.nest.map_structure(_create_buffer, data_spec)
        self._current_size = tfa_common.create_variable(
            name=name + "/size",
            initializer=0,
            dtype=tf.int32,
            shape=(),
            trainable=False)
        self._current_pos = tfa_common.create_variable(
            name=name + "/pos",
            initializer=0,
            dtype=tf.int32,
            shape=(),
            trainable=False)

    @property
    def current_size(self):
        return self._current_size

    def add_batch(self, batch):
        """Add a batch of items to the buffer.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
        """
        batch_size = get_nest_batch_size(batch, tf.int32)
        n = tf.minimum(batch_size, self._capacity)
        indices = tf.range(self._current_pos,
                           self._current_pos + n) % self._capacity
        indices = tf.expand_dims(indices, axis=-1)
        tf.nest.map_structure(
            lambda buf, bat: buf.scatter_nd_update(indices, bat[-n:]),
            self._buffer, batch)

        self._current_pos.assign((self._current_pos + n) % self._capacity)
        self._current_size.assign(
            tf.minimum(self._current_size + n, self._capacity))

    def get_batch(self, batch_size):
        """Get batsh_size random samples in the buffer.

        Args:
            batch_size (int): batch size
        Returns:
            Tensor of shape [batch_size] + tensor_spec.shape
        """
        indices = tf.random.uniform(
            shape=(batch_size, ),
            dtype=tf.int32,
            minval=0,
            maxval=self._current_size)
        return self.get_batch_by_indices(indices)

    def get_batch_by_indices(self, indices):
        return tf.nest.map_structure(
            lambda buffer: tf.gather(buffer, indices, axis=0), self._buffer)