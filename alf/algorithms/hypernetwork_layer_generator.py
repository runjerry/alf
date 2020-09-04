# Neale Ratzlaff
# Hypernetwork_layer_generator.py
# 
"""Generator and Mixer definitions for Hypernetworks
"""
from absl import logging
import gin

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.summary_utils import record_time
from alf.networks import EncodingNetwork, Network


@gin.configurable
class ParamLayers(Algorithm):
    def __init__(
            self,
            noise_dim,
            particles,
            input_tensor_spec,
            conv_layer_params,
            fc_layer_params,
            last_layer_param,
            last_activation,
            use_bias=False,
            hidden_layers=(100, 100),
            activation=torch.nn.ReLU,
            use_fc_bn=False,
            optimizer=None,
            name="LayerGenerator"):
        """
        Layer-wise generator for linear function parameters. 
            the linear function could be any set of parameters that
            can be expressed as a torch layer. We generate the parameters for
            each layer using a generator network, and we evaluate the
            generated parameters on incoming data, by using the parameters in a
            torch.nn.functional function. ParamLayers outputs parameters
            for a weight matrix of size (W_in, W_out). 
        Args:
            noise_dim (int): the dimensionality of the generator latent space
                [if using the mixer]: dimensionality of mixer output
                [if not using the mixer]: dimensionality of random sample    
                (default 256). 
            particles (int):
            input_tensor_spec (TensorSpec):
            conv_layer_params (tuple[tuple]): a tuple of tuples where each 
                tuple takes the format 
                ``(filters, kernel_size, strides, padding, pooling_kerel)``,
                where ``padding`` and ``pooling_kernel`` are optional. 
            fc_layer_params (tuple[int]): a tuple of integers representing
                FC layer sizes
            last_layer_param (tuple): an optional tuple of ``(size, use_bias)``
                appending at the very end. Note that if ``last activation``
                is specified, ``last_layer_size`` has to be specified 
                explicitly.
            last_activation (torch.nn.functional): activation function of
                the additional layer specified by ``last layer size``.
                Note that if ``last layer size`` is not None, 
                ``last_activation`` has to be specified explicitly.
            use_bias (bool): controls the use of a generated bias term in conv
                layers, by default is False. 
            hidden_layers (tuple[int]): a tuple of Ints, representing the width
                of each hidden layer in the generator (default (100, 100) 
                representing 2 hidden layers of width 100
            activation (torch.nn function): function specifying the activation
                function used in the generator hidden layers. Must be an 
                instance of torch.nn, (default torch.nn.ReLU)
            use_fc_bn (bool): turns on or off the use of BatchNorm in
                the hidden layers of the layer generators. 
            optimizer (torch.optim.Optimizer): the optimizer for training.
        """       

        self._noise_dim = noise_dim
        self._particles = particles
        self._noise_spec = TensorSpec(shape=(noise_dim,))
        self._input_tensor_spec = input_tensor_spec
        self._last_layer_size = last_layer_param
        self._last_activation = last_activation
        self._use_bias = use_bias

        self._hidden_layers = hidden_layers
        self._use_fc_bn = use_fc_bn

        assert (callable(activation)
                or activation is None), ("Activation must be an callable "\
                        "function or None, got {}".format(type(activation)))

        self._activation = activation
        
        if conv_layer_params is None:
            self._conv_layer_params = None
            assert len(self._input_tensor_spec.shape) == 1, "Without conv "\
                "layers, input shape must be [N], not {}".format(
                    self._input_tensor_spec.shape)
        else:
            self._conv_layer_params = self._convert_inputs_conv(conv_layer_params)
            assert len(self._input_tensor_spec.shape) >= 3, "If using "\
                "conv layers, input shape must be [C, H, W], not {}".format(
                    self._input_tensor_spec.shape)
        if fc_layer_params is None:
            self._fc_layer_params = None
        else:
            self._fc_layer_params = self._convert_inputs_fc(fc_layer_params)
        self._network = []
        
        super(ParamLayers, self).__init__(
                optimizer=optimizer,
                name=name)
        
        layer_size = []
        input_size = self._input_tensor_spec.shape
        if conv_layer_params is not None:
            for i, layer in enumerate(self._conv_layer_params):
                input_w, output_w, kernel = layer[:3]
                size = input_w * output_w * kernel * kernel
                if self._use_bias:
                    size += output_w
                layer_size.append(size)
                
                # conv output = (W - K + 2P) / S + 1
                # input size is always (c x h x w), higher order conv not supported
                in_d = input_size[-1]
                kernel = layer[2]
                stride = layer[3]
                if len(layer) > 4: # calculate output with padding
                    padding = layer[4]
                else:
                    padding = 0
                running_output = int((in_d - kernel + 2 * padding) / stride + 1)
                if len(layer) > 5: # calculate output with pool
                    pool = layer[5]
                    running_output = int(math.floor(running_output / pool))
                input_size = (input_size[0], running_output, running_output)

        if fc_layer_params is not None:    
            for i, layer in enumerate(self._fc_layer_params):
                input_w, output_w, bias = layer
                bias_dim = output_w if bias else 0
                if i == 0: # first FC layer or first FC layer after conv
                    if conv_layer_params is not None:
                        input_in = input_size[-2] * input_size[-1] * input_w
                    else:
                        input_in = input_size[0]
                    size = input_in * output_w + bias_dim
                else:
                    size = input_w * output_w + bias_dim
                layer_size.append(size)
                input_size = (output_w,)
        # Handle last layer
        if last_layer_param is not None:
            input_dim = torch.prod(torch.tensor(input_size)).item()
            last_layer_w, last_layer_bias = self._last_layer_size
            bias = last_layer_w if last_layer_bias else 0
            size = input_dim * last_layer_w + bias
            layer_size.append(size)
        
        print (layer_size, sum(layer_size))
        self.layer_encoders = nn.ModuleList([EncodingNetwork(
            self._noise_spec,
            fc_layer_params=self._hidden_layers,
            use_fc_bn=self._use_fc_bn,
            last_layer_size=layer_size[i],
            last_activation=self._last_activation,
            name="LayerEncoder_{}".format(i)) for i in range(len(layer_size))])

    def _convert_inputs_conv(self, inputs_conv):
        """ Helper fn. Converts the default input conv layer description from:
            (filters, kernel_size, ...), to
            (in_filters, out_filters, kernel_size, ...)
        """
        new_inputs_conv = []
        data_in_channels = self._input_tensor_spec.shape[0]
        for i, layer in enumerate(inputs_conv):
            if i == 0:
                in_channels = data_in_channels
            extended_layer = (in_channels,) + layer
            in_channels = layer[0]
            new_inputs_conv.append(extended_layer)
        return tuple(new_inputs_conv)

    def _convert_inputs_fc(self, inputs_fc):
        """ Helper fn. Converts the default input fc layer description from:
            (output_width), to
            (input_wdth, output_width)
        """
        new_inputs_fc = []
        if self._conv_layer_params is not None:
            features_output = self._conv_layer_params[-1][1]
        else:
            features_output = self._input_tensor_spec.shape[0]
        for i, layer in enumerate(inputs_fc):
            if i == 0:
                in_width = features_output
            extended_layer = (in_width, *layer)
            new_inputs_fc.append(extended_layer)
            in_width = extended_layer[1]
        return tuple(new_inputs_fc)
    
    def print_hypernetwork_layers(self):
        for layer in self._network:
            print (layer)
    

    def forward(self, inputs=None, training=True):
        """ HyperNetwork Generator Core
        inputs:
            x (torch.tensor) [N, z_dim]:
                N : number of particles (number of generated weight matrices)
                if using the mixer, `x` is a `z_dim` dimensional output
                of the mixer, otherwise `x` is a `z_dim` dimensional sample
                from a standard normal distribution
        returns:
            f(x), weight matrix of size [N, input_dim, output_dim]
        """
        thetas = []
        if inputs is None:
            inputs = torch.randn(self._particles, self._noise_dim)
        for i in range(len(self.layer_encoders)):
            theta = self.layer_encoders[i](inputs=inputs)
            thetas.append(theta[0])
        return torch.cat(thetas, dim=-1), inputs
        
