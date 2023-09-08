import numpy as np

from Class import Network as net
import Tools.Computations as computer
from Class.Layers import Input as i_layer
from Class.Layers import Dense as d_layer
from Class.Layers import OneToOne as oto_layer


class NetworkBuilder:
    def __init__(self, input_size, output_size):
        self.wip_network = net.Network(input_size, output_size)
        self.wip_network.layers.append(i_layer.InputLayer(input_size))

    def build(self):
        layers = self.wip_network.layers
        previous_layer = layers[0]
        for index, layer in enumerate(self.wip_network.layers):
            layer.initialize(previous_layer.layer_size)
            if layer.is_output_layer:
                self.wip_network.output_layer_index = index
            previous_layer = layer

        return self.wip_network

    def build_dense_network(self, hidden_layer_sizes):
        for layer_size in hidden_layer_sizes:
            self.add_dense_layer(layer_size, computer.tan_h, computer.tan_h_with_derivative)
        # Output layer
        self.add_dense_layer(self.wip_network.output_size, computer.sigmoid, computer.sigmoid_with_derivative, is_output_layer=True, use_bias=False)
        return self.build()

    def build_dense_network_softmax_output(self, hidden_layer_sizes):
        for layer_size in hidden_layer_sizes:
            # self.add_dense_layer(layer_size, computer.tan_h, computer.tan_h_with_derivative)
            self.add_dense_layer(layer_size, computer.relu, computer.relu_with_derivative)
        # Output layer bloc (dense + OneToOne softmax)
        self.add_dense_layer(self.wip_network.output_size, computer.sigmoid, computer.sigmoid_with_derivative, is_output_layer=True, use_bias=False)
        self.add_one_to_one_layer(self.wip_network.output_size, computer.softmax)
        return self.build()

    def add_dense_layer(self, layer_size, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True):
        layer = d_layer.DenseLayer(layer_size, activation_function, activation_function_with_derivative, is_output_layer, use_bias)
        self.wip_network.layers.append(layer)

    def add_one_to_one_layer(self, layer_size, activation_function, is_output_layer=False):
        layer = oto_layer.OneToOneLayer(layer_size, activation_function, is_output_layer)
        self.wip_network.layers.append(layer)
