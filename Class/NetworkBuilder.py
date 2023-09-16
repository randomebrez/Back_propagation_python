from Class import Network as net
import Tools.Computations as computer
from Class.Layers import Input as i_layer
from Class.Layers import Dense as d_layer
from Class.Layers import OneToOne as oto_layer
from Class.Layers import ConvDot as cd_layer
from Class.Layers import Convolution as c_layer
from Class.Layers import Flatten as f_layer


class NetworkBuilder:
    def __init__(self, input_shape, output_shape):
        self.wip_network = net.Network(input_shape, output_shape)
        self.wip_network.layers.append(i_layer.InputLayer(input_shape))
        self.activation_functions = {
            'relu': [computer.relu, computer.relu_with_derivative],
            'sigmoid': [computer.sigmoid, computer.sigmoid_with_derivative],
            'tan_h': [computer.tan_h, computer.tan_h_with_derivative]
        }
        self.operations = {
            'softmax': computer.softmax,
            'norm_2': computer.norm_2,
            '': None
        }

    def create_new(self, input_shape, output_shape):
        self.wip_network = net.Network(input_shape, output_shape)
        self.wip_network.layers.append(i_layer.InputLayer(input_shape))

    def build(self):
        previous_layer = self.wip_network.layers[0]
        for index, layer in enumerate(self.wip_network.layers[1:]):
            layer.initialize(previous_layer.output_shape)
            if layer.is_output_layer:
                self.wip_network.output_layer_index = index + 1  # since enumerate starts at second item
            previous_layer = layer

        return self.wip_network

    def add_dense_layer(self, layer_size: int, activation_function: str, is_output_layer=False, use_bias=True, normalization_function=''):
        activations = self.activation_functions[activation_function]
        layer = d_layer.DenseLayer(layer_size, activations[0], activations[1], is_output_layer, use_bias, self.operations[normalization_function])
        self.wip_network.layers.append(layer)

    def add_one_to_one_layer(self, operation: str, is_output_layer=False):
        layer = oto_layer.OneToOneLayer(self.operations[operation], is_output_layer)
        self.wip_network.layers.append(layer)

    def add_conv_dot_layer(self, filter_number: int, kernel_size: int, stride: int, activation_function: str, is_output_layer=False, use_bias=True, normalization_function=''):
        activations = self.activation_functions[activation_function]
        layer = cd_layer.ConvolutionDotLayer(filter_number, kernel_size, stride, activations[0], activations[1], is_output_layer, use_bias, self.operations[normalization_function])
        self.wip_network.layers.append(layer)

    def add_conv_fft_layer(self, filter_number: int, kernel_size: int, stride: int, activation_function: str, is_output_layer=False, use_bias=True, normalization_function=''):
        activations = self.activation_functions[activation_function]
        layer = c_layer.ConvolutionFFTLayer(filter_number, kernel_size, stride, activations[0], activations[1], is_output_layer, use_bias, self.operations[normalization_function])
        self.wip_network.layers.append(layer)

    def add_flat_layer(self, is_output_layer=False):
        layer = f_layer.FlatLayer(is_output_layer)
        self.wip_network.layers.append(layer)