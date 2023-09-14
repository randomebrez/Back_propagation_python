import numpy as np
from Class.Layers import LayerBase
from scipy.signal import fftconvolve


class ConvolutionLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True, normalization_function=None):
        self.filters = []
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': kernel_size, 'stride': stride, 'zero_padding': (0, 0), 'use_bias': use_bias}
        self.biases = []
        self.output_shape = (0, 0, 0)
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        self.normalization = normalization_function
        super().__init__('convolution', is_output_layer)

    def initialize(self, input_shape):
        self.parameters['input_shape'] = input_shape
        self.parameters['zero_padding'] = self.padding_and_output_shape_compute()

        kernel_size = self.parameters['kernel_size']
        filter_number = self.parameters['filter_number']
        depth = input_shape[2]

        self.filters = 0.01 * np.random.rand(filter_number, kernel_size, kernel_size, depth)

        if self.parameters['use_bias']:
            self.biases = np.random.rand(self.parameters['filter_number'])

        self.init_cache()

    def init_cache(self):
        self.cache['sigma_primes'] = []
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        feature_maps = np.zeros(self.output_shape)
        for index, feature in enumerate(self.filters):
            feature_maps[index] = self.convolve_filter(index, inputs)

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(feature_maps)
            self.cache['sigma_primes'] = sigma_primes
        else:
            activations = self.activation_function(feature_maps)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def convolve_filter(self, filter_index, single_input):
        feature = self.filters[filter_index]
        stride = self.parameters['stride']
        kernel_size = self.parameters['kernel_size']
        (padding_x, padding_y) = self.parameters['zero_padding']
        (input_shape_x, input_shape_y, _) = self.parameters['input_shape']

        # Convolve
        conv_res = fftconvolve(feature, single_input)

        # Truncate according to padding
        x_min = kernel_size - (padding_x - 1)
        x_max = np.shape(conv_res)[0] - x_min
        y_min = kernel_size - (padding_y - 1)
        y_max = np.shape(conv_res)[1] - y_min
        truncated_conv_res = conv_res[x_min:x_max, y_min:y_max, :]

        # Slice according to stride
        sliced_conv_res = truncated_conv_res[::stride, ::stride, :]
        # Apply bias and Save feature map
        return sliced_conv_res + self.biases[filter_index]

    def padding_and_output_shape_compute(self):
        input_shape = self.parameters['input_shape']
        kernel_size = self.parameters['kernel_size']
        stride = self.parameters['stride']

        input_shape_x =  input_shape[0]
        input_shape_y = input_shape[1]

        x_jump = int((input_shape_x - (kernel_size - 1)) / stride) + 1
        x_padding_needed = (kernel_size - 1) + stride * x_jump - input_shape_x
        x_padding_each_side = int((x_padding_needed + 1) / 2)

        if input_shape_x != input_shape_y:
            y_jump = int((input_shape_y - (kernel_size - 1)) / stride) + 1
            y_padding_needed = (kernel_size - 1) + stride * y_jump - input_shape_y
            y_padding_each_side = int((y_padding_needed + 1) / 2)

            self.output_shape = (x_jump + 1, y_jump + 1, self.parameters['filter_number'])

            return x_padding_each_side, y_padding_each_side
        else:
            self.output_shape = (x_jump + 1, x_jump + 1, self.parameters['filter_number'])

            return x_padding_each_side, x_padding_each_side
